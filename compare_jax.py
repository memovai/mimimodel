"""Rebuild a checkpoint-layout params dict from .cact tensors and run the
official JAX decode path as ground truth; then diff against needle_np."""
import sys
import types
import numpy as np

sys.path.insert(0, "needle")

from needle_np import NeedleNP

m = NeedleNP("model/needle2.cact")
g, p = m.g, m.p
L, n, C = g["num_layers"], g["mhc_lanes"], g["d_model"]


def stack(key):
    return np.stack([p["layers"][i][key] for i in range(L)])


def stackT(key):
    return np.stack([p["layers"][i][key].T for i in range(L)])


params = {
    "embedding": {"embedding": p["embedding"]},
    "stack": {
        "layers": {"block": {
            "ZCRMSNorm_0": {"scale": stack("norm_in")},
            "self_attn": {
                "q_proj": {"kernel": stackT("q_proj")},
                "k_proj": {"kernel": stackT("k_proj")},
                "v_proj": {"kernel": stackT("v_proj")},
                "gate_proj": {"kernel": stackT("gate_proj")},
                "out_proj": {"kernel": stackT("out_proj")},
                "q_norm": {"scale": stack("q_norm")},
                "k_norm": {"scale": stack("k_norm")},
            },
            "hadamard_mlp": {"d1": stack("d1"), "d2": stack("d2"), "d3": stack("d3")},
            "post_attn_norm": {"scale": stack("post_norm")},
            "attn_gate": np.stack([p["layers"][i]["attn_gate"][0] for i in range(L)]),
            "pre_hada_norm": {"scale": stack("pre_hada")},
        }},
        "final_norm": {"scale": p["final_norm"]},
        "mhc_a_pre": p["mhc_a_pre"], "mhc_a_post": p["mhc_a_post"],
        "mhc_a_res": p["mhc_a_res"], "mhc_b_pre": p["mhc_b_pre"],
        "mhc_b_post": p["mhc_b_post"], "mhc_b_res": p["mhc_b_res"],
        "mhc_phi_pre": p["mhc_phi_pre"].reshape(L, n, C * n).transpose(0, 2, 1),
        "mhc_phi_post": p["mhc_phi_post"].reshape(L, n, C * n).transpose(0, 2, 1),
        "mhc_phi_res": p["mhc_phi_res"].reshape(L, n * n, C * n).transpose(0, 2, 1),
    },
}
for s in range(len(g["engram_layers"])):
    ep = p["engrams"][s]
    params[f"engrams_{s}"] = {
        "embedding": ep["tables"].reshape(g["num_engram_tables"], g["engram_slots"],
                                          g["engram_sub_dim"]),
        "key_proj": {"kernel": ep["key_proj"].T},
        "value_proj": {"kernel": ep["value_proj"].T},
        "taps": ep["taps"],
    }

config = types.SimpleNamespace(
    d_model=C, num_heads=g["num_heads"], num_kv_heads=g["num_kv_heads"],
    num_layers=L, mhc_lanes=n, engram_layers=tuple(g["engram_layers"]),
    engram_orders=tuple(g["engram_orders"]), engram_heads=g["num_engram_tables"] // 2,
    engram_slots=g["engram_slots"], max_seq_len=g["max_seq_len"],
    rope_theta=g["rope_theta"], attn_dim=0, kv_window=g["kv_window"],
)

from needle.model.decode import generate_cached, decode_cfg, init_kv_cache, \
    _forward_cached, precompute_rope_freqs, _attn_width
import jax.numpy as jnp

prompt = sys.argv[1] if len(sys.argv) > 1 else "What's the weather in Paris right now?"

# ---- ground truth text
text = generate_cached(config, params, m.tok, prompt, max_new_tokens=32,
                       kv_window=g["kv_window"])
print("JAX text:", repr(text))

# ---- per-position logits diff vs needle_np
ids = [m.tok.bos_id] + m.tok.encode(prompt)
dcfg = decode_cfg(config, kv_window=g["kv_window"])
max_len = len(ids) + 8
head_dim = _attn_width(config) // config.num_heads
cos, sin = precompute_rope_freqs(head_dim, max_len, config.rope_theta)
kc, vc = init_kv_cache(config, 1, max_len)
hist = jnp.zeros((1, max_len), jnp.int32)
valid = jnp.ones((1, max_len), bool)

m.reset(max_len)
jax_logits = None
for pos, t in enumerate(ids):
    hist = hist.at[0, pos].set(t)
    jl, kc, vc = _forward_cached(params, dcfg, jnp.asarray([[t]], jnp.int32), kc, vc,
                                 jnp.asarray(pos, jnp.int32), cos, sin, None, False,
                                 hist, valid, None)
    ml = m.step(t, pos)
    d = float(np.abs(np.asarray(jl[0, -1]) - ml).max())
    if pos < 3 or d > 1e-2:
        print(f"pos {pos:3d} tok {t:5d} max logit diff {d:.5f}  "
              f"jax_top {int(np.asarray(jl[0,-1]).argmax())} np_top {int(ml.argmax())}")
print("final top tokens: jax", int(np.asarray(jl[0, -1]).argmax()),
      "np", int(ml.argmax()))
