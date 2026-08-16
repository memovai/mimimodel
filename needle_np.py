"""Standalone numpy reference implementation of Needle 2 inference from a .cact blob.

No JAX, no torch — this file is the executable spec for the C port (needle.c).
Parsing/dequant follows needle/needle/model/export.py; forward follows
needle/needle/model/decode.py; helpers follow architecture.py. Kept deliberately
close to the C structure: single-token decode step, explicit KV cache, engram
token history.
"""
import json
import math
import struct
import sys

import numpy as np

TAG = 0x05E12A83
ALIGN = 64
FP16, FP32, CQ, RAW = 1, 2, 3, 4
TERNARY_RECORD_BITS = 5
_HDR_FMT = "<29If"
_REC_FMT = "<BBHIIIIQQII"
REC_SIZE = struct.calcsize(_REC_FMT)
_TK_HDR = "<IIIIIBBH"
_TK_REC = "<fBH"

_ENGRAM_SEED = 0x9E3779B9
_ENGRAM_PRIME = 0x01000193

TK_NORMAL, TK_UNKNOWN, TK_CONTROL, TK_USER_DEFINED, TK_BYTE = 0, 1, 2, 3, 4
_SP_META_SPACE = "▁"


# ---------------------------------------------------------------- dequant

def _walsh_matrix(n):
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def _unpack_lsb(packed, bits, in_pad):
    out = packed.shape[0]
    chunks = packed.reshape(out, in_pad // 8, bits).astype(np.uint64)
    word = np.zeros(chunks.shape[:-1], np.uint64)
    for b in range(bits):
        word |= chunks[..., b] << (8 * b)
    idx = np.empty((out, in_pad // 8, 8), np.uint8)
    mask = (1 << bits) - 1
    for i in range(8):
        idx[..., i] = (word >> (i * bits)) & mask
    return idx.reshape(out, in_pad)


def _cq_unpack(packed, norms, out, in_dim, bits, group, codebooks):
    # codebooks: dict bits -> np.array(2**bits) shared Lloyd-Max unit codebook
    if bits == TERNARY_RECORD_BITS:
        crumbs = _unpack_lsb(packed, 2, (in_dim + group - 1) // group * group)
        idx = np.where(crumbs == 3, 0, crumbs + 1).astype(np.uint8)
        c = 1.2240064 / math.sqrt(group)
        cb = np.array([-c, 0.0, c], np.float32)
    else:
        in_pad = (in_dim + group - 1) // group * group
        idx = _unpack_lsb(packed, bits, in_pad)
        cb = codebooks[bits]
    in_pad = idx.shape[1]
    H = _walsh_matrix(group)
    unit = cb[idx].reshape(out, in_pad // group, group)
    rot = unit * norms.reshape(out, in_pad // group, 1).astype(np.float32)
    w = (rot @ H).reshape(out, in_pad)
    return w[:, :in_dim]


# ---------------------------------------------------------------- .cact loader

def load_cact(path):
    with open(path, "rb") as f:
        raw = f.read()
    hdr = struct.unpack_from(_HDR_FMT, raw, 0)
    tag, num_tensors, cb_n, kv_window, kv_bits = hdr[:5]
    assert tag == TAG, f"bad magic {tag:#x}"
    g = dict(zip(("vocab_size", "d_model", "num_heads", "num_kv_heads", "num_layers",
                  "head_dim", "max_seq_len", "hada_n", "mhc_lanes", "engram_slots",
                  "engram_sub_dim", "num_engram_tables", "engram_conv_taps",
                  "engram_conv_dilation"), hdr[5:19]))
    num_orders, orders4 = hdr[19], hdr[20:24]
    num_sites, sites4 = hdr[24], hdr[25:29]
    g["engram_orders"] = tuple(orders4[:num_orders])
    g["engram_layers"] = tuple(sites4[:num_sites])
    g["rope_theta"] = hdr[29]
    g["kv_window"] = kv_window
    g["kv_bits"] = kv_bits

    off = struct.calcsize(_HDR_FMT)
    cb_all = np.frombuffer(raw[off:off + cb_n * 4], np.float32)
    # cb2[4] | cb3[8] | cb4[16]
    codebooks = {2: cb_all[0:4], 3: cb_all[4:12], 4: cb_all[12:28]}
    off += cb_n * 4

    tensors = []
    for _ in range(num_tensors):
        rec = struct.unpack(_REC_FMT, raw[off:off + REC_SIZE]); off += REC_SIZE
        dtype, ndim = rec[0], rec[1]
        shape = tuple(rec[3:3 + ndim])
        offset, nbytes, group, bits = rec[7], rec[8], rec[9], rec[10]
        blob = raw[offset:offset + nbytes]
        if dtype == FP16:
            tensors.append(np.frombuffer(blob, np.float16).reshape(shape).astype(np.float32))
        elif dtype == FP32:
            tensors.append(np.frombuffer(blob, np.float32).reshape(shape).copy())
        elif dtype == CQ:
            out, in_dim = shape
            in_pad = (in_dim + group - 1) // group * group
            row_bytes = in_pad * 2 // 8 if bits == TERNARY_RECORD_BITS else in_pad * bits // 8
            n_packed = out * row_bytes
            packed = np.frombuffer(blob[:n_packed], np.uint8).reshape(out, -1)
            norms = np.frombuffer(blob[n_packed:], np.float16).reshape(out, in_pad // group)
            tensors.append(_cq_unpack(packed, norms, out, in_dim, bits, group, codebooks))
        elif dtype == RAW:
            tensors.append(bytes(blob))
        else:
            raise ValueError(f"unknown dtype {dtype}")
    return g, tensors


def name_tensors(g, tensors):
    """Assign the canonical positional names from export.py's TENSOR ORDER."""
    t = iter(tensors)
    p = {"embedding": next(t)}
    layers = []
    for _ in range(g["num_layers"]):
        layers.append({k: next(t) for k in
                       ("norm_in", "q_proj", "k_proj", "v_proj", "q_norm", "k_norm",
                        "gate_proj", "out_proj", "post_norm", "attn_gate", "pre_hada",
                        "d1", "d2", "d3")})
    p["layers"] = layers
    for k in ("mhc_a_pre", "mhc_a_post", "mhc_a_res", "mhc_b_pre", "mhc_b_post",
              "mhc_b_res", "mhc_phi_pre", "mhc_phi_post", "mhc_phi_res"):
        p[k] = next(t)
    sites = []
    for _ in range(len(g["engram_layers"])):
        sites.append({k: next(t) for k in ("tables", "key_proj", "value_proj", "taps")})
    p["engrams"] = sites
    p["final_norm"] = next(t)
    rest = list(t)
    raws = [r for r in rest if isinstance(r, (bytes, bytearray))]
    heads = [r for r in rest if not isinstance(r, (bytes, bytearray))]
    p["tokenizer_blob"] = raws[0] if raws else None
    if heads:
        manifest = heads[0].astype(np.int32).ravel().tolist()
        p["heads"] = {}
        names = {1: "contrastive", 2: "confidence"}
        for hi, code in enumerate(manifest):
            probes, proj, bias = heads[1 + hi * 3: 1 + hi * 3 + 3]
            p["heads"][names.get(code, str(code))] = (probes, proj, bias)
    return p


# ---------------------------------------------------------------- tokenizer

class RefTokenizer:
    def __init__(self, blob):
        n, pad, eos, bos, unk, add_dummy, byte_fb, _ = struct.unpack_from(_TK_HDR, blob, 0)
        off = struct.calcsize(_TK_HDR)
        rec = struct.calcsize(_TK_REC)
        self.pieces, self.scores, self.types = [], [], []
        for _ in range(n):
            score, t, ln = struct.unpack_from(_TK_REC, blob, off)
            off += rec
            self.pieces.append(blob[off:off + ln].decode("utf-8"))
            self.scores.append(score)
            self.types.append(t)
            off += ln
        self.pad_id, self.eos_id, self.bos_id, self.unk_id = pad, eos, bos, unk
        self.add_dummy = bool(add_dummy)
        self.byte_fallback = bool(byte_fb)
        self.p2id = {p: i for i, p in enumerate(self.pieces)}
        self.byte_id = {int(p[3:5], 16): i
                        for i, (p, t) in enumerate(zip(self.pieces, self.types))
                        if t == TK_BYTE}
        self.markers = sorted((p for p, t in zip(self.pieces, self.types)
                               if t == TK_USER_DEFINED), key=len, reverse=True)

    def _bpe(self, seg):
        syms = list(seg)
        while len(syms) > 1:
            best_score, best_j = None, -1
            for j in range(len(syms) - 1):
                idx = self.p2id.get(syms[j] + syms[j + 1])
                if idx is not None and (best_score is None or self.scores[idx] > best_score):
                    best_score, best_j = self.scores[idx], j
            if best_j < 0:
                break
            syms[best_j:best_j + 2] = [syms[best_j] + syms[best_j + 1]]
        ids = []
        for s in syms:
            idx = self.p2id.get(s)
            if idx is not None:
                ids.append(idx)
            elif self.byte_fallback:
                ids.extend(self.byte_id[b] for b in s.encode("utf-8"))
            else:
                ids.append(self.unk_id)
        return ids

    def encode(self, text):
        if not text:
            return []
        esc = text.replace(" ", _SP_META_SPACE)
        if self.add_dummy:
            esc = _SP_META_SPACE + esc
        ids, buf, i, n = [], [], 0, len(esc)
        while i < n:
            marker = next((m for m in self.markers if esc.startswith(m, i)), None)
            if marker is not None:
                ids += self._bpe("".join(buf)); buf = []
                ids.append(self.p2id[marker])
                i += len(marker)
            else:
                buf.append(esc[i]); i += 1
        ids += self._bpe("".join(buf))
        return ids

    def decode(self, ids):
        buf = bytearray()
        for i in ids:
            t = self.types[i]
            if t == TK_BYTE:
                buf.append(int(self.pieces[i][3:5], 16))
            elif t in (TK_CONTROL, TK_UNKNOWN):
                continue
            else:
                buf += self.pieces[i].encode("utf-8")
        text = buf.decode("utf-8", "replace").replace(_SP_META_SPACE, " ")
        if self.add_dummy and text.startswith(" "):
            text = text[1:]
        return text


# ---------------------------------------------------------------- math helpers

def _zcrms(x, scale, eps=1e-6):
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (1.0 + scale) * x / rms


def _rms_unit(x, eps=1e-6):
    return x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _silu(x):
    return x * _sigmoid(x)


def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def _sinkhorn(logits, iters=20):
    log_K = logits.astype(np.float64)
    for _ in range(iters):
        m = log_K.max(axis=-1, keepdims=True)
        log_K = log_K - (np.log(np.exp(log_K - m).sum(-1, keepdims=True)) + m)
        m = log_K.max(axis=-2, keepdims=True)
        log_K = log_K - (np.log(np.exp(log_K - m).sum(-2, keepdims=True)) + m)
    return np.exp(log_K).astype(np.float32)


def precompute_rope(head_dim, seq_len, theta):
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    t = np.arange(seq_len, dtype=np.float32)
    ang = np.outer(t, freqs)
    return np.cos(ang), np.sin(ang)


def apply_rope_1(x, cos_p, sin_p):
    """x: (H, hd) single position; cos_p/sin_p: (hd/2,)"""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos_p - x2 * sin_p, x2 * cos_p + x1 * sin_p], axis=-1)


# ---------------------------------------------------------------- engram

def engram_indices_1(hist_window, orders, heads, slots):
    """hist_window: last max(orders) token ids ending at current pos (list of ints,
    oldest first, padded with 0 + valid flags). Returns idx (num_tables,) and
    ngram_ok (num_tables,) for the CURRENT position only."""
    idx, ok = [], []
    W = len(hist_window)
    for oi, order in enumerate(orders):
        for h in range(heads):
            seed = (_ENGRAM_SEED * (oi * heads + h + 1)) & 0xFFFFFFFF
            acc = np.uint32(seed)
            valid = True
            for j in range(order):
                # token at current pos shifted right by j = hist[-1-j]
                if W - 1 - j >= 0:
                    tok, v = hist_window[W - 1 - j]
                else:
                    tok, v = 0, False
                acc = np.uint32((int(acc) ^ int(tok)) * _ENGRAM_PRIME & 0xFFFFFFFF)
                if j == order - 1:
                    valid = v  # _shift_right(win_valid, order-1) at this pos
            acc = np.uint32(int(acc) ^ (int(acc) >> 15))
            idx.append(int(acc) % slots)
            ok.append(valid)
    return np.array(idx, np.int64), np.array(ok, np.float32)


class EngramState:
    """Per-site ring of past v vectors so the causal conv taps see history."""
    def __init__(self, n_sites, d_model, taps, stride):
        self.taps = taps
        self.stride = stride
        depth = (taps - 1) * stride + 1
        self.buf = np.zeros((n_sites, depth, d_model), np.float32)
        self.valid = np.zeros((n_sites, depth), bool)
        self.pos = 0

    def push_and_mix(self, site, v_now, tapw):
        depth = self.buf.shape[1]
        slot = self.pos % depth
        self.buf[site, slot] = v_now
        self.valid[site, slot] = True
        out = np.zeros_like(v_now)
        for j in range(self.taps):
            back = j * self.stride
            p = self.pos - back
            if p < 0:
                continue
            s = p % depth
            if not self.valid[site, s]:
                continue
            out += tapw[j] * self.buf[site, s]
        return out


# ---------------------------------------------------------------- model

class NeedleNP:
    def __init__(self, path):
        self.g, tensors = load_cact(path)
        self.p = name_tensors(self.g, tensors)
        self.tok = RefTokenizer(self.p["tokenizer_blob"])
        g = self.g
        self.n_lanes = g["mhc_lanes"]
        self.orders = g["engram_orders"]
        self.heads_per_order = g["num_engram_tables"] // len(self.orders)
        self.sub_dim = g["engram_sub_dim"]
        L, n = g["num_layers"], self.n_lanes
        lane = np.eye(n, dtype=np.float32)[np.arange(L) % n]
        self.pre_off = 8 * lane - 4          # (L, n)
        self.post_off = -4 * (1 - lane)      # (L, n)
        self.max_orders = max(self.orders)

    def reset(self, max_len):
        g = self.g
        hd = g["head_dim"]
        self.max_len = max_len
        self.k_cache = np.zeros((g["num_layers"], g["num_kv_heads"], max_len, hd), np.float32)
        self.v_cache = np.zeros_like(self.k_cache)
        self.cos, self.sin = precompute_rope(hd, max_len, g["rope_theta"])
        self.hist = []            # full token history (engram hashing)
        self.estate = EngramState(len(g["engram_layers"]), g["d_model"],
                                  g["engram_conv_taps"], self.max_orders)

    def _engram_kv_now(self):
        """k,v per site for the current position (uses self.hist incl. current tok)."""
        g, p = self.g, self.p
        W = self.max_orders
        win = [(self.hist[i], True) if i >= 0 else (0, False)
               for i in range(len(self.hist) - W, len(self.hist))]
        idx, ok = engram_indices_1(win, self.orders, self.heads_per_order,
                                   g["engram_slots"])
        ks, vs = [], []
        for site, ep in enumerate(p["engrams"]):
            tables = ep["tables"]   # (num_tables*slots, sub_dim) dequantized
            slots = g["engram_slots"]
            fetched = np.stack([tables[t * slots + idx[t]] * ok[t]
                                for t in range(len(idx))])      # (T, sub)
            e = fetched.reshape(-1)                              # (d_model,)
            k = ep["key_proj"] @ e                               # [out,in] @ in
            v_now = ep["value_proj"] @ e
            v = self.estate.push_and_mix(site, v_now, ep["taps"])  # taps (n_taps, C)
            ks.append(k); vs.append(v)
        self.estate.pos += 1
        return ks, vs

    def step(self, token, pos):
        """Single-token decode step. Returns logits (vocab,)."""
        g, p = self.g, self.p
        n, C = self.n_lanes, g["d_model"]
        H, KV, hd = g["num_heads"], g["num_kv_heads"], g["head_dim"]
        kv_window = g["kv_window"]

        self.hist.append(int(token))
        emb_row = p["embedding"][token]                    # (C,) dequantized
        x0 = emb_row * math.sqrt(C)
        x = np.tile(x0, (n, 1))                            # (n, C) lanes

        eks, evs = self._engram_kv_now() if g["engram_layers"] else ([], [])
        cos_p, sin_p = self.cos[pos], self.sin[pos]

        for i in range(g["num_layers"]):
            lp = p["layers"][i]
            nx = _rms_unit(x.reshape(n * C))               # (n*C,)
            # phi_pre: stored (L, lanes, nC) after export transpose; row-major [out,in]
            hpre = _sigmoid(p["mhc_a_pre"][i] * (self._phi(p["mhc_phi_pre"], i) @ nx)
                            + p["mhc_b_pre"][i] + self.pre_off[i])   # (n,)
            u = hpre @ x                                   # (C,)
            bx = u
            if i in g["engram_layers"]:
                site = g["engram_layers"].index(i)
                alpha = _sigmoid(float(_rms_unit(u) @ _rms_unit(eks[site])) / math.sqrt(C))
                bx = u + alpha * evs[site]

            # ---- attention block (single position) ----
            skip = bx
            h = _zcrms(bx, lp["norm_in"])
            q = (lp["q_proj"] @ h).reshape(H, hd)
            k = (lp["k_proj"] @ h).reshape(KV, hd)
            v = (lp["v_proj"] @ h).reshape(KV, hd)
            q = _zcrms(q, lp["q_norm"]); k = _zcrms(k, lp["k_norm"])
            q = apply_rope_1(q, cos_p, sin_p)
            k = apply_rope_1(k, cos_p, sin_p)
            self.k_cache[i, :, pos] = k
            self.v_cache[i, :, pos] = v
            lo = 0
            if kv_window:
                lo = max(0, pos + 1 - kv_window)
            reps = H // KV
            outh = np.empty((H, hd), np.float32)
            for kvh in range(KV):
                keys = self.k_cache[i, kvh, lo:pos + 1]    # (T, hd)
                vals = self.v_cache[i, kvh, lo:pos + 1]
                for r in range(reps):
                    qh = q[kvh * reps + r]
                    aw = _softmax(keys @ qh / math.sqrt(hd))
                    outh[kvh * reps + r] = aw @ vals
            out = outh.reshape(H * hd)
            out = out * _sigmoid(lp["gate_proj"] @ h)
            attn = lp["out_proj"] @ out
            attn = _zcrms(attn, lp["post_norm"])
            y = skip + _sigmoid(lp["attn_gate"][0]) * attn

            # ---- hadamard mlp ----
            skip2 = y
            hh = _zcrms(y, lp["pre_hada"])
            Hn = g["hada_n"]
            Wm = _walsh_matrix(Hn)
            z = hh if Hn == C else np.pad(hh, (0, Hn - C))
            z = (lp["d1"] * z) @ Wm
            z = _silu(lp["d2"] * z) @ Wm
            y = skip2 + (lp["d3"] * z)[:C]

            y = y - u
            hpost = 2 * _sigmoid(p["mhc_a_post"][i] * (self._phi(p["mhc_phi_post"], i) @ nx)
                                 + p["mhc_b_post"][i] + self.post_off[i])  # (n,)
            res = self._phi(p["mhc_phi_res"], i) @ nx                       # (n*n,)
            hres = _sinkhorn(p["mhc_a_res"][i] * res.reshape(n, n) + p["mhc_b_res"][i])
            x = hres @ x + hpost[:, None] * y[None, :]

        xm = x.mean(axis=0)
        xm = _zcrms(xm, p["final_norm"])
        logits = p["embedding"] @ xm
        return logits

    def _phi(self, phi, i):
        """phi stored quantized as (L*lanes, nC) [out,in]; slice layer i -> (lanes, nC)."""
        n = self.n_lanes
        if phi.ndim == 2:
            rows = phi.shape[0] // self.g["num_layers"]
            return phi[i * rows:(i + 1) * rows]
        return phi[i]

    def generate(self, prompt, max_new_tokens=96, verbose=False):
        g = self.g
        ids = [self.tok.bos_id] + self.tok.encode(prompt)
        ids = ids[:g["max_seq_len"] - 1]
        max_len = min(g["max_seq_len"], len(ids) + max_new_tokens)
        self.reset(max_len)
        logits = None
        for pos, t in enumerate(ids):
            logits = self.step(t, pos)
        out = []
        pos = len(ids)
        nxt = int(np.argmax(logits))
        for _ in range(max_new_tokens):
            if nxt == self.tok.eos_id or pos >= max_len:
                break
            out.append(nxt)
            if verbose:
                sys.stdout.write(self.tok.decode([nxt]) or "")
                sys.stdout.flush()
            logits = self.step(nxt, pos)
            nxt = int(np.argmax(logits))
            pos += 1
        if verbose:
            print()
        return self.tok.decode(out), out


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "model/needle2.cact"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "What's the weather in Paris right now?"
    m = NeedleNP(path)
    print("geometry:", json.dumps({k: v for k, v in m.g.items()}, default=str))
    text, ids = m.generate(prompt, max_new_tokens=48, verbose=True)
    print("tokens:", ids[:32])
    print("text:", text)
