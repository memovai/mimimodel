"""Inference-speed comparison on google/mobile-actions.

Measures this engine against the official closed-source engine on the same host,
same cases, same prompts — and reports the device number alongside.

What is NOT equal between the two, and matters for reading the result:
  - this engine is scalar C99; the official build is NEON-optimised ARM64
  - both prune the tool list before prompting (BM25 here, embedding+BM25 RRF
    there), so both see a short tools block
  - the official engine reports its own prefill/decode throughput; for this
    engine the batch protocol reports prefill/decode token counts and wall time

    python bench/speed.py --limit 200
    python bench/speed.py --limit 200 --oracle
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))
from mobile_actions import ensure_dataset, to_needle_tools, build_turn


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def run_ours(turns, tools_json, binary):
    qfile = f"/tmp/.speed_q.{os.getpid()}.txt"
    with open(qfile, "w") as fh:
        for s, q in turns:
            fh.write(s.replace("\t", " ") + "\t" + q.replace("\t", " ") + "\n")
    t0 = time.time()
    p = subprocess.run([binary, os.path.join(ROOT, "model", "needle2.cact"),
                        "@" + qfile, tools_json], capture_output=True, text=True)
    wall = time.time() - t0
    os.remove(qfile)
    rows = []
    for line in p.stdout.splitlines():
        f = line.split("\t")
        if len(f) < 4:
            continue
        rows.append({"ms": float(f[1]), "prefill_tok": int(f[2]), "decode_tok": int(f[3])})
    return rows, wall


def run_oracle(turns, tools):
    import needle as needle_mod
    from needle import Needle
    tl = [{"name": t["name"], "description": t["description"],
           "parameters": t["parameters"]} for t in tools]
    agents = {}
    rows = []
    t0 = time.time()
    for s, q in turns:
        ag = agents.get(s)
        if ag is None:
            ag = Needle(tools=tl, system=s)
            agents[s] = ag
        needle_mod._lib().needle_reset()
        t1 = time.time()
        r = ag.complete(q)
        rows.append({"ms": (time.time() - t1) * 1000,
                     "prefill_tps": r.get("prefill_tps"),
                     "decode_tps": r.get("decode_tps"),
                     "peak_ram_mb": r.get("peak_ram_mb")})
    return rows, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--binary", default="/tmp/needle_ma")
    ap.add_argument("--out")
    args = ap.parse_args()

    ev = [r for r in ensure_dataset() if r["metadata"] == "eval"][:args.limit]
    tools = to_needle_tools(ev[0]["tools"])
    tools_json = json.dumps(tools, separators=(",", ":"), ensure_ascii=False)
    turns = [build_turn(r) for r in ev]

    if args.oracle:
        rows, wall = run_oracle(turns, tools)
        lat = [r["ms"] for r in rows]
        ptps = [r["prefill_tps"] for r in rows if r["prefill_tps"]]
        dtps = [r["decode_tps"] for r in rows if r["decode_tps"]]
        ram = [r["peak_ram_mb"] for r in rows if r["peak_ram_mb"]]
        print(f"\n=== official closed-source engine · {len(rows)} calls ===")
        print(f"latency   : median {statistics.median(lat):7.0f} ms   "
              f"p90 {pct(lat,.9):7.0f} ms   total {wall:.0f} s")
        if ptps:
            print(f"prefill   : median {statistics.median(ptps):7.0f} tok/s")
        if dtps:
            print(f"decode    : median {statistics.median(dtps):7.0f} tok/s")
        if ram:
            print(f"peak RAM  : {max(ram):.0f} MB")
    else:
        rows, wall = run_ours(turns, tools_json, args.binary)
        lat = [r["ms"] for r in rows]
        pre = [r["prefill_tok"] for r in rows]
        dec = [r["decode_tok"] for r in rows]
        # per-call throughput: tokens processed over the call's wall time
        tot_tok = sum(pre) + sum(dec)
        print(f"\n=== this engine (scalar C99) · {len(rows)} calls ===")
        print(f"latency   : median {statistics.median(lat):7.0f} ms   "
              f"p90 {pct(lat,.9):7.0f} ms   total {wall:.0f} s")
        print(f"tokens    : prefill median {statistics.median(pre):.0f}, "
              f"decode median {statistics.median(dec):.0f} per call")
        print(f"throughput: {tot_tok / wall:7.0f} tok/s overall "
              f"({sum(pre)/wall:.0f} prefill + {sum(dec)/wall:.0f} decode)")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
