"""Inference-speed comparison on google/mobile-actions.

Measures this engine against the official closed-source engine on the same host
and records. Native mode compares each engine's full retrieval and decode stack;
common-bm25-2 supplies identical candidates for a decoder-focused comparison.

What is NOT equal between the two, and matters for reading the result:
  - this engine is scalar C99; the official build is NEON-optimised ARM64
  - native mode uses different tool retrieval implementations
  - the official engine reports its own prefill/decode throughput; for this
    engine the batch protocol reports its actual prefill/decode phase timers

    python bench/speed.py --limit 200
    python bench/speed.py --limit 200 --oracle
"""
import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))
from mobile_actions import (DATA, build_cases, ensure_dataset, run_oracle,
                            run_ours, sha256_file)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def phase_tps(row, phase):
    tokens = row.get(f"{phase}_tok")
    micros = row.get(f"{phase}_us")
    if not tokens or not micros:
        return None
    return tokens * 1_000_000.0 / micros


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--binary", default="/tmp/needle_ma")
    ap.add_argument("--tool-order", choices=("dataset", "canonical", "fixed-first"),
                    default="canonical")
    ap.add_argument("--retrieval", choices=("native", "common-bm25-2"), default="native")
    ap.add_argument("--out")
    args = ap.parse_args()

    ev = [r for r in ensure_dataset() if r["metadata"] == "eval"][:args.limit]
    cases = build_cases(ev, args.tool_order, args.retrieval)

    t0 = time.perf_counter()
    if args.oracle:
        rows = run_oracle(cases)
        wall = time.perf_counter() - t0
        lat = [r["ms"] for r in rows]
        total_lat = [r["ms"] + r.get("init_ms", 0) for r in rows]
        init = [r.get("init_ms", 0) for r in rows]
        ptps = [r["prefill_tps"] for r in rows if r["prefill_tps"]]
        dtps = [r["decode_tps"] for r in rows if r["decode_tps"]]
        ram = [r["peak_ram_mb"] for r in rows if r["peak_ram_mb"]]
        print(f"\n=== official closed-source engine · {len(rows)} calls ===")
        print(f"protocol  : tool-order={args.tool_order}, retrieval={args.retrieval}, "
              "developer whitespace preserved")
        print(f"complete  : median {statistics.median(lat):7.0f} ms   "
              f"p90 {pct(lat,.9):7.0f} ms   total {wall:.0f} s")
        print(f"init      : median {statistics.median(init):7.0f} ms   "
              f"request total median {statistics.median(total_lat):.0f} ms")
        if ptps:
            print(f"prefill   : median {statistics.median(ptps):7.0f} tok/s")
        if dtps:
            print(f"decode    : median {statistics.median(dtps):7.0f} tok/s")
        if ram:
            print(f"peak RAM  : {max(ram):.0f} MB")
    else:
        rows = run_ours(cases, args.binary)
        wall = time.perf_counter() - t0
        lat = [r["ms"] for r in rows]
        pre = [r["prefill_tok"] for r in rows]
        dec = [r["decode_tok"] for r in rows]
        ptps = [phase_tps(r, "prefill") for r in rows]
        dtps = [phase_tps(r, "decode") for r in rows]
        ptps = [x for x in ptps if x]
        dtps = [x for x in dtps if x]
        print(f"\n=== this engine (scalar C99) · {len(rows)} calls ===")
        print(f"protocol  : tool-order={args.tool_order}, retrieval={args.retrieval}, "
              "developer whitespace preserved")
        print(f"latency   : median {statistics.median(lat):7.0f} ms   "
              f"p90 {pct(lat,.9):7.0f} ms   total {wall:.0f} s")
        print(f"tokens    : prefill median {statistics.median(pre):.0f}, "
              f"decode median {statistics.median(dec):.0f} per call")
        if ptps:
            print(f"prefill   : median {statistics.median(ptps):7.0f} tok/s")
        if dtps:
            print(f"decode    : median {statistics.median(dtps):7.0f} tok/s")
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(rows, handle, indent=1)
        binary_path = args.binary if not args.oracle else __import__("needle")._library_path()
        meta = {
            "engine": "oracle" if args.oracle else "ours",
            "tool_order": args.tool_order,
            "retrieval": args.retrieval,
            "dataset_sha256": sha256_file(DATA),
            "engine_sha256": sha256_file(binary_path),
        }
        if args.oracle:
            needle_mod = __import__("needle")
            meta["cactus_needle_version"] = needle_mod.__version__
            meta["cactus_needle_engine_version"] = __import__(
                "needle.agent.fetch", fromlist=["ENGINE_VERSION"]).ENGINE_VERSION
        with open(args.out + ".meta.json", "w") as handle:
            json.dump(meta, handle, indent=2)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
