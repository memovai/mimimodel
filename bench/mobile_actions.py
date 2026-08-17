"""Benchmark the engine on google/mobile-actions (CC-BY-4.0), the on-device
function-calling eval set published alongside FunctionGemma.

https://huggingface.co/datasets/google/mobile-actions

The dataset uses Google's function-declaration shape with UPPERCASE types; this
converts it to the JSON-Schema shape Needle was trained on. All 9,654 records
share one 7-tool set, so the <tools> block is byte-identical across cases and
the engine's KV prefix cache stays warm for the whole run.

Usage:
    python bench/mobile_actions.py --limit 100            # our engine
    python bench/mobile_actions.py --oracle --limit 100   # official closed lib
    python bench/mobile_actions.py --queries out.txt      # dump queries for the device
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "bench", "mobile_actions.jsonl")
URL = "https://huggingface.co/datasets/google/mobile-actions/resolve/main/dataset.jsonl"


def ensure_dataset():
    if not os.path.exists(DATA):
        print(f"downloading {URL} -> {DATA}", file=sys.stderr)
        urllib.request.urlretrieve(URL, DATA)
    return [json.loads(l) for l in open(DATA) if l.strip()]


def lower_types(node):
    """Google declarations say OBJECT/STRING; JSON Schema says object/string."""
    if isinstance(node, dict):
        return {k: (v.lower() if k == "type" and isinstance(v, str) else lower_types(v))
                for k, v in node.items()}
    if isinstance(node, list):
        return [lower_types(x) for x in node]
    return node


def to_needle_tools(tools):
    out = []
    for t in tools:
        f = t["function"]
        out.append({
            "name": f["name"],
            "description": f.get("description", ""),
            "parameters": lower_types(f.get("parameters", {"type": "object", "properties": {}})),
        })
    return out


def build_turn(rec):
    """The trained prompt format (needle/model/finetune.py: render_example) has a
    dedicated system turn ahead of the user turn. Gluing the dataset's developer
    content onto the front of the query instead costs a lot: measured on the
    official engine, 85% -> 100% tool-name accuracy just from separating them."""
    dev = user = ""
    for m in rec["messages"]:
        if m["role"] == "developer":
            dev = m.get("content", "")
        elif m["role"] == "user":
            user = m.get("content", "")
    return " ".join(dev.split()), " ".join(user.split())


def build_query(rec):
    dev, user = build_turn(rec)
    return (dev + " " + user).strip()


def expected(rec):
    calls = rec["messages"][-1].get("tool_calls", [])
    return [{"name": c["function"]["name"], "arguments": c["function"].get("arguments", {})}
            for c in calls]


def norm(v):
    return str(v).strip().lower()


def score(got, want):
    """Ordered strict exact match, the metric the published leaderboard uses:
    function names, call order and every argument must match. Returns
    (names_ok, exact_ok) where names_ok ignores arguments."""
    if not want:
        return (not got, not got)
    if len(got) != len(want):
        return (False, False)
    names_ok = all(g.get("name") == w["name"] for g, w in zip(got, want))
    if not names_ok:
        return (False, False)
    for g, w in zip(got, want):
        ga = g.get("arguments") or {}
        wa = w["arguments"] or {}
        if set(ga.keys()) != set(wa.keys()):
            return (True, False)
        if any(norm(ga[k]) != norm(wa[k]) for k in wa):
            return (True, False)
    return (True, True)


def run_ours(turns, tools_json, binary):
    """turns is a list of (system, query); the batch protocol takes
    "system<TAB>query" per line."""
    qfile = os.path.join(ROOT, "bench", f".ma_queries.{os.getpid()}.txt")
    with open(qfile, "w") as fh:
        for sysmsg, q in turns:
            fh.write(sysmsg.replace("\n", " ").replace("\t", " ") + "\t"
                     + q.replace("\n", " ").replace("\t", " ") + "\n")
    proc = subprocess.run([binary, os.path.join(ROOT, "model", "needle2.cact"),
                           "@" + qfile, tools_json],
                          capture_output=True, text=True)
    sys.stderr.write(proc.stderr[-300:])
    lines = [l for l in proc.stdout.split("\n") if l != ""]
    if len(lines) != len(turns):
        raise RuntimeError(
            f"batch protocol broken: {len(turns)} queries produced {len(lines)} lines. "
            "A raw control character in a decoded string value splits a record and "
            "silently misaligns every later result.")
    out = []
    for line in lines:
        parts = line.split("\t")
        try:
            out.append((json.loads(parts[0]), float(parts[1])))
        except Exception:
            out.append(([], 0.0))
    os.remove(qfile)
    return out


def run_oracle(turns, tools):
    import needle as needle_mod
    from needle import Needle
    import time
    tl = [{"name": t["name"], "description": t["description"],
           "parameters": t["parameters"]} for t in tools]
    agents = {}
    out = []
    for sysmsg, q in turns:
        agent = agents.get(sysmsg)
        if agent is None:
            agent = Needle(tools=tl, system=sysmsg)
            agents[sysmsg] = agent
        needle_mod._lib().needle_reset()
        t0 = time.time()
        r = agent.complete(q)
        calls = [{"name": c["name"], "arguments": c["arguments"]}
                 for c in (r.get("function_calls") or [])]
        out.append((calls, (time.time() - t0) * 1000))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all eval records")
    ap.add_argument("--oracle", action="store_true", help="run the official closed engine instead")
    ap.add_argument("--binary", default="/tmp/needle_ma")
    ap.add_argument("--queries", help="write the query list to this file and exit (for the device)")
    ap.add_argument("--out", help="write per-case results as JSON")
    args = ap.parse_args()

    ev = [r for r in ensure_dataset() if r["metadata"] == "eval"]
    if args.limit:
        ev = ev[:args.limit]
    tools = to_needle_tools(ev[0]["tools"])
    tools_json = json.dumps(tools, separators=(",", ":"), ensure_ascii=False)
    turns = [build_turn(r) for r in ev]

    if args.queries:
        with open(args.queries, "w") as fh:
            for sysmsg, q in turns:
                fh.write(sysmsg.replace("\n", " ") + "\t" + q.replace("\n", " ") + "\n")
        with open(args.queries + ".tools", "w") as fh:
            fh.write(tools_json)
        print(f"wrote {len(turns)} queries -> {args.queries}")
        return

    results = run_oracle(turns, tools) if args.oracle else run_ours(turns, tools_json, args.binary)

    n = len(ev)
    buckets = {1: [0, 0, 0], 2: [0, 0, 0]}     # [total, names_ok, exact_ok]
    rows = []
    for rec, (got, ms) in zip(ev, results):
        want = expected(rec)
        names_ok, exact_ok = score(got, want)
        b = buckets.setdefault(len(want), [0, 0, 0])
        b[0] += 1
        b[1] += names_ok
        b[2] += exact_ok
        rows.append({"query": build_query(rec)[:120], "want": want, "got": got,
                     "name_ok": names_ok, "exact_ok": exact_ok, "ms": ms,
                     "n_expected": len(want)})

    engine = "official closed-source engine" if args.oracle else "this engine"
    print(f"\n=== google/mobile-actions eval · {engine} · {n} cases ===")
    tot = sum(b[0] for b in buckets.values())
    allx = sum(b[2] for b in buckets.values())
    alln = sum(b[1] for b in buckets.values())
    print(f"accuracy (ordered strict exact, all {tot}): {allx}/{tot} ({100.0*allx/tot:.1f}%)"
          f"   name acc {alln}/{tot} ({100.0*alln/tot:.1f}%)")
    for k in sorted(buckets):
        b = buckets[k]
        if not b[0]:
            continue
        print(f"  {k}-call ({b[0]:3d}): exact {b[2]:3d} ({100.0*b[2]/b[0]:.1f}%)"
              f"   names {b[1]:3d} ({100.0*b[1]/b[0]:.1f}%)")
    lat = sorted(r["ms"] for r in rows if r["ms"] > 0)
    if lat:
        print(f"latency: median {lat[len(lat)//2]:.0f} ms, first call {rows[0]['ms']:.0f} ms "
              f"(cold — prefix cache fills), p90 {lat[int(len(lat)*0.9)]:.0f} ms")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1, ensure_ascii=False)
        print(f"per-case results -> {args.out}")


if __name__ == "__main__":
    main()
