"""Benchmark the engine on google/mobile-actions (CC-BY-4.0), the on-device
function-calling eval set published alongside FunctionGemma.

https://huggingface.co/datasets/google/mobile-actions

The dataset uses Google's function-declaration shape with UPPERCASE types; this
converts it to the JSON-Schema shape Needle was trained on. The seven schemas are
shared, but their order is randomized per record. Dataset-order mode preserves
that property; canonical mode is useful for controlled speed comparisons.

Usage:
    python bench/mobile_actions.py --limit 100            # our engine
    python bench/mobile_actions.py --oracle --limit 100   # official closed lib
    python bench/mobile_actions.py --queries out.txt      # dump queries for the device
"""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
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
    """Preserve the dedicated system turn used by the trained prompt format."""
    dev = user = ""
    for m in rec["messages"]:
        if m["role"] == "developer":
            dev = m.get("content", "")
        elif m["role"] == "user":
            user = m.get("content", "")
    return dev, user


def build_query(rec):
    dev, user = build_turn(rec)
    return (dev + " " + user).strip()


def expected(rec):
    calls = rec["messages"][-1].get("tool_calls", [])
    return [{"name": c["function"]["name"], "arguments": c["function"].get("arguments", {})}
            for c in calls]


def norm(v):
    return str(v).strip().lower()


def score(got, want, metric="strict"):
    """Ordered structural match. Strict preserves value case and whitespace;
    normalized retains the historical case-insensitive string comparison."""
    if metric not in ("strict", "normalized"):
        raise ValueError(f"unknown metric: {metric}")
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
        if metric == "strict" and any(ga[k] != wa[k] for k in wa):
            return (True, False)
        if metric == "normalized" and any(norm(ga[k]) != norm(wa[k]) for k in wa):
            return (True, False)
    return (True, True)


def encode_field(text):
    """Keep the line protocol one-record-per-line without flattening prompts."""
    return text.replace("\x1e", " ").replace("\t", " ").replace("\r\n", "\n").replace("\n", "\x1e")


def ordered_tools(rec, mode, first_tools=None):
    tools = to_needle_tools(rec["tools"])
    if mode == "canonical":
        tools.sort(key=lambda tool: tool["name"])
    elif mode == "fixed-first":
        tools = first_tools
    return tools


def common_bm25_tools(query, tools, top_k=2):
    """Reference-side retrieval for decoder-only comparisons.

    This mirrors needle.c's ASCII tokenization and BM25 constants. Stable sorting
    preserves the selected protocol's tool order when scores tie.
    """
    words = [word.lower() for word in re.findall(r"[A-Za-z0-9]+", query) if len(word) >= 2]
    docs = [re.findall(r"[A-Za-z0-9]+", f"{tool['name']} {tool.get('description', '')}")
            for tool in tools]
    avgdl = sum(max(len(doc), 1) for doc in docs) / max(len(docs), 1)
    scores = [0.0] * len(tools)
    for word in words:
        counts = [sum(token.lower() == word for token in doc) for doc in docs]
        df = sum(count > 0 for count in counts)
        if not df:
            continue
        idf = math.log(1.0 + (len(docs) - df + 0.5) / (df + 0.5))
        for index, count in enumerate(counts):
            if count:
                dl = max(len(docs[index]), 1)
                scores[index] += idf * count * 2.5 / (count + 1.5 * (0.25 + 0.75 * dl / avgdl))
    order = sorted(range(len(tools)), key=lambda index: -scores[index])
    return [tools[index] for index in order[:top_k]]


def build_cases(records, tool_order, retrieval="native"):
    first_tools = to_needle_tools(records[0]["tools"])
    cases = []
    for rec in records:
        system, query = build_turn(rec)
        tools = ordered_tools(rec, tool_order, first_tools)
        if retrieval == "common-bm25-2":
            tools = common_bm25_tools(query, tools, 2)
        tools_json = json.dumps(tools, separators=(",", ":"), ensure_ascii=False)
        cases.append((system, query, tools, tools_json))
    return cases


def run_ours(cases, binary):
    """The batch protocol takes system<TAB>query<TAB>tools per line."""
    qfile = os.path.join(ROOT, "bench", f".ma_queries.{os.getpid()}.txt")
    with open(qfile, "w") as fh:
        for sysmsg, query, _tools, tools_json in cases:
            fh.write(f"{encode_field(sysmsg)}\t{encode_field(query)}\t{tools_json}\n")
    try:
        proc = subprocess.run(
            [binary, os.path.join(ROOT, "model", "needle2.cact"), "@" + qfile, "[]"],
            capture_output=True, text=True,
        )
    finally:
        os.remove(qfile)
    if proc.returncode:
        raise RuntimeError(f"engine exited {proc.returncode}:\n{proc.stderr[-2000:]}")
    sys.stderr.write(proc.stderr[-500:])
    lines = proc.stdout.splitlines()
    if len(lines) != len(cases):
        raise RuntimeError(
            f"batch protocol broken: {len(cases)} queries produced {len(lines)} lines. "
            "A raw control character in a decoded string value splits a record and "
            "silently misaligns every later result.")
    out = []
    for index, line in enumerate(lines):
        parts = line.split("\t")
        try:
            out.append({
                "calls": json.loads(parts[0]),
                "ms": float(parts[1]),
                "prefill_tok": int(parts[2]),
                "decode_tok": int(parts[3]),
                "prefill_us": int(parts[4]) if len(parts) > 4 else None,
                "decode_us": int(parts[5]) if len(parts) > 5 else None,
                "init_ms": 0.0,
            })
        except (ValueError, json.JSONDecodeError, IndexError) as exc:
            raise RuntimeError(f"malformed engine output at case {index}: {line!r}") from exc
    return out


def run_oracle(cases):
    import needle as needle_mod
    from needle import Needle
    agents = {}
    out = []
    for sysmsg, query, _tools, tools_json in cases:
        key = (sysmsg, tools_json)
        agent = agents.get(key)
        init_ms = 0.0
        if agent is None:
            t_init = time.perf_counter()
            agent = Needle(tools=tools_json, system=sysmsg)
            init_ms = (time.perf_counter() - t_init) * 1000
            agents[key] = agent
        needle_mod._lib().needle_reset()
        t0 = time.perf_counter()
        r = agent.complete(query)
        calls = [{"name": c["name"], "arguments": c["arguments"]}
                 for c in (r.get("function_calls") or [])]
        out.append({
            "calls": calls,
            "ms": (time.perf_counter() - t0) * 1000,
            "init_ms": init_ms,
            "prefill_tps": r.get("prefill_tps"),
            "decode_tps": r.get("decode_tps"),
            "peak_ram_mb": r.get("peak_ram_mb"),
        })
    return out


def rescore_file(path):
    payload = json.load(open(path))
    rows = payload.get("rows", []) if isinstance(payload, dict) else payload
    print(f"=== rescore {path} ({len(rows)} rows) ===")
    for metric in ("strict", "normalized"):
        scores = [score(row.get("got") or [], row.get("want") or [], metric) for row in rows]
        print(f"{metric:10s}: exact {sum(x[1] for x in scores)}/{len(rows)}  "
              f"names {sum(x[0] for x in scores)}/{len(rows)}")
    meta_path = path + ".meta.json"
    saved_metric = "normalized"
    if os.path.exists(meta_path):
        with open(meta_path) as handle:
            saved_metric = json.load(handle).get("metric", saved_metric)
    stale = []
    for index, row in enumerate(rows):
        actual = score(row.get("got") or [], row.get("want") or [], saved_metric)
        saved = (row.get("name_ok"), row.get("exact_ok"))
        if None not in saved and actual != saved:
            stale.append(index)
    print(f"saved flag mismatches ({saved_metric} metric): {len(stale)}")
    if stale:
        print("first indices:", ", ".join(map(str, stale[:20])))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all eval records")
    ap.add_argument("--oracle", action="store_true", help="run the official closed engine instead")
    ap.add_argument("--binary", default="/tmp/needle_ma")
    ap.add_argument("--tool-order", choices=("dataset", "canonical", "fixed-first"),
                    default="dataset", help="dataset preserves the per-record randomized order")
    ap.add_argument("--metric", choices=("strict", "normalized"), default="strict")
    ap.add_argument("--retrieval", choices=("native", "common-bm25-2"), default="native",
                    help="common mode gives both engines the same two preselected tools")
    ap.add_argument("--queries", help="write the query list to this file and exit (for the device)")
    ap.add_argument("--out", help="write per-case results as JSON")
    ap.add_argument("--rescore", help="recompute metrics from a saved result and exit")
    args = ap.parse_args()

    if args.rescore:
        rescore_file(args.rescore)
        return

    ev = [r for r in ensure_dataset() if r["metadata"] == "eval"]
    if args.limit:
        ev = ev[:args.limit]
    cases = build_cases(ev, args.tool_order, args.retrieval)

    if args.queries:
        with open(args.queries, "w") as fh:
            for sysmsg, query, _tools, tools_json in cases:
                fh.write(f"{encode_field(sysmsg)}\t{encode_field(query)}\t{tools_json}\n")
        print(f"wrote {len(cases)} queries -> {args.queries}")
        return

    results = run_oracle(cases) if args.oracle else run_ours(cases, args.binary)

    n = len(ev)
    buckets = {1: [0, 0, 0], 2: [0, 0, 0]}     # [total, names_ok, exact_ok]
    rows = []
    for rec, result in zip(ev, results):
        got, ms = result["calls"], result["ms"]
        want = expected(rec)
        names_ok, exact_ok = score(got, want, args.metric)
        b = buckets.setdefault(len(want), [0, 0, 0])
        b[0] += 1
        b[1] += names_ok
        b[2] += exact_ok
        rows.append({"query": build_query(rec)[:120], "want": want, "got": got,
                     "name_ok": names_ok, "exact_ok": exact_ok, "ms": ms,
                     "n_expected": len(want),
                     **{key: value for key, value in result.items() if key != "calls"}})

    engine = "official closed-source engine" if args.oracle else "this engine"
    print(f"\n=== google/mobile-actions eval · {engine} · {n} cases ===")
    print(f"protocol: tool-order={args.tool_order}, retrieval={args.retrieval}, "
          f"metric={args.metric}, developer whitespace preserved")
    tot = sum(b[0] for b in buckets.values())
    allx = sum(b[2] for b in buckets.values())
    alln = sum(b[1] for b in buckets.values())
    metric_label = "ordered strict exact" if args.metric == "strict" else "legacy normalized"
    print(f"accuracy ({metric_label}, all {tot}): {allx}/{tot} ({100.0*allx/tot:.1f}%)"
          f"   name acc {alln}/{tot} ({100.0*alln/tot:.1f}%)")
    for k in sorted(buckets):
        b = buckets[k]
        if not b[0]:
            continue
        print(f"  {k}-call ({b[0]:3d}): exact {b[2]:3d} ({100.0*b[2]/b[0]:.1f}%)"
              f"   names {b[1]:3d} ({100.0*b[1]/b[0]:.1f}%)")
    lat = sorted(r["ms"] for r in rows if r["ms"] > 0)
    if lat:
        print(f"latency: median {lat[len(lat)//2]:.0f} ms, first call {rows[0]['ms']:.0f} ms, "
              f"p90 {lat[int(len(lat)*0.9)]:.0f} ms")
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(rows, handle, indent=1, ensure_ascii=False)
        binary_path = args.binary if not args.oracle else __import__("needle")._library_path()
        meta = {
            "engine": "oracle" if args.oracle else "ours",
            "protocol_version": 2,
            "metric": args.metric,
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
        else:
            model_path = os.path.join(ROOT, "model", "needle2.cact")
            meta["model_sha256"] = sha256_file(model_path)
        with open(args.out + ".meta.json", "w") as handle:
            json.dump(meta, handle, indent=2)
        print(f"per-case results -> {args.out}")


if __name__ == "__main__":
    main()
