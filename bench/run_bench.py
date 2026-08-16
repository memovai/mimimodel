"""Tool-calling quality benchmark: our C engine (needle.c) vs the official
closed-source needle lib (oracle), scored against expected calls.

Match rules:
- expected name null  -> pass if no tool call produced
- arg value "*"       -> any value accepted
- arg value "*sub*"   -> substring match (case-insensitive)
- otherwise           -> exact match (int compare for numbers)
- multi_ok            -> extra calls with these names don't count against
"""
import json
import re
import subprocess
import sys
import time

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = json.load(open(f"{ROOT}/bench/cases.json"))
CASES = spec["cases"]


def parse_calls(text):
    """Extract [{"name":..,"arguments":{..}}] from generated text."""
    text = text.split("</tool_call>")[0].strip()
    if not text.startswith("["):
        text = "[" + text
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\[.*\]", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None


def match_case(case, calls):
    exp = case["expect"]
    if exp["name"] is None:
        return (not calls, "no-call expected, got " + json.dumps(calls))
    if not calls:
        return (False, "no call produced")
    allowed_extra = set(exp.get("multi_ok", []))
    primary = None
    for c in calls:
        if c.get("name") == exp["name"]:
            primary = c
            break
    if primary is None:
        return (False, f"wrong tool: {[c.get('name') for c in calls]}")
    for c in calls:
        if c is not primary and c.get("name") not in allowed_extra and c.get("name") != exp["name"]:
            return (False, f"unexpected extra call {c.get('name')}")
    args = primary.get("arguments") or {}
    for k, want in exp["args"].items():
        got = args.get(k)
        if want == "*":
            if got is None:
                return (False, f"missing arg {k}")
        elif isinstance(want, str) and want.startswith("*") and want.endswith("*") and len(want) > 1:
            sub = want.strip("*").lower()
            if got is None or sub not in str(got).lower():
                return (False, f"arg {k}: want ~'{sub}', got {got!r}")
        else:
            if got is None:
                return (False, f"missing arg {k}")
            try:
                if isinstance(want, (int, float)):
                    if float(got) != float(want):
                        return (False, f"arg {k}: want {want}, got {got!r}")
                elif str(got) != str(want):
                    return (False, f"arg {k}: want {want!r}, got {got!r}")
            except (TypeError, ValueError):
                return (False, f"arg {k}: want {want!r}, got {got!r}")
    return (True, "ok")


def run_ours(q):
    t0 = time.time()
    r = subprocess.run(["/tmp/needle_opt", f"{ROOT}/model/needle2.cact", q,
                        spec["tools_needle"], "80"],
                       capture_output=True, text=True, timeout=120)
    return parse_calls(r.stdout), time.time() - t0, r.stdout.strip()


_oracle = None
def run_oracle(q):
    global _oracle
    import needle as needle_mod
    from needle import Needle
    if _oracle is None:
        _oracle = Needle(tools=spec["tools_oracle"])
    needle_mod._lib().needle_reset()   # clear session state between cases
    t0 = time.time()
    r = _oracle.complete(q)
    calls = [{"name": c["name"], "arguments": c["arguments"]}
             for c in (r.get("function_calls") or [])]
    return calls, time.time() - t0, r.get("confidence")


def main():
    rows = []
    ours_pass = orac_pass = agree = 0
    for case in CASES:
        oc, ot, raw = run_ours(case["q"])
        rc, rt, conf = run_oracle(case["q"])
        op, oreason = match_case(case, oc)
        rp, rreason = match_case(case, rc)
        names_o = tuple(sorted(c.get("name") or "" for c in (oc or [])))
        names_r = tuple(sorted(c.get("name") or "" for c in (rc or [])))
        same = names_o == names_r
        ours_pass += op
        orac_pass += rp
        agree += same
        rows.append((case["id"], case["cat"], op, rp, same, conf, oreason, rreason))
        print(f"{case['id']:16s} {case['cat']:8s} ours={'✅' if op else '❌'} "
              f"oracle={'✅' if rp else '❌'} agree={'=' if same else '≠'} conf={conf} "
              f"{'' if op else '| ours: ' + oreason}{'' if rp else ' | oracle: ' + rreason}")
    n = len(CASES)
    print(f"\n=== ours {ours_pass}/{n} ({100*ours_pass//n}%) | "
          f"oracle {orac_pass}/{n} ({100*orac_pass//n}%) | "
          f"tool-choice agreement {agree}/{n} ({100*agree//n}%)")
    json.dump([{"id": r[0], "cat": r[1], "ours": r[2], "oracle": r[3], "agree": r[4],
                "conf": r[5]} for r in rows],
              open(f"{ROOT}/bench/results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
