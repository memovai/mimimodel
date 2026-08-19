"""Drive the mobile-actions benchmark on a real ESP32-S3 over the serial REPL,
and check the device's answers against the host engine's on the same cases.

The firmware runs the same needle.c, so any disagreement means a portability
bug, not a model difference. Timing here is the real on-device cost.

    python bench/device_runner.py --port /dev/cu.usbmodem... --limit 5
"""
import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time

import serial

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))
from mobile_actions import (build_cases, encode_field, ensure_dataset, expected,
                            score, sha256_file)


def wait_for(ser, needle, timeout, sink=None):
    buf = ""
    deadline = time.time() + timeout
    while time.time() < deadline:
        chunk = ser.read(4096)
        if chunk:
            text = chunk.decode("utf-8", "replace")
            buf += text
            if sink:
                sink.write(text)
                sink.flush()
        if needle in buf:
            return buf
    return None


def stratified_sample(indexed_records, count, seed):
    """Proportionally sample by expected call count with largest remainders."""
    if count <= 0 or count > len(indexed_records):
        raise ValueError(f"sample must be between 1 and {len(indexed_records)}")
    groups = {}
    for item in indexed_records:
        groups.setdefault(len(expected(item[1])), []).append(item)
    quotas = {key: count * len(group) / len(indexed_records)
              for key, group in groups.items()}
    allocation = {key: min(len(groups[key]), int(math.floor(quota)))
                  for key, quota in quotas.items()}
    remaining = count - sum(allocation.values())
    order = sorted(groups, key=lambda key: (quotas[key] - allocation[key],
                                            len(groups[key])), reverse=True)
    while remaining:
        for key in order:
            if allocation[key] < len(groups[key]):
                allocation[key] += 1
                remaining -= 1
                if not remaining:
                    break
    rng = random.Random(seed)
    selected = []
    for key in sorted(groups):
        selected.extend(rng.sample(groups[key], allocation[key]))
    return sorted(selected), allocation


def wilson_interval(successes, total, z=1.96):
    if not total:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return centre - radius, centre + radius


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0,
                    help="proportional random sample stratified by expected call count")
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--indices", help="comma-separated eval indices (for deterministic resume)")
    ap.add_argument("--timeout", type=int, default=600, help="seconds per case")
    ap.add_argument("--out", default="/tmp/ma_device.json")
    ap.add_argument("--binary", default="/tmp/needle_ma")
    ap.add_argument("--tool-order", choices=("dataset", "canonical", "fixed-first"),
                    default="dataset")
    ap.add_argument("--retrieval", choices=("native", "common-bm25-2"), default="native")
    ap.add_argument("--allow-nonstandard", action="store_true",
                    help="allow fast-math/profile firmware (not valid for fair accuracy runs)")
    args = ap.parse_args()

    all_ev = [r for r in ensure_dataset() if r["metadata"] == "eval"]
    indexed = list(enumerate(all_ev))
    allocation = None
    if args.indices:
        requested = [int(value) for value in args.indices.split(",") if value.strip()]
        if len(set(requested)) != len(requested) or any(
                index < 0 or index >= len(indexed) for index in requested):
            ap.error("--indices must be unique eval indices in range")
        indexed = [(index, all_ev[index]) for index in requested]
    elif args.sample:
        indexed, allocation = stratified_sample(indexed, args.sample, args.seed)
    else:
        indexed = indexed[args.offset:args.offset + args.limit]
    indices = [index for index, _rec in indexed]
    ev = [rec for _index, rec in indexed]
    cases = build_cases(ev, args.tool_order, args.retrieval)
    queries = [f"{encode_field(system)}\t{encode_field(query)}\t{tools_json}"
               for system, query, _tools, tools_json in cases]

    # Freeze the binary the host reference is computed with. The device firmware
    # is flashed once and the run takes hours; if needle.c is edited and
    # rebuilt meanwhile, the two sides are different builds and the parity check
    # reports portability bugs that are really just build skew (it did).
    frozen = f"/tmp/needle_frozen.{os.getpid()}"
    import shutil
    shutil.copy2(args.binary, frozen)
    fw = os.path.join(ROOT, "needle-esp32s3", "build", "needle_esp32s3.bin")
    if os.path.exists(fw) and os.path.getmtime(fw) < os.path.getmtime(
            os.path.join(ROOT, "needle.c")):
        print("WARNING: needle-esp32s3/build is older than needle.c — reflash "
              "before trusting the parity column", file=sys.stderr)

    # host reference for the same cases, one process so it mirrors the device
    qfile = "/tmp/ma_dev_q.txt"
    with open(qfile, "w") as fh:
        fh.write("\n".join(queries) + "\n")
    hp = subprocess.run([frozen, os.path.join(ROOT, "model", "needle2.cact"),
                         "@" + qfile, cases[0][3] if cases else "[]"],
                        capture_output=True, text=True)
    host = []
    for line in hp.stdout.splitlines():
        p = line.split("\t")
        try:
            host.append(json.loads(p[0]))
        except Exception:
            host.append([])
    print(f"host reference computed for {len(host)} cases")

    # On a USB-UART bridge, asserting DTR/RTS at open resets the chip into
    # download mode, so deassert both before opening; then pulse EN via RTS.
    ser = serial.Serial()
    ser.port = args.port
    ser.baudrate = 115200
    ser.timeout = 1
    ser.dtr = False
    ser.rts = False
    ser.open()
    ser.rts = True
    time.sleep(0.15)
    ser.rts = False
    print("waiting for the board to boot...")
    boot = wait_for(ser, "REPL ready", 120)
    if boot is None:
        print("ERROR: board never reached the REPL", file=sys.stderr)
        return 1
    build_match = re.search(r"\[needle\] build:[^\r\n]*", boot)
    build_line = build_match.group(0) if build_match else None
    if (not args.allow_nonstandard and
            (not build_line or "fast_math=0" not in build_line or "profile=0" not in build_line)):
        ser.close()
        print(f"ERROR: fair accuracy runs require fast_math=0/profile=0; got {build_line!r}",
              file=sys.stderr)
        return 2
    for pattern in (r"\[needle\] build:[^\r\n]*", r"\[needle\] loaded[^\r\n]*",
                    r"\[needle\] weights cached[^\r\n]*",
                    r"\[pie\][^\r\n]*PASS", r"\[pie\] matvec[^\r\n]*"):
        match = re.search(pattern, boot)
        if match:
            print(match.group(0))
    print("board ready\n")

    sampling = ({"method": "explicit-indices", "indices": indices}
                if args.indices else
                {"method": "proportional-stratified-by-call-count",
                 "seed": args.seed, "allocation": allocation,
                 "indices": indices} if args.sample else
                {"method": "contiguous", "offset": args.offset,
                 "limit": args.limit, "indices": indices})
    meta = {
        "tool_order": args.tool_order,
        "retrieval": args.retrieval,
        "metric": "strict",
        "sampling": sampling,
        "firmware_build": build_line,
        "dataset_sha256": sha256_file(os.path.join(ROOT, "bench", "mobile_actions.jsonl")),
        "model_sha256": sha256_file(os.path.join(ROOT, "model", "needle2.cact")),
        "firmware_sha256": sha256_file(fw) if os.path.exists(fw) else None,
        "host_binary_sha256": sha256_file(frozen),
    }

    rows = []
    for i, (dataset_index, rec, q) in enumerate(zip(indices, ev, queries)):
        want = expected(rec)
        print(f"[{i+1}/{len(ev)}] {q[:80]}...")
        ser.reset_input_buffer()
        # The ESP32 UART RX FIFO is 128 bytes and the REPL only drains it from
        # fgets(); a 300+ char query written in one burst at 115200 overflows it
        # and the device silently sees a mangled query. Feed it in small chunks.
        payload = (q + "\n").encode()
        for k in range(0, len(payload), 32):
            ser.write(payload[k:k + 32])
            ser.flush()
            time.sleep(0.03)
        t0 = time.time()
        out = wait_for(ser, "[needle] total", args.timeout)
        dt = time.time() - t0
        got = []
        dev_line = ""
        dev_tok = None
        if out:
            mt = re.search(r"\[needle\] total[^\r\n]*prefill (\d+) tok", out)
            if mt:
                dev_tok = int(mt.group(1))
            m = re.search(r"\[needle\] call: (.*)", out)
            if m:
                dev_line = m.group(1).strip()
                try:
                    got = json.loads(dev_line)
                except Exception:
                    got = []
        else:
            rows.append({"dataset_index": dataset_index, "n_expected": len(want),
                         "query": q[:100], "want": want, "timed_out": True,
                         "seconds": round(dt, 1)})
            with open(args.out, "w") as handle:
                json.dump(rows, handle, indent=1, ensure_ascii=False)
            meta.update({"status": "timed_out", "timed_out_index": dataset_index,
                         "completed_rows": len(rows) - 1})
            with open(args.out + ".meta.json", "w") as handle:
                json.dump(meta, handle, indent=2)
            ser.close()
            print(f"ERROR: dataset row {dataset_index} timed out after {dt:.1f}s; "
                  "aborting to preserve serial alignment", file=sys.stderr)
            return 3
        name_ok, exact_ok = score(got, want, "strict")
        host_name_ok, host_exact_ok = score(host[i], want, "strict")
        agree = json.dumps(got, sort_keys=True) == json.dumps(host[i], sort_keys=True)
        score_agree = (name_ok, exact_ok) == (host_name_ok, host_exact_ok)
        phase = re.search(
            r"\[needle\] total (\d+) ms \| prefill (\d+) tok ([0-9.]+) tok/s "
            r"\| decode (\d+) tok ([0-9.]+) tok/s", out or "")
        rows.append({"dataset_index": dataset_index, "n_expected": len(want),
                     "query": q[:100], "want": want, "device": got, "host": host[i],
                     "name_ok": name_ok, "exact_ok": exact_ok, "agrees_with_host": agree,
                     "host_name_ok": host_name_ok, "host_exact_ok": host_exact_ok,
                     "agrees_on_score": score_agree,
                     "seconds": round(dt, 1),
                     **({"device_ms": int(phase.group(1)),
                         "prefill_tok": int(phase.group(2)),
                         "prefill_tps": float(phase.group(3)),
                         "decode_tok": int(phase.group(4)),
                         "decode_tps": float(phase.group(5))} if phase else {})})
        with open(args.out, "w") as handle:
            json.dump(rows, handle, indent=1, ensure_ascii=False)
        print(f"      device: {dev_line[:90]}   (device saw {dev_tok} prompt tokens)")
        print(f"      want  : {json.dumps(want, ensure_ascii=False)[:90]}")
        print(f"      {dt:.0f}s | name {'OK' if name_ok else 'X'} | "
              f"host-parity {'OK' if agree else 'MISMATCH'}\n")
    ser.close()

    n = len(rows)
    print(f"=== device results · {n} cases ===")
    print(f"protocol          : tool-order={args.tool_order}, retrieval={args.retrieval}, "
          "strict metric")
    print(f"tool name correct : {sum(r['name_ok'] for r in rows)}/{n}")
    print(f"name+args exact   : {sum(r['exact_ok'] for r in rows)}/{n}")
    print(f"matches host      : {sum(r['agrees_with_host'] for r in rows)}/{n}")
    print(f"matches host score: {sum(r['agrees_on_score'] for r in rows)}/{n}")
    exact = sum(r["exact_ok"] for r in rows)
    low, high = wilson_interval(exact, n)
    print(f"sample exact 95% CI: {100 * low:.1f}% .. {100 * high:.1f}% "
          "(Wilson; descriptive only for a stratified sample)")
    secs = [r["seconds"] for r in rows]
    print(f"seconds per case  : median {sorted(secs)[len(secs)//2]:.0f}, "
          f"min {min(secs):.0f}, max {max(secs):.0f}")
    with open(args.out, "w") as handle:
        json.dump(rows, handle, indent=1, ensure_ascii=False)
    meta["status"] = "complete"
    with open(args.out + ".meta.json", "w") as handle:
        json.dump(meta, handle, indent=2)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
