"""Drive the mobile-actions benchmark on a real ESP32-S3 over the serial REPL,
and check the device's answers against the host engine's on the same cases.

The firmware runs the same needle.c, so any disagreement means a portability
bug, not a model difference. Timing here is the real on-device cost.

    python bench/device_runner.py --port /dev/cu.usbmodem... --limit 5
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import serial

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))
from mobile_actions import ensure_dataset, to_needle_tools, build_turn, expected, score


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=600, help="seconds per case")
    ap.add_argument("--out", default="/tmp/ma_device.json")
    ap.add_argument("--binary", default="/tmp/needle_ma")
    args = ap.parse_args()

    ev = [r for r in ensure_dataset() if r["metadata"] == "eval"]
    ev = ev[args.offset:args.offset + args.limit]
    tools = to_needle_tools(ev[0]["tools"])
    tools_json = json.dumps(tools, separators=(",", ":"), ensure_ascii=False)
    turns = [build_turn(r) for r in ev]
    queries = [(sysmsg.replace("\n", " ").replace("\t", " ") + "\t"
                + q.replace("\n", " ").replace("\t", " ")) for sysmsg, q in turns]

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
                         "@" + qfile, tools_json], capture_output=True, text=True)
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
    if wait_for(ser, "REPL ready", 120) is None:
        print("ERROR: board never reached the REPL", file=sys.stderr)
        return 1
    print("board ready\n")

    rows = []
    for i, (rec, q) in enumerate(zip(ev, queries)):
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
            mt = re.search(r"\[needle\] (\d+) prompt tokens", out)
            if mt:
                dev_tok = int(mt.group(1))
            m = re.search(r"\[needle\] call: (.*)", out)
            if m:
                dev_line = m.group(1).strip()
                try:
                    got = json.loads(dev_line)
                except Exception:
                    got = []
        name_ok, exact_ok = score(got, want)
        agree = json.dumps(got, sort_keys=True) == json.dumps(host[i], sort_keys=True)
        rows.append({"query": q[:100], "want": want, "device": got, "host": host[i],
                     "name_ok": name_ok, "exact_ok": exact_ok, "agrees_with_host": agree,
                     "seconds": round(dt, 1)})
        print(f"      device: {dev_line[:90]}   (device saw {dev_tok} prompt tokens)")
        print(f"      want  : {json.dumps(want, ensure_ascii=False)[:90]}")
        print(f"      {dt:.0f}s | name {'OK' if name_ok else 'X'} | "
              f"host-parity {'OK' if agree else 'MISMATCH'}\n")
    ser.close()

    n = len(rows)
    print(f"=== device results · {n} cases ===")
    print(f"tool name correct : {sum(r['name_ok'] for r in rows)}/{n}")
    print(f"name+args exact   : {sum(r['exact_ok'] for r in rows)}/{n}")
    print(f"matches host      : {sum(r['agrees_with_host'] for r in rows)}/{n}")
    secs = [r["seconds"] for r in rows]
    print(f"seconds per case  : median {sorted(secs)[len(secs)//2]:.0f}, "
          f"min {min(secs):.0f}, max {max(secs):.0f}")
    json.dump(rows, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
