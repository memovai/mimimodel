# Benchmark audit: 2026-08-22

This audit separates host accuracy, host speed, and ESP32-S3 speed. The three runs use different
workloads and their throughput numbers must not be mixed.

## Environment

| Item | Value |
|---|---|
| Host | Apple M4, 10 CPU cores, 16 GB RAM, macOS 26.5.1 |
| Compiler | Apple clang 21.0.0, `cc -O3 -o /tmp/needle_final needle.c -lm` |
| Python | 3.13.5 |
| Official package | `cactus-needle` 2.0.6, engine 2.0.2 |
| Dataset SHA-256 | `91d251ee958cfd295af6c4504c236a3a1ad19517de240c3bc680bacfcbf7e7d9` |
| Model SHA-256 | `b43aabfcaf1a6db6acf488076eab71d823c08697c7af4521fc1d174b60ede5ba` |
| Current host binary SHA-256 | `91ba7850a573476f161476f204b56a457459fc8e68ca5449f0132df8c87f15e0` |
| Official engine SHA-256 | `f8ff35e0ceea5812f3f90bddbb40ca3a8d03af07c84555e116bf0096fa994afd` |

## Host accuracy

Command:

```bash
.venv/bin/python bench/mobile_actions.py --out bench/results/ours_961_final_20260822.json
```

Protocol: all 961 eval rows, dataset tool order, native retrieval, ordered strict exact match,
separate developer/user turns, and original whitespace.

| Metric | Current engine | Official engine 2.0.2 |
|---|---:|---:|
| Strict exact | 669/961 (69.6%) | 665/961 (69.2%) |
| Tool names | 873/961 (90.8%) | 943/961 (98.1%) |
| 1-call strict | 488/640 (76.2%) | 471/640 (73.6%) |
| 2-call strict | 180/320 (56.2%) | 193/320 (60.3%) |

The official accuracy artifact was not rerun because its engine and dataset hashes are unchanged.
It is directly comparable under the same saved protocol.

### Accuracy attribution

The old engine scored 469/961. The official dylib and local model contain byte-identical weights.
Controlled paired runs attribute the recovery to a 256-token reasoning cap (+44 rows), a protected
prompt prefix (+63 after reasoning), and a continuous byte grammar (+96 after context). Capping the
prefix at 160 tokens loses three rows against the full-prefix run while retaining the old 416-row KV
allocation. See [the root-cause report](official-engine-accuracy-gap.md).

## Host speed

Commands, run separately with no competing benchmark process:

```bash
.venv/bin/python bench/speed.py --limit 200 --binary /tmp/needle_final --out bench/results/speed_ours_final_20260822.json
.venv/bin/python bench/speed.py --limit 200 --oracle --out bench/results/speed_oracle_20260822.json
```

Protocol: first 200 eval rows, canonical tool order, native retrieval, serial requests.

| Metric | Current engine | Official engine 2.0.2 |
|---|---:|---:|
| Completion median | 2259 ms | 665 ms |
| Completion P90 | 3122 ms | 877 ms |
| Request median including init | 2259 ms | 948 ms |
| Prefill | 191 tok/s | 1204 tok/s |
| Decode | 141 tok/s | 702 tok/s |

The official engine's median initialization was 293 ms. The custom engine reports parse and native
BM25 retrieval inside completion latency.

## ESP32-S3

The fastest-default fixed one-tool result is 1.94 tok/s prefill, 1.60 tok/s decode,
35.480 s cold, and 16.170 s warm. The final prefix-sink firmware also completed mobile-actions
row 8 strict-exact in 413.742 s: 333 prompt tokens, 216 decode tokens, two calls, and raw host
parity. Device samples establish target feasibility and parity, not population accuracy. The
earlier kernel audit remains in [ESP32-S3 TIE728 performance audit](esp32s3-tie728-audit.md).
