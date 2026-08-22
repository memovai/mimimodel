# ESP32-S3 TIE728 performance audit

This record fixes the workload and build settings behind the README speed claim. It is a timing
and arithmetic-parity audit, not an estimate of google/mobile-actions accuracy.

## Test conditions

| Item | Value |
|---|---|
| Date | 2026-08-22 |
| Firmware SHA-256 | `68e4c0467238ccaa5e51643fb4c713379fea5849d2b0656cbf792886b8e5741f` |
| Board | ESP32-S3 rev 0.2, 240 MHz, 16 MB QIO flash, 8 MB 80 MHz Octal PSRAM |
| Firmware | ESP-IDF 5.5.2, `fast_math=1`, `profile=0`, `sinkhorn_iters=20`, prefix sink 160, reasoning cap 256, byte grammar |
| Model | Needle 2, CQ 2-bit, weights memory-mapped from the `needle` flash partition |
| Cache protocol | same process and serial connection; first request misses, second request hits |

The two requests used this exact query and tool schema:

```text
query: Turn on the flashlight.
tools: [{"name":"turn_on_flashlight","description":"Turns the flashlight on.","parameters":{"type":"object","properties":{}}}]
```

## End-to-end result

| Run | Prefix cache | Prefill | Decode | Total | Output |
|---|---:|---:|---:|---:|---|
| Cold | miss | 52 tok at 1.94 tok/s | 14 tok at 1.60 tok/s | 35.480 s | exact |
| Warm | hit | 14 tok at 1.88 tok/s | 14 tok at 1.60 tok/s | 16.170 s | exact |

Both runs emitted:

```json
[{"name":"turn_on_flashlight","arguments":{}}]
```

Against the pre-TIE728 fast-math build at 41.541/19.444 s, the current default lowers
cold/warm time by 14.6%/16.8%. The changed grammar also emits two fewer tokens on this prompt,
so this is an end-to-end product comparison, not a kernel-only speedup.

## CQ2 kernel result

| Kernel | 512x512 time | Change |
|---|---:|---:|
| Scalar C with fast math, one core | 4.735 ms | baseline |
| TIE728 aligned-load, one core | 3.592 ms | 24.1% lower |
| Scalar C path, two cores | about 2.700 ms | baseline |
| TIE728 aligned-load, two cores | 1.960 ms | 27.4% lower |
| Dense int16 PIE experiment, one core | 31.682 ms | rejected |

At boot, the TIE728 CQ2 kernel is checked against scalar C. The maximum absolute error in this run
was `8.583e-06`. The faster path keeps byte-LUT CQ2 decoding and uses aligned vector float loads
with two rows and eight accumulators. It does not widen every 2-bit weight to an int16 lane.

The instruction syntax and scheduling were cross-checked against Espressif's
[esp-dl TIE728 kernels](https://github.com/espressif/esp-dl/tree/master/esp-dl/dl/base/isa/tie728).

## Device parity sample

The final decoder ran the complex two-call row 8 with dataset tool order, native retrieval, and
strict scoring. It matched the fast-math host byte-for-byte and was strict-exact. The three older
rows below belong to the preceding TIE728 binary and remain as historical parity evidence. These
selected rows check portability; they do not estimate population accuracy.

| Row | Time | Prefill | Decode | Name | Strict | Host parity |
|---:|---:|---:|---:|---:|---:|---:|
| 8 (final decoder) | 413.742 s | 1.57 tok/s | 1.10 tok/s | pass | pass | pass |
| 0 | 319.648 s | 1.58 tok/s | 1.24 tok/s | pass | fail | pass |
| 1 | 255.435 s | 1.59 tok/s | 1.24 tok/s | pass | fail | pass |
| 9 | 169.074 s | 1.65 tok/s | 1.24 tok/s | pass | pass | pass |

Raw rows and reproducibility metadata are stored in
[`bench/results/esp32s3_sink160_grammar_row8_20260822.json`](../bench/results/esp32s3_sink160_grammar_row8_20260822.json),
[`bench/results/esp32s3_fast_tie_rows0_1_20260822.json`](../bench/results/esp32s3_fast_tie_rows0_1_20260822.json)
and [`bench/results/esp32s3_fast_tie_row9_20260822.json`](../bench/results/esp32s3_fast_tie_row9_20260822.json).

## Raw console

```text
[needle] build: fast_math=1 profile=0 sinkhorn_iters=20 prefix_sink=160 reason_max=256 byte_grammar=1
[needle] loaded in 59 ms: 27 layers, d_model 512, vocab 8192, kv_window 256
[tie728] CQ2 aligned-load max abs err 8.583e-06 -> PASS
[pie] matvec 512x512x2b: float 4735 us, int16 31682 us (0.15x)
[tie728] matvec aligned-load 3592 us vs C 4735 us (1.32x)

[needle] call: [{"name":"turn_on_flashlight","arguments":{}}]
[needle] prefix cache: miss
[needle] total 35480 ms | prefill 52 tok 1.94 tok/s | decode 14 tok 1.60 tok/s

[needle] call: [{"name":"turn_on_flashlight","arguments":{}}]
[needle] prefix cache: hit
[needle] total 16170 ms | prefill 14 tok 1.88 tok/s | decode 14 tok 1.60 tok/s
```
