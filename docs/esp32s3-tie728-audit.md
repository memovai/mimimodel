# ESP32-S3 TIE728 performance audit

This record fixes the workload and build settings behind the README speed claim. It is a timing
and arithmetic-parity audit, not an estimate of google/mobile-actions accuracy.

## Test conditions

| Item | Value |
|---|---|
| Date | 2026-08-22 |
| Source | commit `e52c9cd` |
| Board | ESP32-S3 rev 0.2, 240 MHz, 16 MB QIO flash, 8 MB 80 MHz Octal PSRAM |
| Firmware | ESP-IDF 5.5.2, `fast_math=0`, `profile=0`, `sinkhorn_iters=20` |
| Model | Needle 2, CQ 2-bit, weights memory-mapped from the `needle` flash partition |
| Cache protocol | same process and serial connection; first request misses, second request hits |

The two requests used this exact query and tool schema:

```text
query: Turn on the flashlight
tools: [{"name":"turn_on_flashlight","description":"Turns the flashlight on.","parameters":{"type":"object","properties":{}}}]
```

## End-to-end result

| Run | Prefix cache | Prefill | Decode | Total | Output |
|---|---:|---:|---:|---:|---|
| Cold | miss | 51 tok at 1.93 tok/s | 16 tok at 1.62 tok/s | 36.365 s | exact |
| Warm | hit | 13 tok at 1.86 tok/s | 16 tok at 1.62 tok/s | 16.875 s | exact |

Both runs emitted:

```json
[{"name":"turn_on_flashlight","arguments":{}}]
```

The pre-TIE728 fair build measured 42.895 s cold and 20.067 s warm on the same fixed workload.
The current build reduces those times by 15.2% and 15.9%, respectively.

## CQ2 kernel result

| Kernel | 512x512 time | Change |
|---|---:|---:|
| Scalar C, one core | 5.272 ms | baseline |
| TIE728 aligned-load, one core | 3.781 ms | 28.3% lower |
| Scalar C path, two cores | about 2.700 ms | baseline |
| TIE728 aligned-load, two cores | 1.960 ms | 27.4% lower |
| Dense int16 PIE experiment, one core | 31.690 ms | rejected |

At boot, the TIE728 CQ2 kernel is checked against scalar C. The maximum absolute error in this run
was `6.676e-06`. The faster path keeps byte-LUT CQ2 decoding and uses aligned vector float loads
with two rows and eight accumulators. It does not widen every 2-bit weight to an int16 lane.

The instruction syntax and scheduling were cross-checked against Espressif's
[esp-dl TIE728 kernels](https://github.com/espressif/esp-dl/tree/master/esp-dl/dl/base/isa/tie728).

## Raw console

```text
[needle] build: fast_math=0 profile=0 sinkhorn_iters=20
[needle] loaded in 60 ms: 27 layers, d_model 512, vocab 8192, kv_window 256
[tie728] CQ2 aligned-load max abs err 6.676e-06 -> PASS
[pie] matvec 512x512x2b: float 5272 us, int16 31690 us (0.17x)
[tie728] matvec aligned-load 3781 us vs C 5272 us (1.39x)

[needle] call: [{"name":"turn_on_flashlight","arguments":{}}]
[needle] prefix cache: miss
[needle] total 36365 ms | prefill 51 tok 1.93 tok/s | decode 16 tok 1.62 tok/s

[needle] call: [{"name":"turn_on_flashlight","arguments":{}}]
[needle] prefix cache: hit
[needle] total 16875 ms | prefill 13 tok 1.86 tok/s | decode 16 tok 1.62 tok/s
```
