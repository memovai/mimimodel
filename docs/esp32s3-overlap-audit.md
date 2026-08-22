# ESP32-S3 operator-overlap audit

This audit records the scheduling experiment inspired by FlashAttention and DFlash2. The retained
change overlaps independent work across the ESP32-S3's two cores. It does not change model math,
quantization, the prompt, or decoder settings.

## Result

| Workload | Previous default | Overlap build | Time reduction | Output |
|---|---:|---:|---:|---|
| Fixed one-tool, cold | 35.480 s | **33.382 s** | **5.91%** | exact |
| Fixed one-tool, warm | 16.170 s | **15.266 s** | **5.59%** | exact |
| mobile-actions row 9 | 169.074 s | **158.111 s** | **6.48%** | strict exact, host parity |

The warm result is the median of four runs; all four measured exactly 15.266 s. The fixed query
improved from 1.94/1.60 to **2.07/1.69 prefill/decode tok/s** on the cold run. The real
mobile-actions case improved from 1.65/1.24 to 1.75/1.29 tok/s.

## Conditions

| Item | Value |
|---|---|
| Date | 2026-08-22 |
| Final firmware SHA-256 | `525cbc75ffecc51d314337671c520db8259e71aaccb8fadeef352034b65462aa` |
| Repeat-run firmware SHA-256 | `0cb1b92084a36c392cc333ff1fa0046c86d6181ebec1c52448f0ab0625b81cd2` |
| Board | ESP32-S3 rev 0.2, 240 MHz, 16 MB QIO flash, 8 MB 80 MHz Octal PSRAM |
| Firmware | ESP-IDF 5.5.2, `fast_math=1`, `profile=0`, prefix sink 160, reasoning cap 256, byte grammar |
| Scheduling | mHC overlap on, Q lead 128 rows; gate overlap on, lead 32 rows |
| Model | Needle 2 CQ2, 13.7 MB weights memory-mapped from flash |

The fixed workload used the same query, one-tool schema, serial process, and cache protocol as the
[TIE728 baseline audit](esp32s3-tie728-audit.md). Only the firmware scheduling changed.
The final rebuild and repeat-run image differ only in the 32-byte embedded ELF identifier and the
33-byte trailing image digest; a flash readback confirmed their executable payloads are identical.
The final image independently measured 33.381 s cold and 15.266 s warm with the same output.

## What overlaps

At each layer, `phi_post`, `phi_res`, the post sigmoid, and Sinkhorn depend on the already prepared
layer input. Core 1 now computes that chain while core 0 performs hpre mixing, engram injection,
attention normalization, and Q preparation. Core 0 computes the first 128 Q rows before joining the
worker, which hides the tail of the mHC job without leaving the second core idle.

After V projection, core 1 computes the first 32 gate rows while core 0 applies Q/K normalization,
RoPE, and int8 KV-cache quantization. Both jobs join before attention reuses core 1. Each projection
retains its original row order and floating-point accumulation order.

With profiling enabled, mHC-post attribution fell from 3157.720 to 694.685 ms during cold prefill
and from 851.749 to 187.132 ms during decode. Some wait time moves into the Q phase, so phase totals
cannot be compared independently; profiled end-to-end time fell from 35.638 to 33.686 s before the
smaller gate overlap was added.

## FlashAttention and DFlash2 boundary

[FlashAttention](https://arxiv.org/abs/2205.14135) and
[FlashAttention-2](https://arxiv.org/abs/2307.08691) derive their gains from tiled, IO-aware
attention. Needle 2 decode here is batch 1, keeps at most 416 physical KV rows, and already computes
one score row split by KV head. Its two score buffers need about 3.3 KB, and measured attention is
only 6-9% of wall time. A tiled rewrite therefore has little memory traffic or total runtime left to
remove.

[DFlash](https://arxiv.org/abs/2602.06036) and
[DFlash2](https://docs.nvidia.com/nemo/automodel/nemo-automodel/nemo_automodel/components/speculative/dflash/draft_qwen3_dflash2)
motivate block verification for speculative decoding.
That requires a target path which evaluates several proposed tokens together. A two-token CQ2
assembly prototype reused each packed weight load for two activation vectors and passed at maximum
absolute error `8.583e-06`, but improved the paired 512x512 projection only from 7.184 to 6.481 ms,
or **1.11x**. End-to-end blocked prefill would also need batched normalization, mHC, causal
attention, KV writes, MLP, and logits. That prototype was removed because the measured kernel gain
does not justify the added memory and correctness surface.

The useful idea was finer-grained scheduling of already independent operators, where the existing
single-token engine can gain speed without changing numerical behavior.

## Correctness evidence

- The fixed cold run and four warm runs emitted the same flashlight JSON.
- mobile-actions row 9 remained strict exact and matched the host byte for byte.
- The host build preserves the original serial execution path; 20 raw mobile-actions cases matched
  their prior outputs, scores, and token counts.
- The Python benchmark/CLI suite passed 23 tests, and the byte-grammar C test passed.
- The raw device row and hashes are in
  [`bench/results/esp32s3_overlap_row9_20260822.json`](../bench/results/esp32s3_overlap_row9_20260822.json).
- The five fixed-query runs are in
  [`bench/results/esp32s3_overlap_fixed_20260822.json`](../bench/results/esp32s3_overlap_fixed_20260822.json).

These device checks establish arithmetic parity for selected cases. The published 961-case host
accuracy remains 69.6%; a one-row device run is not an accuracy estimate.
