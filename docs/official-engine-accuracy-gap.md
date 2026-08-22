# Official Needle accuracy gap: root-cause report

This report explains the former `48.8%` MimiModel result against the official
Needle engine's `69.2%` on all 961 `google/mobile-actions` eval rows. The final
ESP32-compatible configuration reaches `669/961` (`69.6%`) under the same strict
scorer. That headline does not mean the engines now make identical mistakes.

## Fair comparison

Both engines received the same ordered tool schemas, separate developer and user
turns, preserved whitespace, queries, and expected outputs. Results were rescored
from raw calls with ordered strict equality. The official run used
`cactus-needle` 2.0.6 with native engine 2.0.2.

The weights are identical. The official dylib's embedded `needle_weights` and
`model/needle2.cact` are both 13,737,807 bytes and have SHA-256:

```text
b43aabfcaf1a6db6acf488076eab71d823c08697c7af4521fc1d174b60ede5ba
```

This rules out model export and quantization as causes of the 20.4-point gap.
Fast math is also unrelated: the baseline host run used ordinary `-O3`.

Engine 2.0.2 is a closed dylib, so its per-layer activations cannot be sampled
directly. Its debug mode does expose selected tools, prompt/window lengths,
first-token top logits, reasoning, and final calls. The investigation therefore
compares the last identical boundary (input token IDs) and first observable
divergence (prompt-end logits/generation), then confirms each cause by ablation.

## What the baseline got wrong

| Error class | MimiModel baseline | Official 2.0.2 | Net gap |
|---|---:|---:|---:|
| Too few calls | 137 | 9 | 82 paired rows |
| Too many calls | 37 | 4 | 21 paired rows |
| Wrong tool name | 28 | 5 | 16 paired rows |
| Argument values | 235 | 179 | 85 paired rows |
| Argument keys and values | 55 | 63 | 16 paired rows |
| Argument keys only | 0 | 36 | -24 paired rows |

On paired rows, the official engine alone passed 244 and MimiModel alone passed
48, a net difference of 196 rows. Call count explains 103 rows, tool selection
16, and arguments 77. The weak result was not one broad floating-point drift.

## Controlled ablations

Accuracy runs below used the same model, dataset, prompt builder, metric, and
ordinary host `-O3` binary. They ran concurrently, so their latency fields are
discarded; speed is measured separately.

| Configuration | Strict | Names | Change |
|---|---:|---:|---:|
| Recent 256, reasoning cap 90, segmented decoder | 469/961 (48.8%) | 759/961 (79.0%) | baseline |
| Recent 256, reasoning cap 256, segmented decoder | 513/961 (53.4%) | 839/961 (87.3%) | +44 rows |
| Full prefix sink, reasoning cap 90, segmented decoder | 516/961 (53.7%) | 792/961 (82.4%) | context-only cross-check |
| Full prefix sink, reasoning cap 256, segmented decoder | 576/961 (59.9%) | 877/961 (91.3%) | +63 after reasoning |
| Full prefix sink, reasoning cap 256, byte grammar | 672/961 (69.9%) | 876/961 (91.2%) | +96 after context |
| 160-token sink, reasoning cap 256, byte grammar | **669/961 (69.6%)** | 873/961 (90.8%) | -3 vs full sink |
| Official engine 2.0.2 | 665/961 (69.2%) | **943/961 (98.1%)** | reference |

### 1. The reasoning cutoff was truncating valid generations

The old loop allowed 90 reasoning tokens and then entered its JSON decoder even
when the model had not emitted `<tool_call>`. With the full prefix present, 139
of 961 rows did not open the marker by token 90. At a cap of 256, 960 rows opened
it; the remaining row also fails in the official engine. Raising the cap adds no
work to the median case because generation stops as soon as the marker appears.

### 2. Sliding-window attention dropped prompt instructions

The old KV implementation attended only to the latest 256 tokens. Official debug
output reports a protected prefix plus a 256-token recent window. On row 8, both
engines have exactly the same 333 input token IDs: a 230-token prefix and a
103-token turn. The old implementation therefore discarded the first 77 tokens
at the end of prefill, including the date-bearing system instruction.

With a protected prefix, the first-token logits move toward the official values.
With the full prefix and reasoning cap 256, the representative row reproduces the
official reasoning and both calls exactly. A 160-token protected prefix retains
the same 416 physical KV rows as the previous `256 + 160` ring allocation and
loses only three strict matches over the full-prefix experiment. The full prefix
would need up to 522 rows and roughly 1.56 MB more PSRAM.

The final configuration was also flashed to an 8 MB PSRAM ESP32-S3. Row 8
processed 333 prompt tokens and 216 decode tokens in 413.742 seconds, returned
both calls strict-exact, and matched the fast-math host JSON. This verifies the
memory-constrained prefix sink on the actual target, not only on arm64.

### 3. Segmented teacher-forcing changed the decoder state

The official package documents one byte-level grammar compiled from the schemas.
The old C path forced JSON fragments separately, decoded names through a trie,
decoded each key/value in another loop, and used a hand-tuned logit margin to
decide whether to append a call. Tokens that naturally cross JSON boundaries
were split into a different token history. This caused swapped contact fields,
bad values, and unstable call counts even with the correct attention context.

The new decoder validates every byte of every candidate token against one
continuous schema state. It does not force structural fragments between model
steps. This single change recovers 96 paired rows after the context and reasoning
fixes, the largest isolated increment.

## Retrieval is secondary

The native BM25 simulation retains every expected tool in 914/961 rows. Of the
137 baseline under-call rows, 36 have a retrieval miss and 101 still have full
retrieval recall. Supplying both engines the same BM25 top-2 candidates changes
the gap from 196 to 172 rows. Retrieval mismatch therefore accounts for about 24
rows of the former gap, not the majority.

After the decoder fixes, the 69 remaining under-calls split into 33 retrieval
misses and 36 despite full recall. The decoder/context work reduced the latter
bucket from 101 to 36; BM25 now explains almost half of the residual under-calls.
The official package documents a learned top-5 retrieval head, so retrieval is
the next target for closing name accuracy, not another grammar margin.

The final engine still has lower name accuracy than the official engine
(`90.8%` versus `98.1%`) and more under-calls (`69` versus `9`). It wins more
argument rows under this exact scorer, which makes strict accuracy slightly
higher despite different error distributions. Treat `69.6%` as benchmark parity,
not token-for-token equivalence with the closed engine.

## Reproduction artifacts

- Baseline: `bench/results/ours_961_20260822.json`
- Official: `bench/results/oracle_961_protocol_v2.json`
- Final source run: `bench/results/ours_961_final_20260822.json`
- 160-token sink ablation: `bench/results/ours_961_sink160_reason256_grammar_20260822.json`
- Paired baseline report: `bench/results/accuracy_gap_profile_20260822.json`
- Paired final report: `bench/results/accuracy_gap_final_profile_20260822.json`
- Retrieval report: `bench/results/retrieval_profile_20260822.json`
- Final retrieval report: `bench/results/retrieval_profile_final_20260822.json`
- ESP32-S3 row 8: `bench/results/esp32s3_sink160_grammar_row8_20260822.json`
- Analysis tools: `bench/profile_accuracy_gap.py`, `bench/profile_retrieval.py`

Runtime controls for reproducing old behavior are
`NEEDLE_PREFIX_SINK=0`, `NEEDLE_REASON_MAX=90`, and
`NEEDLE_BYTE_GRAMMAR=0`.
