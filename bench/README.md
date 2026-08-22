# Benchmarks

Two suites live here:

| Suite | What it measures | Runtime |
|---|---|---|
| `mobile_actions.py` | accuracy on [google/mobile-actions](https://huggingface.co/datasets/google/mobile-actions), 961 eval cases | ~35 min host |
| `speed.py` | inference throughput and latency, this engine vs the official one | ~8 min |
| `run_bench.py` | a hand-written 24-case smoke test (device-control flavoured) | ~2 min |
| `device_runner.py` | drives the ESP32-S3 over serial and checks it byte-for-byte against the host | ~5 min per case |

Raw per-case outputs of every published run are in `results/`.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install numpy sentencepiece pyserial

# the oracle: the official closed-source engine, used as the comparison point
.venv/bin/pip install cactus-needle

# weights (13.7 MB, Apache-2.0, not redistributed in this repo)
mkdir -p model && curl -Lo model/needle2.cact \
  https://huggingface.co/Cactus-Compute/needle2/resolve/main/needle2.cact

# the engine binary the harnesses call
cc -O3 -o /tmp/needle_ma needle.c -lm

# fast grammar-state regression (no model inference)
cc -O2 -o /tmp/test_byte_grammar bench/test_byte_grammar.c -lm
/tmp/test_byte_grammar
```

The mobile-actions dataset (25 MB) downloads itself on first run into
`bench/mobile_actions.jsonl`.

Oracle versions matter. New result sidecars record the Python package, native
engine, dataset, model, and binary hashes. The current audit used package 2.0.6
loading engine 2.0.2. The historical artifact recorded none of this, so its exact
official build cannot be proven after the fact.

## Accuracy

```bash
# this engine, all 961 eval cases
.venv/bin/python bench/mobile_actions.py --out bench/results/mine.json

# the official engine on the identical prompts, as a reference point
.venv/bin/python bench/mobile_actions.py --oracle --out bench/results/oracle.json

# quick iteration
.venv/bin/python bench/mobile_actions.py --limit 300
```

The final 2026-08-22 rerun gives:

```
=== google/mobile-actions eval · this engine · 961 cases ===
accuracy (ordered strict exact, all 961): 669/961 (69.6%)   name acc 873/961 (90.8%)
  1-call (640): exact 488 (76.2%)   names 629 (98.3%)
  2-call (320): exact 180 (56.2%)   names 243 (75.9%)
  3-call (  1): exact   1 (100.0%)  names   1 (100.0%)
```

### Where it stands

| | this engine | official engine 2.0.2, identical inputs |
|---|---|---|
| strict accuracy | **69.6%** | 69.2% |
| name accuracy | 90.8% | **98.1%** |
| 1-call strict | **76.2%** | 73.6% |
| 2-call strict | 56.2% | **60.3%** |

The weights embedded in the official engine and `model/needle2.cact` are
byte-identical. The old `48.8%` result came from three engine differences:

| Controlled change | Strict result | Increment |
|---|---:|---:|
| old recent-window context, 90-token reasoning cap, segmented decoder | 469/961 | baseline |
| allow up to 256 reasoning tokens | 513/961 | +44 |
| retain the prompt prefix alongside the recent window | 576/961 | +63 |
| replace segmented teacher-forcing with one continuous byte grammar | 672/961 | +96 |
| cap the protected prefix at 160 tokens for ESP32 memory | **669/961** | -3 |

The 160-token prefix uses the same 416 physical KV rows as the old ring. A full
prefix needs up to 522 rows and about 1.56 MB more PSRAM. The cap keeps nearly all
of the accuracy gain without increasing the ESP32 KV allocation.

This is strict-score parity, not output parity. The final engine still under-calls
more often and has lower name accuracy, while it wins more argument rows. The
paired error taxonomy, first-divergence traces, retrieval ablation, and exact
reproduction files are in
[the official-engine gap report](../docs/official-engine-accuracy-gap.md).

The former 76.9% official number was not a valid re-score: it summed stale flags
in `oracle_961.json`. Run `python bench/mobile_actions.py --rescore <file>` to
derive metrics from the saved raw outputs.

Cactus publishes 63.7% exact / 98.3% names for Needle 2 on the same named split
and strict metric. The public package scores 69.2%/98.1% here. The site does not
publish its prompt/schema conversion, selected-tool traces, raw rows, or binary
hash, so the 5.5-point exact difference cannot be attributed. Treat the website
number as an external reference, not a third directly comparable column.

### Multi-call

321 of 961 rows expect more than one call, so a single-call engine has a 66.6%
ceiling. The default byte grammar lets the model choose `,` or `]` in the same
continuous constrained stream and caps the result at four calls. It no longer uses
the legacy `NEEDLE_CONT_MARGIN` heuristic. The final 2-call bucket is 180/320
strict, up from 84/320 in the segmented baseline.

### The metric

`accuracy` is **ordered strict exact match**: function names, call order and every argument must match. A turn
that expects two calls and gets one is a miss. `name acc` is the same comparison
ignoring argument values.

Strict is now the default. `--metric normalized` reproduces the old
`str(value).strip().lower()` behavior for historical comparisons.

### Protocol audit

The seven schemas are identical across eval records, but their order is randomized:
961 rows contain 881 ordered variants. The old harness always used row zero's
ordering and flattened developer newlines. On the first 100 cases, preserving the
dataset order changed 25 self-engine outputs and 22 official-engine outputs. Fixed
order moved strict accuracy 54% to 56% here and 70% to 72% officially; wins and
losses both occurred, so this is measurement bias rather than an optimization.

`--tool-order dataset` is the accuracy default. `canonical` sorts by name for a
controlled speed workload, and `fixed-first` exists only to reproduce old runs.
The result sidecar records the metric and protocol plus dataset, model, binary,
official package, and engine hashes.

Native mode still compares two different retrieval policies. For decoder-focused
comparison, `--retrieval common-bm25-2` preselects the same two schemas before
calling either engine, so neither native retriever engages. Its full-answer recall
ceiling is 895/961 (93.1%): 637/640 single-call and 258/321 multi-call cases. On the
first 100 rows it scored 56% strict / 87% names here, versus 73% / 96% officially.
This isolates decoder behavior but is not a replacement for native end-to-end eval.

### Why the harness looks the way it does

Two details cost a lot of accuracy if you get them wrong, and both are easy to get
wrong:

**The developer turn is a system message, not a prefix.** The dataset's records
carry a `developer` turn with the current date, which several cases need ("this
Friday"). Needle's trained format (`needle/model/finetune.py: render_example`) has a
dedicated system turn ahead of the user turn. Gluing that text onto the front of
the query instead is worth **13 points of tool-name accuracy on the official
engine** (85% → 100% on a 200-case sample). `build_turn()` returns the two parts
separately and both engines receive them the same way.

**The tool list has to be pruned.** Needle attends over a 256-token sliding window.
The 7-tool block in this dataset is 417 tokens, so selection degrades without a
retrieval stage. The official Needle package documents a learned top-5 retrieval
head when more than five tools are declared. This engine uses BM25, which gets
97.1% recall@3 on this set without another model forward pass. It kicks in only
when the block exceeds `NEEDLE_TOOLS_BUDGET` (180 tokens), so small tool sets pass
through untouched and keep the KV prefix cache warm.

## Speed

Run these **alone**. Anything else on the machine skews them; a run contaminated by
a second benchmark process reported latencies roughly 2× too high.

```bash
.venv/bin/python bench/speed.py --limit 200
.venv/bin/python bench/speed.py --limit 200 --oracle
```

The old table reported 166/59 tok/s by dividing prefill and decode token counts
separately by the same whole-request wall time. Those are not phase throughputs.
The batch protocol now exports the phase timers already maintained by `needle.c`.

Current-source audit on an Apple M4 Mac (16 GB), first 200 cases with canonical
tool order and native retrieval, run serially on 2026-08-22:

| | this engine | official engine | ratio |
|---|---|---|---|
| completion latency, median | 2259 ms | 665 ms | 3.4× |
| completion latency, P90 | 3122 ms | 877 ms | 3.6× |
| request latency incl. init | 2259 ms | 948 ms | 2.4× |
| prefill | 191 tok/s | 1204 tok/s | 6.3× |
| decode | 141 tok/s | 702 tok/s | 5.0× |

Official `Needle(...)` initialization costs a 293 ms median in this run. The C
engine's parse and BM25 retrieval remain inside its completion measurement. Raw
rows and hashes are in `results/speed_ours_final_20260822.json` and
`results/speed_oracle_20260822.json`.

In the earlier 100-case audit, common BM25 top-2 measured 1509 ms versus
98 ms, or 186 ms for the official engine including its 79 ms initialization. Phase
throughput was 252/196 tok/s here and 1979/1296 officially. This is the cleaner
kernel/decoder comparison; native mode is the relevant product comparison.

The instruction-level comparison still is not symmetric: this engine is portable
scalar C99 and the official build is ARM64/NEON optimized.

### Optimization priority

A three-call host profile attributes 56% of forward time to Q/K/V/gate CQ
matvecs and another 19% to out projection plus Hadamard work. mHC was 9%,
attention itself 6%, logits 5%, and engram 3%. The exact split will move on the
ESP32 because flash and PSRAM bandwidth differ, but it makes the first target
clear: optimize packed-weight reads, CQ decode/MAC reuse, and hot-matrix caching
before spending time on attention.

Compiler flags were not a shortcut on the tested M4 host. `-O3`,
`-O3 -mcpu=native`, and `-O3 -mcpu=native -flto` were effectively tied;
`-Ofast -mcpu=native -flto` was about 43% slower on the same warm sample. Keep
`-O3` as the baseline and require an output-parity check for every kernel change.

## On the ESP32-S3

```bash
cd needle-esp32s3
# fastest product build (default): fast math, no profiler
idf.py -DNEEDLE_FAST_MATH=ON -DNEEDLE_PROFILE=OFF build
# reproducibility build: IEEE-style math, no profiler
idf.py -DNEEDLE_FAST_MATH=OFF -DNEEDLE_PROFILE=OFF build
# hotspot profiling build
idf.py -DNEEDLE_FAST_MATH=ON -DNEEDLE_PROFILE=ON build
idf.py -p /dev/tty.usbXXX flash
cd .. && .venv/bin/python bench/device_runner.py --port /dev/tty.usbXXX \
  --sample 12 --seed 20260819 --timeout 1200 --allow-nonstandard
```

`device_runner.py` sends each case over the serial REPL and compares the board's
answer against the host engine's on the same case. Both run the same `needle.c`
against the same weights, but Xtensa and arm64 can still take different greedy
branches. Running all 961 on the board would take about three days, so device
samples must report confidence intervals and both raw-output and score parity.

### 2026-08-22 fastest-default audit

The final `fast_math=1`, `profile=0`, 160-token prefix-sink firmware measured a
fixed one-tool workload at 35.480 s cold and 16.170 s warm. Cold prefill/decode
was 1.94/1.60 tok/s; warm was 1.88/1.60. Both outputs were identical and the
TIE728 boot self-test passed at 8.583e-06 maximum absolute error.

The accuracy-attribution row 8 also ran on the board. It consumed 333 prompt and
216 decode tokens in 413.742 s, returned both calls strict-exact, and matched the
fast-math host output. Its artifact is
`results/esp32s3_sink160_grammar_row8_20260822.json`.

The three rows below are the earlier TIE728 speed/parity audit retained for
historical comparison:

| row | time | prefill | decode | name | strict | host parity |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 319.648 s | 1.58 tok/s | 1.24 tok/s | pass | fail | pass |
| 1 | 255.435 s | 1.59 tok/s | 1.24 tok/s | pass | fail | pass |
| 9 | 169.074 s | 1.65 tok/s | 1.24 tok/s | pass | pass | pass |

This is 3/3 raw host parity and score parity, not a population accuracy estimate.
Exact payloads, hashes, and raw rows are in the dated artifacts under `results/`.

### 2026-08-19 protocol audit

Real board: ESP32-S3 rev 0.2 at 240 MHz, 16 MB flash, 8 MB octal PSRAM at
80 MHz. With dataset tool order, native retrieval, preserved developer
whitespace, and strict scoring:

| workload | `-O3` | `-O3 -ffast-math` | change |
|---|---:|---:|---:|
| fixed one-tool schema, cold | 42.895 s | 41.541 s | -3.2% |
| same schema, prefix-cache hit | 20.067 s | 19.444 s | -3.1% |
| mobile-actions row 0 | 364 s | 352.4 s | -3.2% |
| mobile-actions row 1 | 292 s | 282.8 s | -3.2% |

The short workload measured 1.65 prefill / 1.41 decode tok/s at baseline and
1.70 / 1.46 tok/s with fast math. Three pre-TIE728 fast-math mobile-actions rows all
matched the standard host output byte-for-byte; row 9 was also a strict-exact
dataset success and took 189.2 s. Three rows are a parity check, not an accuracy
estimate. This historical audit kept fast math opt-in; it is now the product
default, while IEEE-style math remains available for controlled reproduction.

Board profiling puts Q/K/V/gate at 48-52%, out projection plus Hadamard at
17-18%, mHC at 16-17%, and attention at only 6-9%. The existing dual-core split
is effective: the same 512x512 CQ2 matvec measured 5.004 ms single-core and
2.559 ms dual-core (1.96x). Rejected experiments are recorded with the raw audit
in `results/device_protocol_audit_20260819.json`: PIE int16 was 3-6x slower and
changed decode length, an eight-row float kernel was 25% slower on-device, and
eight Sinkhorn iterations changed all first ten host outputs.

The fair build (`fast_math=0`, `profile=0`) was then run on a proportional,
stratified random sample: seed 20260819, eight 1-call rows and four 2-call rows.

| sampled device result | value |
|---|---:|
| strict exact | 5/12 (41.7%; Wilson 95% CI 19.3-68.0%) |
| name accuracy | 9/12 (75.0%; Wilson 95% CI 46.8-91.1%) |
| byte-for-byte host parity | 10/12 |
| strict/name pass-fail parity | 12/12 |
| latency | median 315.5 s (203.1-357.7) |
| throughput | median 1.39 prefill / 1.11 decode tok/s |

This is real device accuracy on those 12 rows, but the interval is too wide to
claim a population accuracy. Two contact-extraction rows took different wrong
token branches on Xtensa and arm64 while retaining the same name/exact outcome.
After eight continuous rows, a ninth request also exceeded the original 650 s
timeout; resetting the board and rerunning the same row completed in 355.9 s.
The timeout is excluded from the denominator and retained in
`esp32s3_mobile_actions_interrupted_timeout_20260819.json`. The runner now aborts
on timeout instead of sending another query into an unaligned serial stream and
supports `--indices` for deterministic resume.

The older legacy-protocol run measured (ESP32-S3, 240 MHz, 8 MB PSRAM, 12 cases,
firmware and host binary built
from the same source):

| | value |
|---|---|
| per call | median 294 s (191–398) |
| vs the same code on the host | ~150× slower |
| host parity | 11/12 byte-for-byte identical |

The old device result below remains useful provenance, but neither old nor new
sample licenses copying the host's 961-row percentage onto the board.

### The one case that disagreed

Case 8 asks for a calendar event *and* an email. The board answered `send_email`;
the host answered `create_calendar_event` with degenerate arguments. Both are
deterministic — the host reproduces its answer byte-for-byte on every re-run.

It is not a porting bug. Recompiling the **host** with `-O3 -ffast-math` and
changing nothing else flips it to the board's exact answer, `send_email` with
identical arguments. The two tool names are 0.57 nats apart in mean token
logprob, so the greedy argmax at the first name token is on a knife edge and any
change in floating-point association tips it. Xtensa and arm64 round differently;
that is enough.

So the honest form of the parity claim is: **the device reproduces the host
byte-for-byte except where the model itself is numerically undecided.** Cases like
that are not ones the engine gets reliably right on either side.

That 310 s is this benchmark's worst case: every record carries a different date in
its system turn, so the cached prefix (system + tools) never repeats and every call
pays a full ~480-token prefill. With a stable system prompt and a fixed tool set —
what a real deployment looks like — the KV prefix cache hits and the same board
answers in about 30 s (measured separately: 241 s cold, 29 s warm).

### Two device gotchas

- **Console.** `printf`/`stdin` only ever reach the *primary* console; a secondary
  console mirrors `ESP_LOG` but not stdio. Native-USB boards need
  `CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG=y`, USB-UART-bridge boards need
  `CONFIG_ESP_CONSOLE_UART_DEFAULT=y`. `esptool flash_id` prints
  `USB mode: USB-Serial/JTAG` on the former only — that is how to tell them apart.
- **Build skew.** The firmware is flashed once and a 25-case run takes two hours. If
  `needle.c` is edited and the host binary rebuilt during that window, the two sides
  are different builds and every disagreement is spurious. This produced a 22/25
  parity result that read as a portability bug and was not one — the giveaway was
  that the *device* output was cleaner than the host's on two of the three
  mismatches, which is impossible for the same code. `device_runner.py` now copies
  the host binary aside at startup and warns if `needle-esp32s3/build` is older than
  `needle.c`.
- **Serial open.** On a bridge board, pyserial asserting DTR/RTS at open drives EN
  and IO0 and boots the chip into download mode, which looks exactly like a dead
  firmware. `device_runner.py` deasserts both before opening.

## Results in `results/`

| File | What |
|---|---|
| `ours_961.json` | legacy fixed-order run, default config (`MAX_CALLS=4`, `CONT_MARGIN=2.0`) |
| `ours_961_protocol_v2.json` | pre-TIE728 dataset-order strict run, 474/961 |
| `ours_961_20260822.json` | current-source dataset-order strict run, 469/961 |
| `ours_961_sink160_reason256_grammar_20260822.json` | final ESP32-sized decoder configuration, 669/961 |
| `ours_961_final_20260822.json` | final source/binary rerun after ESP32 scratch fixes, same raw outputs, 669/961 |
| `ours_961_singlecall.json` | same build forced to `NEEDLE_MAX_CALLS=1`, the single-vs-multi comparison |
| `ours_961_margin0_multicall.json` | multi-call at `CONT_MARGIN=0` — the full-eval version of the sweep's top row |
| `oracle_961.json` | legacy official output; saved pass flags are stale, re-score raw fields |
| `oracle_961_protocol_v2.json` | package 2.0.6 / engine 2.0.2 on identical protocol-v2 inputs, 665/961 |
| `mobile_actions_protocol_v2_audit_20260819.json` | hashes, current/published/historical comparison, and device summary |
| `esp32s3_mobile_actions_stratified12_20260819.json` | fair-build real-board sample; 12 valid rows plus reproducibility sidecar |
| `esp32s3_mobile_actions_interrupted_timeout_20260819.json` | interrupted continuous run retaining the excluded timeout |
| `device_parity12.json` | ESP32-S3, firmware and host built from the same source — the parity number quoted above |
| `device_5.json`, `device_25.json` | earlier ESP32-S3 samples. `device_25` was run against a stale firmware build; its 22/25 parity is a measurement artefact, kept only because the trap is worth seeing |
| `speed_ours.json`, `speed_oracle.json` | per-call latency and throughput |
| `speed_ours_final_20260822.json`, `speed_oracle_20260822.json` | final-source 200-case M4 speed audit |
| `esp32s3_fast_tie_*_20260822.json` | fastest-default device timing and host-parity rows |
| `esp32s3_sink160_grammar_row8_20260822.json` | final firmware, complex two-call/date row, strict and host parity |

Each row holds the query, expected calls, produced calls, pass flags and wall time.
New runs also write `<out>.meta.json` with reproducibility hashes. Treat pass flags
as derived data and use `--rescore`; the old oracle file demonstrates why.

## Tuning knobs

Environment variables, all read at call time:

| Variable | Default | Effect |
|---|---|---|
| `NEEDLE_TOOLS_BUDGET` | 180 | token budget above which BM25 prunes the tool list; `0` disables pruning |
| `NEEDLE_MAX_CALLS` | 4 | cap on tool calls per turn; `1` restores single-call behaviour |
| `NEEDLE_PREFIX_SINK` | 160 | protected prefix tokens; `0` disables, `1` keeps the full prefix, `system` keeps only the system turn |
| `NEEDLE_REASON_MAX` | 256 | reasoning-token cap before `<tool_call>`; the old value 90 truncated 139 rows in one ablation |
| `NEEDLE_BYTE_GRAMMAR` | 1 | continuous byte grammar; `0` restores the legacy segmented decoder |
| `NEEDLE_CONT_MARGIN` | 2.0 | legacy segmented decoder only: comma-vs-close margin when `NEEDLE_BYTE_GRAMMAR=0` |
| `NEEDLE_NAME_SCORED` | unset | legacy segmented decoder only: score candidate names by mean token logprob |
| `NEEDLE_NO_VALMASK` | unset | legacy segmented decoder only: disable the initial string-value structure mask |
| `NEEDLE_NO_PREFIX_CACHE` | unset | force a cold prefill on every call |
| `NEEDLE_FREE` | unset | unconstrained decoding, for inspecting raw model output |
| `NEEDLE_DEBUG` | unset | legacy decoder: print name scores and branch decisions |
| `NEEDLE_DEBUG_VAL` | unset | legacy decoder: print top logits at the start of argument values |
| `NEEDLE_DEBUG_STOP` | unset | legacy decoder: print comma-vs-close logits |

## Two traps worth knowing about

**Check I/O alignment before you hunt for state bugs.** Accuracy appeared to decay
with position in a batch — 68% early, 22% in the tail. It looked exactly like state
contamination, and a canary query interleaved every 10 calls disproved that (220/220
identical). The actual cause: a decoded string value containing a raw newline split
one record into two, so 961 queries produced 962 output lines and every result after
the split was scored against the wrong case. `wc -l` would have found it in seconds.
The harness now raises on a line-count mismatch instead of letting `zip` truncate,
and the engine JSON-escapes control characters.

**A broken measurement is worse than no measurement.** The same misalignment made a
correct change look like a 43-point regression, and it was reverted on that
evidence. It was re-tested after the harness was fixed and turned out to be worth
+63 exact matches with zero regressions.
