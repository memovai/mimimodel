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

The 2026-08-19 protocol-v2 rerun gives:

```
=== google/mobile-actions eval · this engine · 961 cases ===
accuracy (ordered strict exact, all 961): 474/961 (49.3%)   name acc 760/961 (79.1%)
  1-call (640): exact 386 (60.3%)   names 613 (95.8%)
  2-call (320): exact  88 (27.5%)   names 147 (45.9%)
  3-call (  1): exact   0 ( 0.0%)   names   0 ( 0.0%)
latency: median 1814 ms, p90 2262 ms
```

### Where it stands

| | this engine | official engine 2.0.2, identical inputs |
|---|---|---|
| strict accuracy | 49.3% | 69.2% |
| name accuracy | 79.1% | 98.1% |
| 1-call strict | 60.3% | 73.6% |
| 2-call strict | 27.5% | 60.3% |

The historical custom output scores 469/961 (48.8%) strict or 484/961 (50.4%)
under `strip().lower()`. The former 76.9% official number was not a valid
re-score: it summed flags in `oracle_961.json`, where 63 rows disagree with their
own output. That raw output scores 679/961 (70.7%) strict. Protocol-v2 changed 339
custom outputs (40 wins, 35 losses) and 196 official outputs (55 wins, 69 losses),
which is why a single headline cannot be moved between protocols. Run
`python bench/mobile_actions.py --rescore <file>` to audit any artifact; it reads
the saved metric from the sidecar before checking flags.

The remaining gap is in this engine, not the model or the data. It splits cleanly:

- **1-call (640 cases): 60.3% against 73.6%.** Names are close (95.8% vs
  99.2%); the loss is in argument values.
- **2-call (320 cases): 27.5% against 60.3%.** This is where most of the total gap
  lives. Deciding *whether to open a second call* is the weak part — see
  [Multi-call](#multi-call).

Cactus publishes 63.7% exact / 98.3% names for Needle 2 on the same named split
and strict metric. The public package scores 69.2%/98.1% here. The site does not
publish its prompt/schema conversion, selected-tool traces, raw rows, or binary
hash, so the 5.5-point exact difference cannot be attributed. Treat the website
number as an external reference, not a third directly comparable column.

### Multi-call

321 of the 961 cases expect more than one tool call, so a single-call engine has a
hard ceiling of 66.6%. The multi-call path is **on by default**
(`NEEDLE_MAX_CALLS=4`). The following ablation uses the legacy workload and one knob:

| | `MAX_CALLS=1` | `MAX_CALLS=4` (default) |
|---|---|---|
| strict accuracy | 39.3% | **48.8%** |
| name accuracy | 63.6% | **78.9%** |
| 1-call strict (640) | 59.1% | 58.8% |
| 1-call names (640) | 95.5% | 94.7% |
| 2-call strict (320) | 0% | **29.1%** |

+9.5 points overall for 0.3 points of single-call accuracy. That trade only became
this favourable after the two bugs below were fixed; before them the same switch
was a net loss, which is why it shipped disabled at first.

After emitting a call, the engine compares the logit of `,` (open another call)
against `]` (stop). `NEEDLE_CONT_MARGIN` is how far ahead `,` must be before another
call is opened, so it trades 2-call recall against 1-call precision. Swept over 300
cases:

| margin | overall | 1-call exact | 1-call names | 2-call exact |
|---|---|---|---|---|
| 0 | 39.0% | 54.0% | 83.0% | 9.0% |
| 1.0 | 46.7% | 58.0% | 91.5% | 24.0% |
| **2.0** (default) | **49.3%** | 60.5% | 96.5% | **27.0%** |
| 3.0 | 42.0% | 61.0% | 97.0% | 4.0% |
| 4.0+ | 40.7% | 61.0% | 97.0% | 0% |

The curve is sharp on both sides. Below 2.0 the model opens calls it should not and
1-call name accuracy collapses; above 2.0 it almost never opens a second call and
the 2-call bucket goes to zero. There is no setting that is good at both.

#### Historical 2-call diagnosis: call *count*, not call *content*

Splitting the legacy run's 320 two-call cases by how many calls each engine emitted:

| calls emitted | this engine | official engine |
|---|---|---|
| 0 | 0 | 6 |
| 1 | 132 | 2 |
| **2 (correct)** | **155** | **311** |
| 3 | 30 | 1 |
| 4 | 3 | 0 |

Restricted to the turns where it emitted exactly two calls, this engine gets both
tool names right **98%** of the time and the whole answer strict **60%** — against
the official engine's 100% and 64%, and close to this engine's own 1-call rate.
Call count is the larger problem, but argument decoding remains a measurable gap.

The problem is that it only reaches two calls on 155 of 320 (48%) where the
official engine reaches it on 311 (97%). 132 turns stop one call short and 33
overshoot. So the entire 2-call deficit is one binary decision made 320 times from
a single logit comparison, and the official engine evidently does not make it that
way. Anything that predicts call count better — a stop classifier, a second pass
that re-reads the query after the first call — is worth far more here than any
improvement to argument decoding.

Two bugs had to be fixed before any of this measured correctly, and both looked
like model quality problems:

**The call opening was forced as two segments** (`[` then `{"name":"`) instead of
one. `dc_force` masks logits per segment, so splitting it changed which tokens the
model walks through and cost 1-call exact 61.1% → 50.8%. Forcing `[{"name":"` as a
single segment fixed it.

**The next call's opening was forced before checking there was a candidate left.**
With BM25 pruning the tool list to 2–3 entries, a turn could emit every candidate
and still force `,{"name":"`, then break out of the loop with nothing to fill it —
producing `...},{"name":"]`, which fails to parse and scores zero. This hit 172 of
the 320 two-call cases. The candidate check now runs *before* the forced opening.

The second bug also inverted the tuning conclusion: the margin sweep run before the
fix said larger margins hurt the 2-call bucket. After the fix, 2.0 is the clear peak.

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
The 7-tool block in this dataset is 417 tokens, so most of it sits outside the
window and selection collapses — measured 22% tool-name accuracy with the full
block against 77% with a 191-token one. The reference engine solves this with tool
RAG (`tool_rag_top_k = 2` is its *default*; see
`cactus-engine/src/utils.h:507`). This engine does the same with BM25, which gets
97.1% recall@3 on this set for no forward pass. It kicks in only when the block
exceeds `NEEDLE_TOOLS_BUDGET` (180 tokens), so small tool sets pass through
untouched and keep the KV prefix cache warm.

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

Protocol audit on an M4 Mac, first 100 dataset-order cases, serial:

| | this engine | official engine | ratio |
|---|---|---|---|
| completion latency, median | 1711 ms | 464 ms | 3.7× |
| request latency incl. init | 1711 ms | 667 ms | 2.6× |
| prefill | 244 tok/s | 1664 tok/s | 6.8× |
| decode | 195 tok/s | 996 tok/s | 5.1× |
| peak RSS | 20 MB | 163 MB | |

Official `Needle(...)` initialization was previously excluded from per-call
latency. It costs a 202 ms median with randomized tool order in this sample, versus
31 ms when order is fixed and its tool-index cache can be reused. The C engine's
parse and BM25 retrieval remain inside its completion measurement.

With common BM25 top-2 on the same 100 cases, median completion was 1509 ms versus
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
# parity build (default): IEEE-style math, no profiler
idf.py -DNEEDLE_FAST_MATH=OFF -DNEEDLE_PROFILE=OFF build
# performance audit build used below
idf.py -DNEEDLE_FAST_MATH=ON -DNEEDLE_PROFILE=ON build
idf.py -p /dev/tty.usbXXX flash
cd .. && .venv/bin/python bench/device_runner.py --port /dev/tty.usbXXX \
  --sample 12 --seed 20260819 --timeout 1200
```

`device_runner.py` sends each case over the serial REPL and compares the board's
answer against the host engine's on the same case. Both run the same `needle.c`
against the same weights, but Xtensa and arm64 can still take different greedy
branches. Running all 961 on the board would take about three days, so device
samples must report confidence intervals and both raw-output and score parity.

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
1.70 / 1.46 tok/s with fast math. Three fast-math mobile-actions rows all
matched the standard host output byte-for-byte; row 9 was also a strict-exact
dataset success and took 189.2 s. Three rows are a parity check, not an accuracy
estimate, so fast math remains an explicit opt-in.

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
| `ours_961_protocol_v2.json` | current dataset-order strict run, 474/961 |
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

Each row holds the query, expected calls, produced calls, pass flags and wall time.
New runs also write `<out>.meta.json` with reproducibility hashes. Treat pass flags
as derived data and use `--rescore`; the old oracle file demonstrates why.

## Tuning knobs

Environment variables, all read at call time:

| Variable | Default | Effect |
|---|---|---|
| `NEEDLE_TOOLS_BUDGET` | 180 | token budget above which BM25 prunes the tool list; `0` disables pruning |
| `NEEDLE_MAX_CALLS` | 4 | cap on tool calls per turn; `1` restores single-call behaviour |
| `NEEDLE_CONT_MARGIN` | 2.0 | logit margin `,` must beat `]` by before another call is opened; swept above |
| `NEEDLE_NAME_SCORED` | unset | score candidate tool names by mean token logprob instead of the greedy prefix walk (measured worse: 58% vs 64%) |
| `NEEDLE_NO_VALMASK` | unset | stop masking pure-structure tokens at the start of a string value (measured worse: −63 exact matches) |
| `NEEDLE_NO_PREFIX_CACHE` | unset | force a cold prefill on every call |
| `NEEDLE_FREE` | unset | unconstrained decoding, for inspecting raw model output |
| `NEEDLE_DEBUG` | unset | print tool-name candidate scores and branch decisions to stderr |
| `NEEDLE_DEBUG_VAL` | unset | print the top logits at the start of each argument value |
| `NEEDLE_DEBUG_STOP` | unset | print the `,` vs `]` logits at each stop-or-continue decision |

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
