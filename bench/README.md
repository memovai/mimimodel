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

## Accuracy

```bash
# this engine, all 961 eval cases
.venv/bin/python bench/mobile_actions.py --out bench/results/mine.json

# the official engine on the identical prompts, as a reference point
.venv/bin/python bench/mobile_actions.py --oracle --out bench/results/oracle.json

# quick iteration
.venv/bin/python bench/mobile_actions.py --limit 300
```

Output:

```
=== google/mobile-actions eval · this engine · 961 cases ===
accuracy (ordered strict exact, all 961): 484/961 (50.4%)   name acc 758/961 (78.9%)
  1-call (640): exact 389 (60.8%)   names 606 (94.7%)
  2-call (320): exact  95 (29.7%)   names 152 (47.5%)
  3-call (  1): exact   0 ( 0.0%)   names   0 ( 0.0%)
latency: median 1946 ms, p90 2427 ms
```

### Where it stands

All three columns are the same model and the same 961 prompts; only the engine
and the scoring normalisation differ.

| | this engine | official engine, same prompts | published leaderboard |
|---|---|---|---|
| accuracy | 50.4% | 76.9% | 63.7% |
| name acc | 78.9% | 99.2% | 98.3% |
| 1-call | 60.8% | 75.6% | 71.3% |
| 2-call | 29.7% | 79.4% | 48.4% |
| median latency | 1946 ms | 603 ms | — |

Those latencies are the accuracy run's own timings, taken while both engines were
scored; [Speed](#speed) measures the same thing properly on a dedicated 200-call
run and gets 1748 ms / 530 ms.

The official engine measured through this harness lands within 0.9 points of its
published *name* accuracy, which is the evidence that the harness itself is sound.
Both measured columns sit above the published *accuracy* because of the
lowercase/strip normalisation described under [The metric](#the-metric); that
leniency applies to both engines equally.

The remaining gap is in this engine, not the model or the data. It splits cleanly:

- **1-call (640 cases): 60.8% against 75.6%.** Names are nearly matched (94.7% vs
  99.8%); the loss is in argument values.
- **2-call (320 cases): 29.7% against 79.4%.** This is where most of the total gap
  lives. Deciding *whether to open a second call* is the weak part — see
  [Multi-call](#multi-call).

### Multi-call

321 of the 961 cases expect more than one tool call, so a single-call engine has a
hard ceiling of 66.6%. The multi-call path is **on by default**
(`NEEDLE_MAX_CALLS=4`). Same build, same prompts, one knob:

| | `MAX_CALLS=1` | `MAX_CALLS=4` (default) |
|---|---|---|
| overall accuracy | 40.7% | **50.4%** |
| name accuracy | 63.6% | **78.9%** |
| 1-call exact (640) | 61.1% | 60.8% |
| 1-call names (640) | 95.5% | 94.7% |
| 2-call exact (320) | 0% | **29.7%** |

+9.7 points overall for 0.3 points of single-call accuracy. That trade only became
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

#### The remaining 2-call gap is call *count*, not call *content*

Splitting the 320 two-call cases by how many calls each engine actually emitted:

| calls emitted | this engine | official engine |
|---|---|---|
| 0 | 0 | 6 |
| 1 | 132 | 2 |
| **2 (correct)** | **155** | **311** |
| 3 | 30 | 1 |
| 4 | 3 | 0 |

Restricted to the turns where it emitted exactly two calls, this engine gets both
tool names right **98%** of the time and the whole answer exact **61%** — against
the official engine's 100% and 81%, and in line with this engine's own 1-call exact
rate of 60.8%. Content quality is not the problem on these turns.

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

`accuracy` is **ordered strict exact match** — the same definition the published
leaderboard uses: function names, call order and every argument must match. A turn
that expects two calls and gets one is a miss. `name acc` is the same comparison
ignoring argument values.

One deliberate deviation: argument values are compared after `strip().lower()`.
That is slightly more lenient than the published metric, and it applies equally to
both engines, so the head-to-head is fair while the absolute numbers sit a few
points above the leaderboard's.

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

Measured on an M-series Mac, 200 calls each, serial:

| | this engine | official engine | ratio |
|---|---|---|---|
| latency, median | 1748 ms | 530 ms | 3.3× slower |
| latency, p90 | 2225 ms | 734 ms | 3.0× slower |
| prefill | ~166 tok/s | 1504 tok/s | 9.1× slower |
| decode | ~59 tok/s | 869 tok/s | 15× slower |
| peak RSS | 20 MB | 165 MB | |

The per-token gap (9–15×) is much wider than the end-to-end gap (3.3×) because the
two engines do different amounts of non-model work per call: the official engine
runs an embedding forward pass for its tool retrieval, this one runs BM25.

**Not comparable:** this engine is scalar C99; the official build is NEON-optimised
ARM64. Most of the per-token gap is that, not a design difference.

## On the ESP32-S3

```bash
cd needle-esp32s3 && idf.py build && idf.py -p /dev/tty.usbXXX flash
cd .. && .venv/bin/python bench/device_runner.py --port /dev/tty.usbXXX --limit 25
```

`device_runner.py` sends each case over the serial REPL and compares the board's
answer against the host engine's on the same case. Both run the same `needle.c`
against the same weights, so agreement is what licenses reading the host's 961-case
accuracy as the device's accuracy. Running all 961 on the board would take about
three days.

Measured (ESP32-S3, 240 MHz, 8 MB PSRAM, 12 cases, firmware and host binary built
from the same source):

| | value |
|---|---|
| per call | median 294 s (191–398) |
| vs the same code on the host | ~150× slower |
| host parity | 11/12 byte-for-byte identical |

The device is scored by parity, not by its own accuracy number: 12 cases is far too
small a sample to read an accuracy percentage off. Parity is the claim that
transfers.

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
| `ours_961.json` | this engine, default config (`MAX_CALLS=4`, `CONT_MARGIN=2.0`), full eval |
| `ours_961_singlecall.json` | same build forced to `NEEDLE_MAX_CALLS=1`, the single-vs-multi comparison |
| `ours_961_margin0_multicall.json` | multi-call at `CONT_MARGIN=0` — the full-eval version of the sweep's top row |
| `oracle_961.json` | the official engine, same prompts, full eval |
| `device_parity12.json` | ESP32-S3, firmware and host built from the same source — the parity number quoted above |
| `device_5.json`, `device_25.json` | earlier ESP32-S3 samples. `device_25` was run against a stale firmware build; its 22/25 parity is a measurement artefact, kept only because the trap is worth seeing |
| `speed_ours.json`, `speed_oracle.json` | per-call latency and throughput |

Each row holds the query, the expected calls, what the engine produced, the pass
flags and the wall time, so a run can be re-scored offline without re-running it.
That is worth knowing: when a metric definition changes, re-score the saved rows
rather than spending another 30 minutes of inference.

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
