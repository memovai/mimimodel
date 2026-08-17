# Benchmarks

Two suites live here:

| Suite | What it measures | Runtime |
|---|---|---|
| `mobile_actions.py` | accuracy on [google/mobile-actions](https://huggingface.co/datasets/google/mobile-actions), 961 eval cases | ~30 min host, ~3 h device |
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
accuracy (ordered strict exact, all 961): 391/961 (40.7%)   name acc 611/961 (63.6%)
  1-call (640): exact 391 (61.1%)   names 611 (95.5%)
  2-call (320): exact   0 ( 0.0%)   names   0 ( 0.0%)
latency: median 1741 ms, p90 2245 ms
```

### Where it stands

| | this engine | official engine, same prompts | published |
|---|---|---|---|
| accuracy | 40.7% | 71.3% | 63.7% |
| name acc | 63.6% | 98.0% | 98.3% |
| 1-call | 61.1% | 75.6% | 71.3% |
| 2-call | 0% | 62.5% | 48.4% |

The official engine measured through this harness lands within 0.3 points of its
published name accuracy, which is the evidence that the harness itself is sound.
It scores above the published *accuracy* because of the lowercase/strip
normalisation noted above.

Every point of the remaining gap is in this engine, not the model or the data.
The two-call column is the whole story of the second half: 320 of 961 cases need
more than one call and the default configuration emits exactly one.

### Multi-call

`NEEDLE_MAX_CALLS` defaults to 1. The multi-call path exists and works — it
answers 7.2% of the two-call cases — but on the same build over the full eval it
comes out behind: 40.0% against 40.7%, with 1-call name accuracy dropping from
95.5% to 84.8%. It perturbs the first call more than the extra calls are worth.

Part of that was a real bug, now fixed: the call opening was being forced as two
segments (`[` then `{"name":"`) instead of one. `dc_force` masks logits per
segment, so splitting it changed which tokens the model walks through and cost
1-call exact 61.1% → 50.8%. Forcing `[{"name":"` as a single segment recovered
most of it, but not all — the residual regression is unexplained and open.

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
| latency, median | 1820 ms | 587 ms | 3.1× slower |
| latency, p90 | 2478 ms | 774 ms | 3.2× slower |
| prefill | ~154 tok/s | 1408 tok/s | 9.1× slower |
| decode | ~64 tok/s | 797 tok/s | 12× slower |
| peak RSS | 20 MB | 163 MB | |

The per-token gap (9–12×) is much wider than the end-to-end gap (3.1×) because the
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
answer against the host engine's on the same case. Because both run the same
`needle.c` against the same weights, **any disagreement is a portability bug**, and
byte-for-byte agreement is what licenses reading the host's 961-case accuracy as
the device's accuracy. Running 961 cases on the board would take about three hours.

Measured (ESP32-S3, 240 MHz, 8 MB PSRAM):

| | value |
|---|---|
| per call | median 342 s (285–384) |
| vs the same code on the host | ~188× slower |
| host parity | byte-for-byte identical |

That 342 s is this benchmark's worst case: every record carries a different date in
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
- **Serial open.** On a bridge board, pyserial asserting DTR/RTS at open drives EN
  and IO0 and boots the chip into download mode, which looks exactly like a dead
  firmware. `device_runner.py` deasserts both before opening.

## Results in `results/`

| File | What |
|---|---|
| `ours_961.json` | this engine, default config, full eval |
| `ours_961_multicall.json` | same build with `NEEDLE_MAX_CALLS=4`, for the comparison above |
| `oracle_961.json` | the official engine, same prompts, full eval |
| `device_5.json`, `device_25.json` | ESP32-S3 samples, each row flagged with host parity |
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
| `NEEDLE_CONT_MARGIN` | 0 | logit margin the model must clear to open another call |
| `NEEDLE_NAME_SCORED` | unset | score candidate tool names by mean token logprob instead of the greedy prefix walk (measured worse: 58% vs 64%) |
| `NEEDLE_NO_VALMASK` | unset | stop masking pure-structure tokens at the start of a string value (measured worse: −63 exact matches) |
| `NEEDLE_NO_PREFIX_CACHE` | unset | force a cold prefill on every call |
| `NEEDLE_FREE` | unset | unconstrained decoding, for inspecting raw model output |

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
