# Running Needle as a local-first backend inside mimiclaw

> **Status: design document. Not implemented.** Everything below that is labelled *measured* was
> measured on hardware; everything else is a proposal. Numbers come from the engine in this
> repository and from [`memovai/mimiclaw`](https://github.com/memovai/mimiclaw) at the time of
> writing.

[mimiclaw](https://github.com/memovai/mimiclaw) is an ESP32-S3 agent firmware that calls the Claude
API for every message: it needs network, it costs API credits, and user data leaves the device.
This document works out whether the 45M local model from this repository can be dropped in as a
*second* LLM backend, tried first, with the cloud as fallback — and what it actually costs.

The goal is narrow: keep high-frequency device commands ("turn the light on", "read pin 7", "what
time is it") on the device, and let everything else go to the cloud unchanged.

## Three things to know before reading further

1. **Local is slower than cloud, not faster.** A local call takes **~29 s** (241 s on the first
   call after boot); the cloud answers in 3–8 s. Local-first buys offline operation, zero API
   cost, and data locality — at 4–6× the latency on device commands. If the goal is latency, this
   integration is net negative.
2. **Chinese does not work.** *Measured:* 0/5 on Chinese device commands (「打开5号引脚」 selects
   `gpio_read`, 「现在几点」 selects `gpio_on`). The official closed-source engine fails the same
   way, so this is the model's boundary, not the port's. The saving grace: confidence on these
   sits at 0.02–0.22, so a threshold routes them to the cloud correctly.
3. **Without a gate, it will take wrong actions.** The model **always** emits a tool call, even
   for "tell me a joke". With no confidence gate the local path hijacks every message and drives
   GPIO from it.

## Feasibility: yes, but the flash has to be repartitioned

| Resource | mimiclaw today | Needle needs | Verdict |
|---|---|---|---|
| **Flash, 16 MB** | app 1.31 MB (62.5% of its 2 MB slot); SPIFFS partition is 11.8 MB but holds **11.7 KB of actual content** | 13.74 MB of weights | ⚠️ Repartition: shrink SPIFFS, drop the OTA A/B pair |
| **PSRAM, 8 MB** | 200–400 KB steady state | 5.0–5.5 MB | ✅ Comfortable, 2 MB+ left over |
| **Internal SRAM, ~334 KB** | 180–250 KB free; the agent task already has a 24→20→16→14→12 KB stack fallback ladder | 154 KB static + 42 KB hot | ⚠️ **The real bottleneck.** Needle's static footprint must come down to ~60 KB |
| **Watchdog** | `TWDT=5s`, but `PANIC` is off and no task subscribes; the cloud path already tolerates 120 s | 29 s of compute | ✅ Warnings at worst, no reboot |

## Three measurements that shaped the design

### 1. The confidence head is what makes local-first safe — and this engine doesn't implement it yet

The official engine emits a confidence score, and *measured*, it discriminates well. Same command
set, split tool table, threshold 0.70:

```
Turn off pin 2       conf=0.974  gpio_off           ✓ local
Read pin 9           conf=0.995  gpio_read          ✓ local
What time is it?     conf=0.989  get_current_time   ✓ local
打开5号引脚            conf=0.216  gpio_read          ✗ → cloud (correctly caught)
现在几点              conf=0.029  gpio_on            ✗ → cloud (correctly caught)
Tell me a joke       conf=0.526  (declined)         → cloud (correct)
```

Over 15 cases: **9 served locally, 7 of those correct, every Chinese case correctly deflected.**

The `ConfidenceHead` (8 probes, attention-pooled, then `Dense(1)`) weights **are already in the
`.cact` file** (`heads.manifest`, `confidence_head.*`). This engine currently skips the probe
heads — see the `optional probe heads (fp16, skipped)` comment in `needle.c`. Implementing it is
the first priority: roughly 100 lines, and the pooling can be computed streaming with an online
softmax, so the memory and compute cost is negligible.

### 2. The local tool table must be reshaped around the model's strengths, not copied from the host

*Measured:* the model picks tool names and extracts integers reliably, but gets boolean/semantic
arguments wrong. Feeding mimiclaw's `gpio_write(pin, state)` verbatim:

```
Turn on pin 5   → gpio_write{pin:5,  state:0}   ✗ turns it off
Set GPIO 12 low → gpio_write{pin:12, state:1}   ✗ turns it on
write accuracy 1/5
```

Splitting it, for the local path only, into two tools with no boolean argument, and mapping back
in the adapter:

```
Turn on pin 5                → gpio_on{pin:5}    ✓
Turn off pin 2               → gpio_off{pin:2}   ✓
Switch on the light on pin 8 → gpio_on{pin:8}    ✓
write accuracy 5/6
```

So the adapter is **not** a mechanical rename of `input_schema` → `parameters`. It is a
hand-written local tool table plus a reverse mapping.

### 3. The 256-token attention window is a hard constraint

*Measured:* all 14 of mimiclaw's tools serialize to **1113 tokens**, 4.3× the window, and the
model starts picking the wrong tool. The local path should expose the device-control trio only —
four entries after the `gpio_on`/`gpio_off` split, about 240 tokens, entirely inside the window.

For reference: 6 tools ≈ 622 tokens, 3 tools ≈ 235 tokens.

## Architecture

### The seam (there is exactly one)

`llm_chat_tools()` is declared at `main/llm/llm_proxy.h:58`, implemented at
`main/llm/llm_proxy.c:550`, and called from exactly one place in the whole project:
`main/agent/agent_loop.c:238`, inside a ReAct loop capped at 10 iterations.

**Branch inside `llm_chat_tools`; leave `agent_loop.c` untouched.** The call site is unique, the
`llm_response_t` ownership contract (`llm_response_free`) is unchanged, and the ReAct retry logic
is unaffected. This also matches the existing two-backends-with-a-preference precedent in the
codebase: `tool_web_search.c` prefers Tavily and falls back to Brave.

### New files

| File | Responsibility |
|---|---|
| `components/needle/` | this repository's `needle.c` as an IDF component, plus `needle_confidence()` |
| `main/llm/llm_local.c` | local backend: mmap the weights, own the local tool table and reverse mapping, fill `llm_response_t` |
| `main/llm/llm_route.c` | the routing decision (pseudocode below) |

### Existing files to touch

- `main/llm/llm_proxy.c:550` — branch at the top of `llm_chat_tools`. Also move the
  "no API key configured" bail-out at `:557` into the cloud branch, or a fully offline device
  can't even reach the local path.
- `main/CMakeLists.txt` — two new sources, `needle` in `REQUIRES`
- `partitions.csv`, `sdkconfig.defaults.esp32s3` — see below
- `main/cli/serial_cli.c` — `set_local_mode off|auto|offline_only` and `local_status`, following
  the existing `set_model_provider` pattern at `:162`

## Partition table (16 MB, OTA A/B dropped)

```
# Name,   Type, SubType, Offset,    Size
nvs,      data, nvs,     0x9000,    0x6000     #   24 KB
phy_init, data, phy,     0xF000,    0x1000     #    4 KB
factory,  app,  factory, 0x10000,   0x1A0000   # 1.625 MB (app measures 1.31 MB, 384 KB spare)
needle,   data, 0x40,    0x1B0000,  0xD20000   # 13.44 MB of weights
spiffs,   data, spiffs,  0xED0000,  0x130000   # 1.19 MB (current content: 11.7 KB)
```

⚠️ **Tight:** the weights are 13,737,807 bytes and the partition is 13,762,560 — **24 KB of
headroom**. A larger `.cact` forces a re-layout.

Dropped: `otadata` + `ota_1` (remote OTA becomes USB-only), and `coredump` (already dead space —
`CONFIG_ESP_COREDUMP_ENABLE_TO_NONE=y`).

If 1.625 MB turns out to be too small for the app: mimiclaw currently builds at `-Og` with
assertions (`CONFIG_COMPILER_OPTIMIZATION_DEBUG=y`). Switching to `-Os` and trading
`CONFIG_MBEDTLS_CERTIFICATE_BUNDLE_DEFAULT_FULL` for `CMN` (~45 KB) frees 200 KB+.

## Memory plan

**PSRAM (8 MB).** Needle's KV ring 4.4–5.0 MB (`kv_alloc = min(max_len, kv_window + slack)`, with
`max_new` capped at 60–80 so `max_len ≈ 320`) plus 484 KB of model state ≈ **5.0–5.5 MB**.
mimiclaw uses 200–400 KB. The weight cache takes whatever is left (~2 MB); it is pure speed and
can be sacrificed.

**Internal SRAM — the bottleneck.** Needle's static footprint has to drop from 154 KB to ~60 KB.
The largest symbols, from `xtensa-esp32s3-elf-nm --size-sort`:

| Symbol | Now | Action |
|---|---|---|
| `starts` (BPE merge table) | 32.8 KB | → PSRAM heap, and size it to the actual tool JSON (~4 KB) |
| `pids` / `rids` (token buffers) | 20 KB | → PSRAM heap; a ≤400-token local prompt needs 2 KB |
| `prefix` / `rest` | 12 KB | → PSRAM heap |
| `tools[16] NTool` | 11 KB | → PSRAM heap, or shrink `NT_MAX_TOOLS` (the local path has 4) |
| `g_model` + LUTs | 17 KB | keep (hot) |
| `xn`, `gate`, `aw`, the 512-float scratches | ~42 KB | **keep in internal SRAM** — the weight stream evicts the shared data cache, so PSRAM-resident activations re-miss on every matvec |

⚠️ `NTool tools[16]` is ~11 KB and once overflowed a 48 KB task stack. Every large local in
`needle_toolcall` / `dc_*` is `static` or heap for this reason.

## Concurrency

- **Core collision.** The agent task is pinned to core 1 at priority 6 (`agent_loop.c:344`), and
  Needle's matvec worker is also pinned to core 1 at priority 5. Move the worker to **core 0** or
  the dual-core split stops working — *measured*, it is the single largest speedup (~1.8×).
- Keeping Telegram/WiFi responsive across 29 s of compute: the decode loop already yields with
  `vTaskDelay(1)`. Keep it.
- Reuse the existing "🐱mimi is working..." status message (`agent_loop.c:220-235`) on the local
  path too.

## Routing

```c
// llm_route.c — called at the top of llm_chat_tools
bool should_try_local(messages, tools_json) {
    if (local_mode == OFF) return false;
    if (local_mode == OFFLINE_ONLY && network_is_up()) return false;
    if (!needle_is_loaded()) return false;
    if (message_count(messages) > 1) return false;   // later ReAct turns carry tool
                                                     // results and history; Needle takes
                                                     // a bare query and nothing else
    if (!text_has_digit_or_time_keyword(last_user_text)) return false;   // see below
    return true;
}

esp_err_t llm_chat_tools(sys, messages, tools_json, resp) {
    if (should_try_local(messages, tools_json)) {
        local_result_t lr = llm_local_call(last_user_text(messages));
        if (lr.made_a_call && lr.confidence >= LOCAL_CONF_THRESHOLD /* 0.70 */
            && tool_is_in_local_allowlist(lr.name)) {
            fill_llm_response(resp, &lr);   // includes gpio_on/off → gpio_write mapping
            return ESP_OK;
        }
        // otherwise fall through to the cloud, and log it so hit rate is measurable
    }
    return llm_cloud_call(sys, messages, tools_json, resp);   // the existing implementation
}
```

**Why a text pre-gate is needed on top of confidence.** One failure mode slips past the threshold:
*measured*, "How are you?" hits `get_current_time` at **conf=0.996** — high confidence, completely
wrong. The local tool table contains only device tools, so chit-chat gets force-mapped onto one.
Requiring a digit (for the GPIO tools) or a time keyword cleanly rejects chit-chat without
touching real device commands.

**Default threshold: 0.70.** *Measured*, at 0.70 nine of fifteen cases go local and seven of those
are right. Raising it to 0.90 is safer but drops correct cases like "Turn on pin 5" (0.869).

## Local tool table and reverse mapping

Hand-written — it deliberately does not mirror `tool_registry.c`, because the point is to reshape
it. Guard against drift with a **boot-time assertion**: every `maps_to` name in the local table
must be findable in `tool_registry_get_tools_json()`, or the local path refuses to enable and logs
why.

| Local tool (fed to Needle) | Mapped back to |
|---|---|
| `gpio_on(pin)` | `gpio_write{pin, state:1}` |
| `gpio_off(pin)` | `gpio_write{pin, state:0}` |
| `gpio_read(pin)` | `gpio_read{pin}` |
| `get_current_time()` | `get_current_time{}` |

`llm_tool_call_t.name` is 32 bytes, `id` is 64, and a turn holds at most 4 calls
(`mimi_config.h:79`). The local path produces exactly one call, so all of these fit. Fill `id`
with something like `"local_0"` (the cloud path uses `toolu_xxx`).

## Lifecycle

- **Load at boot.** `needle_load()` only mmaps and parses the tensor directory — *measured* 48 ms
  — so it belongs in `app_main`, near `main/mimi.c:134`, after `llm_proxy_init()`.
- **Keep the KV ring allocated.** Allocate 4.4–5.0 MB once at startup, never per call. Freeing it
  destroys the `<tools>` prefix cache, and that cache is the difference between a 29 s call and a
  241 s one.
- The local tool table is constant, so after the first call every call is warm.

## Degradation

| Failure | Behaviour |
|---|---|
| `needle` partition missing or bad magic | `needle_is_loaded() = false`, everything goes to cloud, one warning at boot |
| PSRAM allocation fails | same; do not retry (it fragments) |
| Local returns `-1`, or the JSON won't parse | fall through to cloud silently |
| Low confidence, or tool not in the allowlist | fall through to cloud silently — **this is the normal path, not an error** |
| No network *and* local is unsure | cloud call fails; `agent_loop.c:240` already handles it ("Sorry, I encountered an error."). Worth changing to say the device is offline and the local model couldn't handle it |

## Phases

**Phase 1 — prove the loop end to end.** In this repository's standalone project, not mimiclaw:
add the local tool table and reverse mapping, verify over the serial REPL.
*Done when:* typing "turn on pin 5" on the board prints `gpio_write{"pin":5,"state":1}`.

**Phase 2 — implement the confidence head.** Read `heads.manifest` plus the probe/proj/bias
tensors from the `.cact`, pool streaming.
*Done when:* on the 15 commands in this document, the C engine's confidence is within 0.05 of the
official Python engine's. **Do not proceed past this phase without it** — the threshold is the
entire safety story.

**Phase 3 — slim down and componentize.** Move to `components/needle/`, static footprint to
~60 KB, matvec worker to core 0.
*Done when:* the standalone project still passes and `heap_caps_get_free_size(MALLOC_CAP_INTERNAL)`
shows 90 KB+ more free than today.

**Phase 4 — wire into mimiclaw.** Partition table, `llm_local.c` / `llm_route.c`, the branch in
`llm_chat_tools`, the CLI switch.
*Done when:* Telegram "turn on pin 5" is served locally (log shows `[route] local, conf=0.87`) and
actually drives the pin; 「打开5号引脚」 goes to cloud; "tell me a joke" goes to cloud.

**Phase 5 — instrument and tune.** Log hit rate and the confidence distribution at the routing
point; retune the threshold against a week of real traffic.

## Verification

- **Unit:** the Phase 2 confidence cross-check, reusing the structure of `bench/run_bench.py` with
  the `cactus-needle` package as oracle
- **Quality:** turn the 15 commands here into `bench/cases_mimiclaw.json` and track three numbers
  per change — local hits, correct-among-hits, Chinese-correctly-deflected
- **End to end:** real Telegram messages, comparing the serial log's routing decision against the
  actual GPIO level
- **Memory:** run mimiclaw's existing `heap_info` CLI command (`serial_cli.c:242`) at the end of
  every phase

## Risks that are easy to underestimate

1. **Internal SRAM is the binding constraint, not PSRAM.** mimiclaw's agent task already carries a
   24→12 KB stack fallback ladder, which is direct evidence the author hit the internal-SRAM
   ceiling. Another 100 KB from Needle can push that task down to 12 KB, and a single stack frame
   in `context_builder.c:93-105` wants 10 KB. If the Phase 3 slim-down falls short, the
   integration fails as *intermittent stack overflows*, which are miserable to debug.
2. **24 KB of headroom in the weights partition.** A new model revision forces a re-layout.
3. **Dropping OTA is an irreversible product decision** — devices already in the field lose remote
   firmware updates.
4. **Device commands go from ~5 s to ~29 s.** If responsiveness matters more to users than
   independence, the net value of this integration is negative. Ship the Phase 4 CLI switch
   defaulting to `offline_only` and let the user opt into `auto`.
