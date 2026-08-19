/* Needle 2 (45M, CQ2) on ESP32-S3 — weights memory-mapped from flash.
 *
 * Boot: maps the `needle` partition, loads the model, runs a demo tool-call,
 * then serves a REPL on the USB-serial console: each line = a user query
 * against the built-in demo tool set.
 */
#define NEEDLE_NO_MAIN
#include "needle_core.c"

#include "esp_partition.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static Needle g_model;

/* google/mobile-actions eval tool set (CC-BY-4.0) — 7 tools, 417 tokens,
 * pruned per query by the engine's BM25 retrieval to fit the window */
static const char *DEMO_TOOLS =
    "[{\"name\":\"open_wifi_settings\",\"description\":\"Opens the Wi-Fi settings.\",\"parameters\":{\"type\":\"object\",\"properties\":{}}},{\"name\":\"create_contact\",\"description\":\"Creates a contact in the phone's contact list.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"email\":{\"type\":\"string\",\"description\":\"The email address of the contact.\"},\"last_name\":{\"type\":\"string\",\"description\":\"The last name of the contact.\"},\"first_name\":{\"type\":\"string\",\"description\":\"The first name of the contact.\"},\"phone_number\":{\"type\":\"string\",\"description\":\"The phone number of the contact.\"}},\"required\":[\"first_name\",\"last_name\"]}},{\"name\":\"show_map\",\"description\":\"Shows a location on the map.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\",\"description\":\"The location to search for. May be the name of a place, a business, or an address.\"}},\"required\":[\"query\"]}},{\"name\":\"create_calendar_event\",\"description\":\"Creates a new calendar event.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"title\":{\"type\":\"string\",\"description\":\"The title of the event.\"},\"datetime\":{\"type\":\"string\",\"description\":\"The date and time of the event in the format YYYY-MM-DDTHH:MM:SS.\"}},\"required\":[\"title\",\"datetime\"]}},{\"name\":\"send_email\",\"description\":\"Sends an email.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"subject\":{\"type\":\"string\",\"description\":\"The subject of the email.\"},\"body\":{\"type\":\"string\",\"description\":\"The body of the email.\"},\"to\":{\"type\":\"string\",\"description\":\"The email address of the recipient.\"}},\"required\":[\"to\",\"subject\"]}},{\"name\":\"turn_off_flashlight\",\"description\":\"Turns the flashlight off.\",\"parameters\":{\"type\":\"object\",\"properties\":{}}},{\"name\":\"turn_on_flashlight\",\"description\":\"Turns the flashlight on.\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}]";

static void print_heap(const char *tag) {
    printf("[heap] %s: internal free %u KB, psram free %u KB\n", tag,
           (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
           (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
}

static void run_query(const char *query, int max_new) {
    (void)max_new;
    /* Mirror the host batch protocol: optional system and per-row tools fields. */
    static char qbuf[4096];
    snprintf(qbuf, sizeof qbuf, "%s", query);
    char *sys_part = NULL, *q_part = qbuf;
    const char *tools = DEMO_TOOLS;
    char *tabp = strchr(qbuf, '\t');
    if (tabp) {
        *tabp = 0; sys_part = qbuf; q_part = tabp + 1;
        char *tab2 = strchr(q_part, '\t');
        if (tab2) { *tab2 = 0; tools = tab2 + 1; }
    }
    for (char *p = sys_part; p && *p; p++) if ((uint8_t)*p == 0x1e) *p = '\n';
    for (char *p = q_part; *p; p++) if ((uint8_t)*p == 0x1e) *p = '\n';
    static char callbuf[2048];
    int64_t t0 = esp_timer_get_time();
    int cl = needle_toolcall_sys(&g_model, sys_part, q_part, tools,
                                 callbuf, sizeof callbuf);
    int64_t t1 = esp_timer_get_time();
    if (cl >= 0)
        printf("[needle] call: %s\n", callbuf);
    else
        printf("[needle] constrained decode failed\n");
    printf("[needle] total %lld ms | prefill %d tok %.2f tok/s | decode %d tok %.2f tok/s\n",
           (t1 - t0) / 1000,
           g_needle_stats.prefill_tok,
           g_needle_stats.prefill_tok * 1e6 / (double)g_needle_stats.prefill_us,
           g_needle_stats.decode_tok,
           g_needle_stats.decode_tok * 1e6 / (double)g_needle_stats.decode_us);
    print_heap("after query");
    needle_prof_dump();
}

void app_main(void) {
    printf("\n=== Needle 2 on ESP32-S3 (mimimodel) ===\n");
#ifdef __FAST_MATH__
    const int fast_math = 1;
#else
    const int fast_math = 0;
#endif
#ifdef NEEDLE_PROF
    const int profile = 1;
#else
    const int profile = 0;
#endif
    printf("[needle] build: fast_math=%d profile=%d sinkhorn_iters=%d\n",
           fast_math, profile, NEEDLE_SINKHORN_ITERS);
    print_heap("boot");

    const esp_partition_t *part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, 0x40, "needle");
    if (!part) {
        printf("ERROR: 'needle' partition not found\n");
        return;
    }
    const void *blob = NULL;
    esp_partition_mmap_handle_t mh;
    esp_err_t err = esp_partition_mmap(part, 0, part->size,
                                       ESP_PARTITION_MMAP_DATA, &blob, &mh);
    if (err != ESP_OK) {
        printf("ERROR: mmap failed: %d\n", err);
        return;
    }
    printf("[needle] weights mapped at %p (%u KB partition)\n", blob,
           (unsigned)(part->size / 1024));

    int64_t t0 = esp_timer_get_time();
    if (needle_load(&g_model, (const uint8_t *)blob, part->size) != 0) {
        printf("ERROR: needle_load failed (is the weights partition flashed?)\n");
        return;
    }
    printf("[needle] loaded in %lld ms: %lu layers, d_model %lu, vocab %lu, kv_window %lu\n",
           (esp_timer_get_time() - t0) / 1000, (unsigned long)g_model.n_layers,
           (unsigned long)g_model.d_model, (unsigned long)g_model.vocab,
           (unsigned long)g_model.kv_window);
    print_heap("after load");

    /* Cache as many weights as fit without displacing KV/runtime state. */
    {
        g_needle_part = part;
        g_needle_mmap_base = (const uint8_t *)blob;
        int64_t tc = esp_timer_get_time();
        size_t pf = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
        /* ring KV needs L*KV*window*hd*2 (int8 k+v) + scales; keep it + 600KB slack */
        size_t kv_rows = g_model.kv_window + NEEDLE_KV_SLACK;
        size_t kv_need = (size_t)g_model.n_layers * g_model.n_kv * kv_rows
                         * g_model.head_dim * 2
                       + (size_t)g_model.n_layers * g_model.n_kv * kv_rows * 8
                       + 600 * 1024;
        size_t budget = pf > kv_need ? pf - kv_need : 0;
        size_t cached = needle_cache_psram(&g_model, budget);
        printf("[needle] weights cached in PSRAM: %u KB in %lld ms\n",
               (unsigned)(cached / 1024), (esp_timer_get_time() - tc) / 1000);
    }

    /* PIE self-test: the SIMD integer matvec must agree with the scalar float
     * one on a real weight matrix, or every downstream number is garbage. */
    {
        needle_reset(&g_model, 8);
        CQMat *W = &g_model.layers[0].q_proj;
        static float xin[512], y_f[512], y_i[512];
        for (int i = 0; i < 512; i++) xin[i] = sinf(i * 0.37f) * 0.8f;
        cq_prepare_x(&g_model, W, xin, g_model.xh);
        cq_matvec_2b(&g_model, W, g_model.xh, y_f);
        cq_matvec_i16(&g_model, W, g_model.xh, y_i);
        float maxabs = 0, maxerr = 0;
        for (int i = 0; i < 512; i++) {
            if (fabsf(y_f[i]) > maxabs) maxabs = fabsf(y_f[i]);
            float e = fabsf(y_f[i] - y_i[i]);
            if (e > maxerr) maxerr = e;
        }
        printf("[pie] float y[0..3] = %.4f %.4f %.4f %.4f\n",
               y_f[0], y_f[1], y_f[2], y_f[3]);
        printf("[pie] int16 y[0..3] = %.4f %.4f %.4f %.4f\n",
               y_i[0], y_i[1], y_i[2], y_i[3]);
        printf("[pie] max |y| %.4f, max abs err %.5f, rel %.2e -> %s\n",
               maxabs, maxerr, maxerr / (maxabs > 0 ? maxabs : 1),
               maxerr < maxabs * 1e-3f ? "PASS" : "FAIL");

        int64_t ta = esp_timer_get_time();
        for (int r = 0; r < 20; r++) cq_matvec_2b(&g_model, W, g_model.xh, y_f);
        int64_t tb = esp_timer_get_time();
        for (int r = 0; r < 20; r++) cq_matvec_i16(&g_model, W, g_model.xh, y_i);
        int64_t tc = esp_timer_get_time();
#ifdef NEEDLE_PROF
        for (int r = 0; r < 20; r++) cq_matvec_mt(&g_model, W, g_model.xh, y_f);
        int64_t td = esp_timer_get_time();
#endif
        printf("[pie] matvec 512x512x2b: float %lld us, int16 %lld us (%.2fx)\n",
               (long long)((tb - ta) / 20), (long long)((tc - tb) / 20),
               (double)(tb - ta) / (double)(tc - tb));
#ifdef NEEDLE_PROF
        printf("[matvec] single-core %lld us, dual-core %lld us (%.2fx)\n",
               (long long)((tb - ta) / 20), (long long)((td - tc) / 20),
               (double)(tb - ta) / (double)(td - tc));
#endif
    }

    (void)0; // run_query("What's the weather in Paris right now?", 32);   /* cold: full prefill */
    (void)0; // run_query("Set a timer for 10 minutes", 32);                /* warm: prefix reused */

    printf("\n[needle] REPL ready. Type a query, press enter.\n> ");
    fflush(stdout);
    /* Accumulate until a newline. fgets() on the IDF console returns a PARTIAL
     * line whenever the RX buffer momentarily drains, so a query that arrives
     * in several chunks would otherwise be executed as several queries. */
    static char line[4096];
    size_t ln = 0;
    for (;;) {
        int c = fgetc(stdin);
        if (c == EOF) { vTaskDelay(pdMS_TO_TICKS(10)); continue; }
        if (c == '\n' || c == '\r') {
            if (ln == 0) continue;
            line[ln] = 0;
            run_query(line, 48);   /* accepts system<TAB>query<TAB>tools */
            ln = 0;
            printf("\n> ");
            fflush(stdout);
            continue;
        }
        if (ln + 1 < sizeof line) line[ln++] = (char)c;
    }
}
