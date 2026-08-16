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

static const char *DEMO_TOOLS =
    "[{\"name\":\"gpio_write\",\"description\":\"Set a GPIO pin high or low to control a device\",\"parameters\":{\"type\":\"object\",\"properties\":{\"pin\":{\"type\":\"integer\",\"description\":\"GPIO pin number\"},\"value\":{\"type\":\"integer\",\"description\":\"1 for on, 0 for off\"}},\"required\":[\"pin\",\"value\"]}},{\"name\":\"get_time\",\"description\":\"Get the current date and time\",\"parameters\":{\"type\":\"object\",\"properties\":{}}},{\"name\":\"set_timer\",\"description\":\"Set a countdown timer\",\"parameters\":{\"type\":\"object\",\"properties\":{\"minutes\":{\"type\":\"integer\",\"description\":\"Timer length in minutes\"}},\"required\":[\"minutes\"]}},{\"name\":\"get_weather\",\"description\":\"Get current weather for a city\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\",\"description\":\"City name\"}},\"required\":[\"city\"]}},{\"name\":\"web_search\",\"description\":\"Search the web for information\",\"parameters\":{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\",\"description\":\"Search query\"}},\"required\":[\"query\"]}},{\"name\":\"send_message\",\"description\":\"Send a text message to a contact\",\"parameters\":{\"type\":\"object\",\"properties\":{\"to\":{\"type\":\"string\",\"description\":\"Contact name\"},\"text\":{\"type\":\"string\",\"description\":\"Message body\"}},\"required\":[\"to\",\"text\"]}}]";

static void print_heap(const char *tag) {
    printf("[heap] %s: internal free %u KB, psram free %u KB\n", tag,
           (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
           (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
}

static void run_query(const char *query, int max_new) {
    char prompt[4096];
    snprintf(prompt, sizeof prompt, "%s<tools>%s</s><tool_call>", query, DEMO_TOOLS);

    static int ids[1024];
    ids[0] = (int)g_model.bos_id;
    int n_ids = 1 + needle_encode(&g_model, prompt, ids + 1, 1023);
    printf("[needle] %d prompt tokens\n", n_ids);

    (void)max_new; (void)n_ids;
    static char callbuf[2048];
    int64_t t0 = esp_timer_get_time();
    int cl = needle_toolcall(&g_model, query, DEMO_TOOLS, callbuf, sizeof callbuf);
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

    /* weights in PSRAM read ~3x faster than flash; cache as many as fit,
     * keeping 3MB for KV cache and runtime state */
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
        printf("[pie] matvec 512x512x2b: float %lld us, int16 %lld us (%.2fx)\n",
               (long long)((tb - ta) / 20), (long long)((tc - tb) / 20),
               (double)(tb - ta) / (double)(tc - tb));
    }

    run_query("What's the weather in Paris right now?", 32);   /* cold: full prefill */
    run_query("Set a timer for 10 minutes", 32);                /* warm: prefix reused */

    printf("\n[needle] REPL ready. Type a query, press enter.\n> ");
    fflush(stdout);
    static char line[512];
    for (;;) {
        if (fgets(line, sizeof line, stdin)) {
            size_t ln = strlen(line);
            while (ln && (line[ln - 1] == '\n' || line[ln - 1] == '\r')) line[--ln] = 0;
            if (ln == 0) { printf("> "); fflush(stdout); continue; }
            run_query(line, 48);
            printf("\n> ");
            fflush(stdout);
        } else {
            vTaskDelay(pdMS_TO_TICKS(50));
        }
    }
}
