/* gcc -O2 bench/test_empty_tools.c -lm -o /tmp/test-empty && /tmp/test-empty */
#define NEEDLE_NO_MAIN
#include "../needle.c"
#include <assert.h>

int main(void) {
    const char *empty[] = {"[]", " \t[ \r\n ]\n"};
    for (unsigned i = 0; i < sizeof empty / sizeof *empty; i++) {
        char out[4] = "xxx";
        g_needle_stats.prefill_tok = g_needle_stats.decode_tok = 42;
        /* A null model proves the empty-list path never starts inference. */
        assert(needle_toolcall_sys(NULL, NULL, "turn on the light", empty[i], out, 3) == 2);
        assert(strcmp(out, "[]") == 0 && out[3] == '\0');
        assert(g_needle_stats.prefill_tok == 0 && g_needle_stats.decode_tok == 0);
        for (size_t size = 0; size < 3; size++) {
            memcpy(out, "xxx", 4);
            assert(needle_toolcall_sys(NULL, NULL, "test", empty[i], out, size) == -1);
            assert(strcmp(out, "xxx") == 0);
        }
    }
    /* Disable retrieval so malformed schemas can be tested without weights. */
    setenv("NEEDLE_TOOLS_BUDGET", "0", 1);
    const char *invalid[] = {"", "[", "[] garbage", "[{}]", "{}"};
    for (unsigned i = 0; i < sizeof invalid / sizeof *invalid; i++) {
        char out[3];
        assert(needle_toolcall_sys(NULL, NULL, "test", invalid[i], out, sizeof out) == -1);
    }
    return 0;
}
