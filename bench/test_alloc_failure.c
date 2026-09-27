#define _POSIX_C_SOURCE 200809L
#include <stdlib.h>
#include <assert.h>
#include <errno.h>
static int allocations, fail_at;
static int fail(void) { return ++allocations == fail_at; }
static void *test_malloc(size_t n) { return fail() ? NULL : malloc(n); }
static void *test_calloc(size_t n, size_t s) { return fail() ? NULL : calloc(n, s); }
static int test_align(void **p, size_t a, size_t n) {
    return fail() ? ENOMEM : posix_memalign(p, a, n);
}
#define malloc test_malloc
#define calloc test_calloc
#define posix_memalign test_align
#define NEEDLE_NO_MAIN
#include "../needle.c"
#undef malloc
#undef calloc
#undef posix_memalign

static void cleanup(Needle *m) {
    void *buffers[] = {m->k_cache, m->v_cache, m->k_scale, m->v_scale, m->attn_scores,
        m->ering, m->ering_valid, m->hist, m->px_ering, m->px_ering_valid,
        m->x, m->nx, m->u, m->h, m->q, m->k, m->v, m->att_out, m->xh, m->xh2,
        m->xq[0], m->xq[1], m->xs[0], m->xs[1], m->z, m->logits,
        m->ek, m->ev, m->e, m->cosv, m->sinv};
    for (unsigned i = 0; i < sizeof buffers / sizeof *buffers; i++) free(buffers[i]);
}
static void init(Needle *m) {
    memset(m, 0, sizeof *m);
    m->d_model = 8; m->head_dim = 4; m->n_layers = 1;
    m->n_kv = m->n_heads = m->n_sites = m->n_orders = m->taps = 1;
    m->lanes = 2; m->hada_n = m->vocab = 8; m->rope_theta = 10000;
    m->orders[0] = 1;
}
int main(void) {
    static Needle m;
    for (int warm = 0; warm < 2; warm++) {
        init(&m); fail_at = allocations = 0;
        if (warm) needle_reset(&m, 16);
        allocations = 0; needle_reset(&m, 16);
        int count = allocations;
        assert(m.max_len == 16); cleanup(&m);
        for (int fault = 1; fault <= count; fault++) {
            init(&m); fail_at = allocations = 0;
            if (warm) needle_reset(&m, 16);
            allocations = 0; fail_at = fault;
            needle_reset(&m, 16);
            assert(m.max_len == 0 && !m.px_valid);
            fail_at = 0;
            needle_reset(&m, 16);
            assert(m.max_len == 16 && m.cosv && m.sinv && m.nx);
            cleanup(&m);
        }
    }
    init(&m);
    int ids[8] = {99}; allocations = 0; fail_at = 1;
    assert(needle_encode(&m, "hello", ids, 8) == -1 && ids[0] == 99);
    allocations = 0;
    assert(encode_raw(&m, "hello", ids, 8) == -1);
    int32_t empty_slot = -1;
    m.p2id_cap = 1; m.p2id_slot = &empty_slot; m.byte_fallback = 1;
    setenv("NEEDLE_PREFIX_SINK", "system", 1);
    for (int budget = 0; budget <= 1; budget++) {
        setenv("NEEDLE_TOOLS_BUDGET", budget ? "1" : "0", 1);
        for (int fault = 1; fault <= 3; fault++) {
            allocations = 0; fail_at = fault;
            char out[8] = "keep";
            assert(needle_toolcall_sys(&m, "system", "hello", "[{\"name\":\"gpio\"}]", out, sizeof out) == -1);
            assert(strcmp(out, "keep") == 0);
        }
    }
    return 0;
}
