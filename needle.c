/* needle.c — single-file CPU inference engine for Cactus Needle 2 (.cact format).
 *
 * Portable C99: builds on macOS/Linux (cc -O2 needle.c -lm) and ESP32-S3
 * (ESP-IDF component; weights memory-mapped from a flash partition).
 *
 * Format/math reference: needle/needle/model/{export,decode,architecture}.py
 * Validated token-for-token against needle_np.py (which is validated against
 * the official JAX implementation).
 *
 * Weights stay in-place (mmap'd flash on ESP32): CQ-packed matrices are
 * dequantized on the fly inside the matvec using the Hadamard identity
 *   (unit·H)·x == unit·(H·x)
 * so each matvec does one fast Walsh-Hadamard transform of the activation
 * per 128-wide group, then plain codebook-weighted dot products.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ config */

#define NEEDLE_TAG 0x05E12A83u
#define DT_FP16 1
#define DT_FP32 2
#define DT_CQ   3
#define DT_RAW  4
#define TERNARY_RECORD_BITS 5

/* hot scratch buffers must live in internal SRAM on ESP32: the weight stream
 * evicts everything from the shared data cache, so PSRAM-resident activations
 * would re-miss on every matvec row */
#ifdef ESP_PLATFORM
#include "esp_heap_caps.h"
#define fast_malloc(sz) heap_caps_malloc((sz), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)
#define aligned_alloc16(sz) \
    heap_caps_aligned_alloc(16, (((sz) + 15) & ~(size_t)15), \
                            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)
#else
#define fast_malloc malloc
static inline void *aligned_alloc16(size_t sz) {
    void *p = NULL;
    if (posix_memalign(&p, 16, (sz + 15) & ~(size_t)15) != 0) return NULL;
    return p;
}
#endif

#define NEEDLE_KV_SLACK 160   /* rows beyond the window: one call's decode */
#define MAX_LAYERS   32
#define MAX_SITES    4
#define MAX_TABLES   8
#define MAX_LANES    8
#define MAX_TAPS     8

/* ------------------------------------------------------------ fp16 helpers */

static inline float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t man = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) { bits = sign; }
        else { /* subnormal */
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) { man <<= 1; exp--; }
            man &= 0x3FF;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* --------------------------------------------------------------- fast WHT */

/* In-place unnormalized fast Walsh-Hadamard transform, n = power of two.
 * Matches _walsh_matrix(n) (Sylvester order) up to the 1/sqrt(n) factor,
 * which callers apply. */
static void fwht(float *x, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            for (int j = i; j < i + len; j++) {
                float a = x[j], b = x[j + len];
                x[j] = a + b;
                x[j + len] = a - b;
            }
        }
    }
}

/* ------------------------------------------------------------- model defs */

typedef struct {          /* view of one CQ-packed matrix, data stays in blob */
    const uint8_t *packed;   /* row-major LSB-first bitstream, in_pad*bits/8 per row */
    const uint16_t *norms;   /* fp16 per-group L2 norms, (out, in_pad/group) */
    uint32_t out, in, in_pad, group, bits;
} CQMat;

typedef struct {
    const float *cb;       /* codebook for this bit width (points into header) */
    float cb_tern[3];
} CQCode;

typedef struct {
    float *norm_in, *q_norm, *k_norm, *post_norm, *pre_hada, *d1, *d2, *d3;
    float attn_gate;
    CQMat q_proj, k_proj, v_proj, gate_proj, out_proj;
} Layer;

typedef struct {
    CQMat tables, key_proj, value_proj;
    float *taps;           /* (n_taps, d_model) */
} EngramSite;

typedef struct {
    /* geometry */
    uint32_t vocab, d_model, n_heads, n_kv, n_layers, head_dim, max_seq,
             hada_n, lanes, slots, sub_dim, n_tables, taps, dilation,
             kv_window, kv_bits;
    uint32_t orders[4], n_orders, sites[4], n_sites;
    float rope_theta;

    const uint8_t *blob;   /* whole .cact in memory / mapped flash */
    float codebook[28];    /* cb2[4] | cb3[8] | cb4[16] */
    float lut2[256][4];    /* byte -> 4 decoded 2-bit codebook values */
    float lut4[256][2];    /* byte -> 2 decoded 4-bit codebook values */
    /* int16 mirrors of the same LUTs, for the SIMD/integer matvec path.
     * Weight value == lut*_i16[b][j] * cb*_scale  (exact: only 4/16 levels) */
    int16_t lut2_i16[256][4];
    int16_t lut4_i16[256][2];
    float cb2_scale, cb4_scale;

    CQMat embedding;
    Layer layers[MAX_LAYERS];
    /* mhc: fp32 copies of small tensors */
    float *a_pre, *a_post, *a_res, *b_pre, *b_post, *b_res; /* (L,lanes) / (L,n,n) */
    CQMat phi_pre, phi_post, phi_res;  /* (L*lanes, n*C) / (L*n*n, n*C) */
    EngramSite engrams[MAX_SITES];
    float *final_norm;

    /* tokenizer */
    uint32_t n_pieces, pad_id, eos_id, bos_id, unk_id;
    uint8_t add_dummy, byte_fallback;
    const char **pieces;   /* heap array of pointers into heap copies */
    uint16_t *piece_len;
    float *scores;
    uint8_t *types;
    int byte_id[256];
    int32_t *p2id_slot;            /* open-addressing hash: piece -> id */
    uint32_t p2id_cap;
    int marker_ids[24];            /* USER_DEFINED pieces, longest-first */
    int n_markers;

    /* run state */
    uint32_t max_len, kv_alloc;   /* kv cache is a ring of kv_alloc positions */
    int8_t  *k_cache, *v_cache;   /* (L, n_kv, max_len, head_dim) int8 */
    float   *k_scale, *v_scale;   /* (L, n_kv, max_len) */
    float   *ering;               /* engram v ring (n_sites, depth, C) */
    uint8_t *ering_valid;
    uint32_t ering_depth, epos;
    int     *hist;                /* token history for engram hashing */
    uint32_t hist_len;
    /* KV prefix cache: the <tools> block is identical across calls, so its
     * KV rows stay valid and only the query needs prefilling. */
    uint32_t px_hash, px_n, px_hist_len, px_epos, px_valid;
    float   *px_ering;
    uint8_t *px_ering_valid;

    /* scratch */
    float *x, *nx, *u, *bx, *h, *q, *k, *v, *att_out, *xh, *xh2, *z, *logits,
          *ek, *ev, *e;
    /* quantized companions of xh / xh2, filled by cq_prepare_x. Indexed by
     * which prep buffer a caller passed (see prep_slot()). */
    int16_t *xq[2];
    float   *xs[2];
    float *cosv, *sinv;           /* rope tables (max_len, head_dim/2) */
} Needle;

/* ------------------------------------------------------------ .cact parse */

static void mv_start_worker(void);   /* defined with the matvec code below */

static const uint8_t *g_dir;   /* tensor directory cursor during load */
static const uint8_t *g_base;

static void rec_next(uint8_t *dtype, uint8_t *ndim, uint32_t shape[4],
                     uint64_t *offset, uint64_t *nbytes, uint32_t *group,
                     uint32_t *bits) {
    const uint8_t *r = g_dir;
    *dtype = r[0]; *ndim = r[1];
    memcpy(shape, r + 4, 16);
    memcpy(offset, r + 20, 8);
    memcpy(nbytes, r + 28, 8);
    memcpy(group, r + 36, 4);
    memcpy(bits, r + 40, 4);
    g_dir += 44;
}

static CQMat rec_cq(void) {
    uint8_t dt, nd; uint32_t sh[4], group, bits; uint64_t off, nb;
    rec_next(&dt, &nd, sh, &off, &nb, &group, &bits);
    if (dt != DT_CQ) { fprintf(stderr, "expected CQ tensor, got %d\n", dt); exit(1); }
    CQMat m;
    m.out = sh[0]; m.in = sh[1];
    m.group = group; m.bits = bits;
    m.in_pad = (m.in + group - 1) / group * group;
    uint32_t row_bytes = (bits == TERNARY_RECORD_BITS) ? m.in_pad * 2 / 8
                                                       : m.in_pad * bits / 8;
    m.packed = g_base + off;
    m.norms = (const uint16_t *)(g_base + off + (uint64_t)m.out * row_bytes);
    return m;
}

/* fp16 tensor -> freshly malloc'd fp32 array (small tensors only) */
static float *rec_f32(uint64_t *count_out) {
    uint8_t dt, nd; uint32_t sh[4], group, bits; uint64_t off, nb;
    rec_next(&dt, &nd, sh, &off, &nb, &group, &bits);
    if (dt != DT_FP16 && dt != DT_FP32) { fprintf(stderr, "expected FP tensor, got %d\n", dt); exit(1); }
    uint64_t n = 1;
    for (int i = 0; i < nd; i++) n *= sh[i];
    if (nd == 0) n = nb / (dt == DT_FP16 ? 2 : 4);
    float *out = (float *)malloc(n * sizeof(float));
    if (dt == DT_FP16) {
        const uint16_t *src = (const uint16_t *)(g_base + off);
        for (uint64_t i = 0; i < n; i++) out[i] = fp16_to_f32(src[i]);
    } else {
        memcpy(out, g_base + off, n * 4);
    }
    if (count_out) *count_out = n;
    return out;
}

static void load_tokenizer(Needle *m, const uint8_t *blob, uint64_t nbytes) {
    (void)nbytes;
    memcpy(&m->n_pieces, blob, 4);
    memcpy(&m->pad_id, blob + 4, 4);
    memcpy(&m->eos_id, blob + 8, 4);
    memcpy(&m->bos_id, blob + 12, 4);
    memcpy(&m->unk_id, blob + 16, 4);
    m->add_dummy = blob[20];
    m->byte_fallback = blob[21];
    const uint8_t *p = blob + 24;
    uint32_t n = m->n_pieces;
    m->pieces = (const char **)malloc(n * sizeof(char *));
    m->piece_len = (uint16_t *)malloc(n * 2);
    m->scores = (float *)malloc(n * 4);
    m->types = (uint8_t *)malloc(n);
    for (int i = 0; i < 256; i++) m->byte_id[i] = -1;
    /* pass 1: total string bytes -> single arena (lands in PSRAM on ESP32) */
    const uint8_t *scan = p;
    size_t arena_sz = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint16_t ln; memcpy(&ln, scan + 5, 2);
        arena_sz += ln + 1;
        scan += 7 + ln;
    }
    char *arena = (char *)malloc(arena_sz);
    for (uint32_t i = 0; i < n; i++) {
        memcpy(&m->scores[i], p, 4);
        m->types[i] = p[4];
        uint16_t ln; memcpy(&ln, p + 5, 2);
        p += 7;
        memcpy(arena, p, ln); arena[ln] = 0;
        m->pieces[i] = arena;
        m->piece_len[i] = ln;
        arena += ln + 1;
        p += ln;
        if (m->types[i] == 4 && ln >= 5) {  /* TK_BYTE "<0xAB>" */
            unsigned b;
            if (sscanf(m->pieces[i], "<0x%02X>", &b) == 1) m->byte_id[b] = (int)i;
        }
    }
    /* piece -> id hash (linear probing), for O(1) BPE merge lookups */
    m->p2id_cap = 32768;
    m->p2id_slot = (int32_t *)malloc(m->p2id_cap * 4);
    for (uint32_t i = 0; i < m->p2id_cap; i++) m->p2id_slot[i] = -1;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t h = 2166136261u;
        for (int k = 0; k < m->piece_len[i]; k++)
            h = (h ^ (uint8_t)m->pieces[i][k]) * 16777619u;
        h &= m->p2id_cap - 1;
        while (m->p2id_slot[h] >= 0) h = (h + 1) & (m->p2id_cap - 1);
        m->p2id_slot[h] = (int32_t)i;
    }
    /* marker list sorted longest-first */
    m->n_markers = 0;
    for (uint32_t i = 0; i < n && m->n_markers < 24; i++)
        if (m->types[i] == 3) m->marker_ids[m->n_markers++] = (int)i;
    for (int a = 0; a < m->n_markers; a++)
        for (int b = a + 1; b < m->n_markers; b++)
            if (m->piece_len[m->marker_ids[b]] > m->piece_len[m->marker_ids[a]]) {
                int t = m->marker_ids[a];
                m->marker_ids[a] = m->marker_ids[b];
                m->marker_ids[b] = t;
            }
}

int needle_load(Needle *m, const uint8_t *blob, uint64_t size) {
    memset(m, 0, sizeof(*m));
    m->blob = blob;
    (void)size;
    uint32_t tag; memcpy(&tag, blob, 4);
    if (tag != NEEDLE_TAG) { fprintf(stderr, "bad magic\n"); return -1; }
    const uint32_t *h = (const uint32_t *)blob;
    uint32_t num_tensors = h[1], cb_n = h[2];
    m->kv_window = h[3]; m->kv_bits = h[4];
    m->vocab = h[5]; m->d_model = h[6]; m->n_heads = h[7]; m->n_kv = h[8];
    m->n_layers = h[9]; m->head_dim = h[10]; m->max_seq = h[11];
    m->hada_n = h[12]; m->lanes = h[13]; m->slots = h[14]; m->sub_dim = h[15];
    m->n_tables = h[16]; m->taps = h[17]; m->dilation = h[18];
    m->n_orders = h[19];
    for (int i = 0; i < 4; i++) m->orders[i] = h[20 + i];
    m->n_sites = h[24];
    for (int i = 0; i < 4; i++) m->sites[i] = h[25 + i];
    memcpy(&m->rope_theta, &h[29], 4);

    const uint8_t *p = blob + 30 * 4;
    memcpy(m->codebook, p, cb_n * 4);
    p += cb_n * 4;

    for (int b = 0; b < 256; b++) {
        for (int j = 0; j < 4; j++) m->lut2[b][j] = m->codebook[(b >> (2 * j)) & 3];
        m->lut4[b][0] = m->codebook[12 + (b & 15)];
        m->lut4[b][1] = m->codebook[12 + ((b >> 4) & 15)];
    }
    {
        float mx2 = 0, mx4 = 0;
        for (int i = 0; i < 4; i++)  if (fabsf(m->codebook[i]) > mx2) mx2 = fabsf(m->codebook[i]);
        for (int i = 12; i < 28; i++) if (fabsf(m->codebook[i]) > mx4) mx4 = fabsf(m->codebook[i]);
        m->cb2_scale = mx2 / 32767.0f;
        m->cb4_scale = mx4 / 32767.0f;
        for (int b = 0; b < 256; b++) {
            for (int j = 0; j < 4; j++)
                m->lut2_i16[b][j] = (int16_t)lrintf(m->lut2[b][j] / m->cb2_scale);
            for (int j = 0; j < 2; j++)
                m->lut4_i16[b][j] = (int16_t)lrintf(m->lut4[b][j] / m->cb4_scale);
        }
    }

    g_base = blob;
    g_dir = p;

    m->embedding = rec_cq();
    for (uint32_t i = 0; i < m->n_layers; i++) {
        Layer *L = &m->layers[i];
        L->norm_in = rec_f32(NULL);
        L->q_proj = rec_cq(); L->k_proj = rec_cq(); L->v_proj = rec_cq();
        L->q_norm = rec_f32(NULL); L->k_norm = rec_f32(NULL);
        L->gate_proj = rec_cq(); L->out_proj = rec_cq();
        L->post_norm = rec_f32(NULL);
        float *g1 = rec_f32(NULL); L->attn_gate = g1[0]; free(g1);
        L->pre_hada = rec_f32(NULL);
        L->d1 = rec_f32(NULL); L->d2 = rec_f32(NULL); L->d3 = rec_f32(NULL);
    }
    m->a_pre = rec_f32(NULL); m->a_post = rec_f32(NULL); m->a_res = rec_f32(NULL);
    m->b_pre = rec_f32(NULL); m->b_post = rec_f32(NULL); m->b_res = rec_f32(NULL);
    m->phi_pre = rec_cq(); m->phi_post = rec_cq(); m->phi_res = rec_cq();
    for (uint32_t s = 0; s < m->n_sites; s++) {
        EngramSite *E = &m->engrams[s];
        E->tables = rec_cq();
        E->key_proj = rec_cq();
        E->value_proj = rec_cq();
        E->taps = rec_f32(NULL);
    }
    m->final_norm = rec_f32(NULL);

    /* remaining tensors: optional probe heads (fp16, skipped) + RAW tokenizer */
    uint32_t consumed = 1 + m->n_layers * 14 + 9 + m->n_sites * 4 + 1;
    for (uint32_t i = consumed; i < num_tensors; i++) {
        uint8_t dt, nd; uint32_t sh[4], group, bits; uint64_t off, nb;
        rec_next(&dt, &nd, sh, &off, &nb, &group, &bits);
        if (dt == DT_RAW) load_tokenizer(m, blob + off, nb);
    }
    mv_start_worker();
    return 0;
}

/* ----------------------------------------------------------- CQ matvec ---
 * y[r] = sum_g norm[r,g] * sum_k cb[idx_{r,g,k}] * xh[g*group + k]
 * where xh = FWHT(x_group)/sqrt(group) per group.  x has length in (padded
 * internally).  Accumulates y = W @ x for W (out, in).                     */

/* Which quantized companion buffer belongs to this prepared-x pointer.
 * Only two prep buffers exist (xh, xh2) and prepare/matvec always run as a
 * pair on the same one, so the pointer identifies the slot. */
static inline int prep_slot(const Needle *m, const float *xh) {
    return xh == m->xh2 ? 1 : 0;
}

static void cq_prepare_x(const Needle *m, const CQMat *W, const float *x,
                         float *xh /* in_pad */) {
    uint32_t G = W->group;
    float inv = 1.0f / sqrtf((float)G);
    int slot = prep_slot(m, xh);
    int16_t *xq = m->xq[slot];
    float *xs = m->xs[slot];
    for (uint32_t g = 0; g < W->in_pad; g += G) {
        for (uint32_t k = 0; k < G; k++) {
            uint32_t src = g + k;
            xh[g + k] = (src < W->in) ? x[src] : 0.0f;
        }
        fwht(xh + g, G);
        float mx = 0.0f;
        for (uint32_t k = 0; k < G; k++) {
            xh[g + k] *= inv;
            float a = fabsf(xh[g + k]);
            if (a > mx) mx = a;
        }
        /* per-group symmetric int16 quantization; 15 bits is far more than
         * the 2/4-bit weights need, so this costs no measurable accuracy */
        if (mx < 1e-30f) {
            memset(xq + g, 0, G * sizeof(int16_t));
            xs[g / G] = 0.0f;
        } else {
            float q = 32767.0f / mx;
            xs[g / G] = mx / 32767.0f;
            for (uint32_t k = 0; k < G; k++)
                xq[g + k] = (int16_t)lrintf(xh[g + k] * q);
        }
    }
}

/* ---- integer matvec: int16 activations x int16-decoded weights ----------
 * y[r] = cb_scale * sum_g norm[r,g] * xs[g] * <wq[r,g,:], xq[g,:]>
 * The inner product is the SIMD hot spot; on ESP32-S3 it maps to the PIE
 * ee.vmulas.s16.accx.ld.ip 128-bit MAC (8 lanes/instruction). */

#if defined(ESP_PLATFORM) && defined(__XTENSA__) && defined(NEEDLE_PIE)
#define NEEDLE_HAVE_PIE 1
/* 40-bit accumulator dot product over n int16 pairs, n a multiple of 8.
 * Both pointers must be 16-byte aligned. */
static inline int64_t dot_i16(const int16_t *a, const int16_t *b, int n) {
    int32_t lo, hi;
    __asm__ volatile(
        "movi.n   %0, 0                 \n"
        "wur.accx_0 %0                  \n"
        "wur.accx_1 %0                  \n"
        "loopgtz  %4, 1f                \n"
        "ee.vld.128.ip q0, %2, 16       \n"
        "ee.vld.128.ip q1, %3, 16       \n"
        "ee.vmulas.s16.accx q0, q1      \n"
        "1:                             \n"
        "rur.accx_0 %0                  \n"
        "rur.accx_1 %1                  \n"
        : "=&a"(lo), "=&a"(hi), "+a"(a), "+a"(b)
        : "a"(n >> 3)
        : "memory");
    /* ACCX is 40-bit; accx_1 gives bits [39:32] zero-extended into an AR */
    int64_t v = ((int64_t)(hi & 0xFF) << 32) | (uint32_t)lo;
    if (v & 0x8000000000LL) v -= 0x10000000000LL;
    return v;
}
#else
static inline int64_t dot_i16(const int16_t *a, const int16_t *b, int n) {
    int64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = 0; i < n; i += 4) {
        s0 += (int32_t)a[i]     * b[i];
        s1 += (int32_t)a[i + 1] * b[i + 1];
        s2 += (int32_t)a[i + 2] * b[i + 2];
        s3 += (int32_t)a[i + 3] * b[i + 3];
    }
    return (s0 + s1) + (s2 + s3);
}
#endif

static void cq_matvec_i16(const Needle *m, const CQMat *W, const float *xh,
                          float *y) {
    uint32_t G = W->group, n_groups = W->in_pad / G;
    int slot = prep_slot(m, xh);
    const int16_t *xq = m->xq[slot];
    const float *xs = m->xs[slot];
    uint32_t row_bytes = (W->bits == 2) ? W->in_pad / 4 : W->in_pad / 2;
    float wscale = (W->bits == 2) ? m->cb2_scale : m->cb4_scale;
    static __attribute__((aligned(16))) int16_t wbuf[256];
    for (uint32_t r = 0; r < W->out; r++) {
        const uint8_t *row = W->packed + (uint64_t)r * row_bytes;
        const uint16_t *nrm = W->norms + (uint64_t)r * n_groups;
        float acc = 0.0f;
        for (uint32_t g = 0; g < n_groups; g++) {
            if (xs[g] == 0.0f) continue;
            /* decode this group's weights into int16 via the byte LUT */
            if (W->bits == 2) {
                const uint8_t *bp = row + g * G / 4;
                for (uint32_t k = 0; k < G; k += 4)
                    memcpy(wbuf + k, m->lut2_i16[bp[k >> 2]], 4 * sizeof(int16_t));
            } else {
                const uint8_t *bp = row + g * G / 2;
                for (uint32_t k = 0; k < G; k += 2)
                    memcpy(wbuf + k, m->lut4_i16[bp[k >> 1]], 2 * sizeof(int16_t));
            }
            int64_t d = dot_i16(wbuf, xq + g * G, (int)G);
            acc += (float)d * xs[g] * fp16_to_f32(nrm[g]);
        }
        y[r] = acc * wscale;
    }
}

/* 2-bit quad-row kernel: four rows share each activation load */
static void cq_matvec_2b(const Needle *m, const CQMat *W, const float *xh,
                         float *y) {
    uint32_t G = W->group, n_groups = W->in_pad / G;
    uint32_t row_bytes = W->in_pad / 4;
    const float (*lut)[4] = (const float (*)[4])m->lut2;
    uint32_t r = 0;
    for (; r + 4 <= W->out; r += 4) {
        const uint8_t *r0 = W->packed + (uint64_t)r * row_bytes;
        const uint8_t *r1 = r0 + row_bytes, *r2 = r1 + row_bytes, *r3 = r2 + row_bytes;
        const uint16_t *n0 = W->norms + (uint64_t)r * n_groups;
        const uint16_t *n1 = n0 + n_groups, *n2 = n1 + n_groups, *n3 = n2 + n_groups;
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (uint32_t g = 0; g < n_groups; g++) {
            const float *xg = xh + g * G;
            uint32_t boff = g * G / 4;
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (uint32_t k = 0; k < G; k += 4) {
                uint32_t bi = boff + (k >> 2);
                const float *e0 = lut[r0[bi]], *e1 = lut[r1[bi]];
                const float *e2 = lut[r2[bi]], *e3 = lut[r3[bi]];
                float x0 = xg[k], x1 = xg[k + 1], x2 = xg[k + 2], x3 = xg[k + 3];
                s0 += e0[0] * x0 + e0[1] * x1 + e0[2] * x2 + e0[3] * x3;
                s1 += e1[0] * x0 + e1[1] * x1 + e1[2] * x2 + e1[3] * x3;
                s2 += e2[0] * x0 + e2[1] * x1 + e2[2] * x2 + e2[3] * x3;
                s3 += e3[0] * x0 + e3[1] * x1 + e3[2] * x2 + e3[3] * x3;
            }
            a0 += fp16_to_f32(n0[g]) * s0;
            a1 += fp16_to_f32(n1[g]) * s1;
            a2 += fp16_to_f32(n2[g]) * s2;
            a3 += fp16_to_f32(n3[g]) * s3;
        }
        y[r] = a0; y[r + 1] = a1; y[r + 2] = a2; y[r + 3] = a3;
    }
    for (; r < W->out; r++) {   /* remainder rows, one at a time */
        const uint8_t *r0 = W->packed + (uint64_t)r * row_bytes;
        const uint16_t *n0 = W->norms + (uint64_t)r * n_groups;
        float a0 = 0;
        for (uint32_t g = 0; g < n_groups; g++) {
            const float *xg = xh + g * G;
            uint32_t boff = g * G / 4;
            float s0 = 0;
            for (uint32_t k = 0; k < G; k += 4) {
                const float *e0 = lut[r0[boff + (k >> 2)]];
                s0 += e0[0] * xg[k] + e0[1] * xg[k + 1]
                    + e0[2] * xg[k + 2] + e0[3] * xg[k + 3];
            }
            a0 += fp16_to_f32(n0[g]) * s0;
        }
        y[r] = a0;
    }
}

static void cq_matvec(const Needle *m, const CQMat *W, const float *xh,
                      float *y) {
    /* The integer path only wins where a SIMD MAC exists: on scalar cores the
     * weight-decode round trip costs more than it saves, and the host's
     * auto-vectorized float loops are faster still. */
#ifdef NEEDLE_HAVE_PIE
    if (W->bits == 2 || W->bits == 4) { cq_matvec_i16(m, W, xh, y); return; }
#endif
    if (W->bits == 2) { cq_matvec_2b(m, W, xh, y); return; }
    uint32_t G = W->group, n_groups = W->in_pad / G;
    const float *cb = (W->bits == 2) ? m->codebook
                    : (W->bits == 3) ? m->codebook + 4
                    : m->codebook + 12;
    uint32_t row_bytes = (W->bits == TERNARY_RECORD_BITS) ? W->in_pad / 4
                                                          : W->in_pad * W->bits / 8;
    for (uint32_t r = 0; r < W->out; r++) {
        const uint8_t *row = W->packed + (uint64_t)r * row_bytes;
        const uint16_t *nrm = W->norms + (uint64_t)r * n_groups;
        float acc = 0.0f;
        if (W->bits == 4) {
            const float (*lut)[2] = (const float (*)[2])m->lut4;
            for (uint32_t g = 0; g < n_groups; g++) {
                const float *xg = xh + g * G;
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                const uint8_t *bp = row + g * G / 2;
                for (uint32_t k = 0; k < G; k += 8) {
                    const uint8_t *bk = bp + (k >> 1);
                    const float *e0 = lut[bk[0]], *e1 = lut[bk[1]];
                    const float *e2 = lut[bk[2]], *e3 = lut[bk[3]];
                    const float *x0 = xg + k;
                    s0 += e0[0] * x0[0] + e0[1] * x0[1];
                    s1 += e1[0] * x0[2] + e1[1] * x0[3];
                    s2 += e2[0] * x0[4] + e2[1] * x0[5];
                    s3 += e3[0] * x0[6] + e3[1] * x0[7];
                }
                acc += fp16_to_f32(nrm[g]) * ((s0 + s1) + (s2 + s3));
            }
        } else if (W->bits == TERNARY_RECORD_BITS) {
            float c = 1.2240064f / sqrtf((float)G);
            for (uint32_t g = 0; g < n_groups; g++) {
                const float *xg = xh + g * G;
                float s = 0.0f;
                const uint8_t *bp = row + g * G / 4;
                for (uint32_t k = 0; k < G; k += 4) {
                    uint8_t b = bp[k >> 2];
                    static const float lut[4] = {0.0f, 1.0f, -1.0f, 0.0f};
                    /* crumb codes: 3,0,1 -> -1,0,+1 ; i.e. 0->0? see export:
                       crumbs = idx==0 ? 3 : idx-1 ; trit idx 0,1,2 -> -c,0,+c
                       so crumb 3 -> -c, 0 -> 0, 1 -> +c */
                    static const float lut2[4] = {0.0f, 1.0f, 0.0f, -1.0f};
                    (void)lut;
                    s += lut2[b & 3] * xg[k] + lut2[(b >> 2) & 3] * xg[k + 1]
                       + lut2[(b >> 4) & 3] * xg[k + 2] + lut2[(b >> 6) & 3] * xg[k + 3];
                }
                acc += fp16_to_f32(nrm[g]) * c * s;
            }
        } else { /* 3-bit: slow generic LSB-first bitstream */
            for (uint32_t g = 0; g < n_groups; g++) {
                const float *xg = xh + g * G;
                float s = 0.0f;
                for (uint32_t k = 0; k < G; k++) {
                    uint32_t bitpos = (g * G + k) * 3;
                    uint32_t byte = bitpos >> 3, sh = bitpos & 7;
                    uint32_t w = row[byte] | ((uint32_t)row[byte + 1] << 8);
                    s += cb[(w >> sh) & 7] * xg[k];
                }
                acc += fp16_to_f32(nrm[g]) * s;
            }
        }
        y[r] = acc;
    }
}

static CQMat cq_rows(const CQMat *W, uint32_t r0, uint32_t nrows) {
    CQMat s = *W;
    uint32_t row_bytes = (W->bits == TERNARY_RECORD_BITS) ? W->in_pad / 4
                                                          : W->in_pad * W->bits / 8;
    s.packed += (uint64_t)r0 * row_bytes;
    s.norms += (uint64_t)r0 * (W->in_pad / W->group);
    s.out = nrows;
    return s;
}

/* ---- dual-core matvec: worker on core 1 takes the upper half of the rows */
#ifdef ESP_PLATFORM
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

typedef struct { void (*fn)(void *); void *arg; } ParJob;
static ParJob g_parjob;
static SemaphoreHandle_t g_mv_go, g_mv_done;

static void mv_worker(void *arg) {
    (void)arg;
    for (;;) {
        xSemaphoreTake(g_mv_go, portMAX_DELAY);
        g_parjob.fn(g_parjob.arg);
        xSemaphoreGive(g_mv_done);
    }
}

static void mv_start_worker(void) {
    if (g_mv_go) return;
    g_mv_go = xSemaphoreCreateBinary();
    g_mv_done = xSemaphoreCreateBinary();
    xTaskCreatePinnedToCore(mv_worker, "needle_mv", 6144, NULL, 5, NULL, 1);
}

static int par_submit(void (*fn)(void *), void *arg) {
    if (!g_mv_go) return 0;
    g_parjob.fn = fn; g_parjob.arg = arg;
    xSemaphoreGive(g_mv_go);
    return 1;
}
static void par_wait(void) { xSemaphoreTake(g_mv_done, portMAX_DELAY); }

typedef struct { const Needle *m; CQMat W; const float *xh; float *y; uint32_t r0; } MvJob;
static void mv_job_fn(void *p) {
    MvJob *j = (MvJob *)p;
    CQMat sub = cq_rows(&j->W, j->r0, j->W.out - j->r0);
    cq_matvec(j->m, &sub, j->xh, j->y + j->r0);
}

static void cq_matvec_mt(const Needle *m, const CQMat *W, const float *xh, float *y) {
    static MvJob job;
    if (!g_mv_go || W->out < 64) { cq_matvec(m, W, xh, y); return; }
    uint32_t half = W->out / 2;
    job.m = m; job.W = *W; job.xh = xh; job.y = y; job.r0 = half;
    if (!par_submit(mv_job_fn, &job)) { cq_matvec(m, W, xh, y); return; }
    CQMat sub = cq_rows(W, 0, half);
    cq_matvec(m, &sub, xh, y);
    par_wait();
}
#else
static void mv_start_worker(void) {}
static int par_submit(void (*fn)(void *), void *arg) { (void)fn; (void)arg; return 0; }
static void par_wait(void) {}
static void cq_matvec_mt(const Needle *m, const CQMat *W, const float *xh, float *y) {
    cq_matvec(m, W, xh, y);
}
#endif

/* Copy a CQ matrix (packed indices + norms are contiguous in the blob) into
 * PSRAM, which the CPU reads ~2x faster than QIO flash. Returns bytes used. */
#ifdef ESP_PLATFORM
#include "esp_heap_caps.h"
#include "esp_partition.h"

/* set by the app so cached weights can be read via the SPI driver (fast DMA
 * path) instead of memcpy through the mmap cache */
const esp_partition_t *g_needle_part = NULL;
const uint8_t *g_needle_mmap_base = NULL;

static size_t cq_cache_psram(CQMat *W, size_t budget) {
    uint32_t row_bytes = (W->bits == TERNARY_RECORD_BITS) ? W->in_pad / 4
                                                          : W->in_pad * W->bits / 8;
    size_t total = (size_t)W->out * row_bytes
                 + (size_t)W->out * (W->in_pad / W->group) * 2;
    if (total > budget) return 0;
    uint8_t *buf = (uint8_t *)heap_caps_malloc(total, MALLOC_CAP_SPIRAM);
    if (!buf) return 0;
    if (g_needle_part && g_needle_mmap_base) {
        size_t off = (size_t)(W->packed - g_needle_mmap_base);
        if (esp_partition_read(g_needle_part, off, buf, total) != ESP_OK) {
            heap_caps_free(buf);
            return 0;
        }
    } else {
        memcpy(buf, W->packed, total);
    }
    W->packed = buf;
    W->norms = (const uint16_t *)(buf + (size_t)W->out * row_bytes);
    return total;
}

size_t needle_cache_psram(Needle *m, size_t budget) {
    size_t used = 0, got;
    CQMat *order[8 + MAX_LAYERS * 5];
    int n = 0;
    order[n++] = &m->phi_pre; order[n++] = &m->phi_post; order[n++] = &m->phi_res;
    for (uint32_t s = 0; s < m->n_sites; s++) {
        order[n++] = &m->engrams[s].key_proj;
        order[n++] = &m->engrams[s].value_proj;
    }
    for (uint32_t i = 0; i < m->n_layers; i++) {
        Layer *L = &m->layers[i];
        order[n++] = &L->q_proj; order[n++] = &L->k_proj; order[n++] = &L->v_proj;
        order[n++] = &L->gate_proj; order[n++] = &L->out_proj;
    }
    for (int i = 0; i < n; i++) {
        got = cq_cache_psram(order[i], budget - used);
        if (!got) break;
        used += got;
    }
    return used;
}
#endif

/* full W @ x for CQ matrix (allocount xh scratch outside) */
static void cq_apply(const Needle *m, const CQMat *W, const float *x,
                     float *y, float *xh) {
    cq_prepare_x(m, W, x, xh);
    cq_matvec_mt(m, W, xh, y);
}

/* dequantize a single CQ row into out[in] (used for embedding + engram rows) */
static void cq_row(const Needle *m, const CQMat *W, uint32_t r, float *out) {
    uint32_t G = W->group, n_groups = W->in_pad / G;
    const float *cb = (W->bits == 2) ? m->codebook
                    : (W->bits == 3) ? m->codebook + 4
                    : m->codebook + 12;
    uint32_t row_bytes = (W->bits == TERNARY_RECORD_BITS) ? W->in_pad / 4
                                                          : W->in_pad * W->bits / 8;
    const uint8_t *row = W->packed + (uint64_t)r * row_bytes;
    const uint16_t *nrm = W->norms + (uint64_t)r * n_groups;
    static float tmp[512];
    for (uint32_t g = 0; g < n_groups; g++) {
        float norm = fp16_to_f32(nrm[g]);
        if (W->bits == 4) {
            const uint8_t *bp = row + g * G / 2;
            for (uint32_t k = 0; k < G; k += 2) {
                uint8_t b = bp[k >> 1];
                tmp[k] = cb[b & 15] * norm;
                tmp[k + 1] = cb[(b >> 4) & 15] * norm;
            }
        } else if (W->bits == 2) {
            const uint8_t *bp = row + g * G / 4;
            for (uint32_t k = 0; k < G; k += 4) {
                uint8_t b = bp[k >> 2];
                tmp[k] = cb[b & 3] * norm;
                tmp[k + 1] = cb[(b >> 2) & 3] * norm;
                tmp[k + 2] = cb[(b >> 4) & 3] * norm;
                tmp[k + 3] = cb[(b >> 6) & 3] * norm;
            }
        } else {
            for (uint32_t k = 0; k < G; k++) tmp[k] = 0;
        }
        fwht(tmp, G);
        float inv = 1.0f / sqrtf((float)G);
        for (uint32_t k = 0; k < G; k++) {
            uint32_t dst = g * G + k;
            if (dst < W->in) out[dst] = tmp[k] * inv;
        }
    }
}

/* ------------------------------------------------------------- math bits */

static inline float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float siluf_(float x) { return x * sigmoidf_(x); }

static void zcrms(const float *x, const float *scale, float *out, int n) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + 1e-6f);
    for (int i = 0; i < n; i++) out[i] = (1.0f + scale[i]) * x[i] * inv;
}

static void rms_unit(const float *x, float *out, int n) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + 1e-6f);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv;
}

static void softmax_(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* sinkhorn over an n×n matrix of logits (n = lanes, tiny), log-space with
 * per-axis max subtraction (underflow-safe, matches the reference exactly) */
static void sinkhorn(float *lg, int n, int iters) {
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < n; i++) {           /* rows (axis -1) */
            float mx = lg[i * n];
            for (int j = 1; j < n; j++) if (lg[i * n + j] > mx) mx = lg[i * n + j];
            float s = 0;
            for (int j = 0; j < n; j++) s += expf(lg[i * n + j] - mx);
            float lse = logf(s) + mx;
            for (int j = 0; j < n; j++) lg[i * n + j] -= lse;
        }
        for (int j = 0; j < n; j++) {           /* cols (axis -2) */
            float mx = lg[j];
            for (int i = 1; i < n; i++) if (lg[i * n + j] > mx) mx = lg[i * n + j];
            float s = 0;
            for (int i = 0; i < n; i++) s += expf(lg[i * n + j] - mx);
            float lse = logf(s) + mx;
            for (int i = 0; i < n; i++) lg[i * n + j] -= lse;
        }
    }
    for (int i = 0; i < n * n; i++) lg[i] = expf(lg[i]);
}

/* ----------------------------------------------------------------- state */

void needle_reset(Needle *m, uint32_t max_len) {
    m->max_len = max_len;
    uint32_t C = m->d_model, hd = m->head_dim, L = m->n_layers, KV = m->n_kv;
    uint32_t n = m->lanes;
    /* A row is overwritten kv_alloc positions later, so kv_alloc > kv_window
     * keeps every in-window row intact — and leaves the prefix rows valid for
     * the next call as long as the slack covers one call's decode. */
    uint32_t want = m->kv_window ? m->kv_window + NEEDLE_KV_SLACK : max_len;
    m->kv_alloc = want < max_len ? want : max_len;
    free(m->k_cache); free(m->v_cache); free(m->k_scale); free(m->v_scale);
    free(m->ering); free(m->ering_valid); free(m->hist);
    m->k_cache = (int8_t *)calloc((size_t)L * KV * m->kv_alloc * hd, 1);
    m->v_cache = (int8_t *)calloc((size_t)L * KV * m->kv_alloc * hd, 1);
    m->k_scale = (float *)calloc((size_t)L * KV * m->kv_alloc, 4);
    m->v_scale = (float *)calloc((size_t)L * KV * m->kv_alloc, 4);
    if (!m->k_cache || !m->v_cache || !m->k_scale || !m->v_scale) {
        fprintf(stderr, "needle_reset: KV alloc failed (len %u)\n", (unsigned)m->kv_alloc);
        m->max_len = 0;
        return;
    }
    uint32_t maxo = 0;
    for (uint32_t i = 0; i < m->n_orders; i++) if (m->orders[i] > maxo) maxo = m->orders[i];
    m->ering_depth = (m->taps - 1) * maxo + 1;
    m->ering = (float *)calloc((size_t)m->n_sites * m->ering_depth * C, 4);
    m->ering_valid = (uint8_t *)calloc((size_t)m->n_sites * m->ering_depth, 1);
    free(m->px_ering); free(m->px_ering_valid);
    m->px_ering = (float *)calloc((size_t)m->n_sites * m->ering_depth * C, 4);
    m->px_ering_valid = (uint8_t *)calloc((size_t)m->n_sites * m->ering_depth, 1);
    m->px_valid = 0;
    m->epos = 0;
    m->hist = (int *)malloc(max_len * sizeof(int));
    m->hist_len = 0;

    if (!m->x) {
        uint32_t A = m->n_heads * hd;   /* attn width */
        m->x = (float *)fast_malloc((size_t)n * C * 4);
        m->nx = (float *)fast_malloc((size_t)n * C * 4);
        m->u = (float *)fast_malloc(C * 4);
        m->bx = (float *)fast_malloc(C * 4);
        m->h = (float *)fast_malloc(C * 4);
        m->q = (float *)fast_malloc(A * 4);
        m->k = (float *)fast_malloc((size_t)KV * hd * 4);
        m->v = (float *)fast_malloc((size_t)KV * hd * 4);
        m->att_out = (float *)fast_malloc(A * 4);
        m->xh = (float *)fast_malloc((size_t)n * C * 4);   /* >= biggest in_pad */
        m->xh2 = (float *)fast_malloc((size_t)n * C * 4);  /* phi-prepared nx */
        for (int s = 0; s < 2; s++) {
            /* ee.vld.128 needs 16-byte alignment; group offsets are multiples
             * of 256 bytes so aligning the base is enough */
            m->xq[s] = (int16_t *)aligned_alloc16((size_t)n * C * sizeof(int16_t));
            m->xs[s] = (float *)fast_malloc((size_t)n * C / 8 * sizeof(float));
        }
        m->z = (float *)fast_malloc(m->hada_n * 4);
        m->logits = (float *)malloc(m->vocab * 4);         /* cold: PSRAM ok */
        m->ek = (float *)fast_malloc((size_t)m->n_sites * C * 4);
        m->ev = (float *)fast_malloc((size_t)m->n_sites * C * 4);
        m->e = (float *)fast_malloc(C * 4);
    }
    free(m->cosv); free(m->sinv);
    uint32_t half = hd / 2;
    m->cosv = (float *)malloc((size_t)max_len * half * 4);
    m->sinv = (float *)malloc((size_t)max_len * half * 4);
    for (uint32_t t = 0; t < max_len; t++)
        for (uint32_t i = 0; i < half; i++) {
            float freq = powf(m->rope_theta, -(float)(2 * i) / hd);
            m->cosv[t * half + i] = cosf(t * freq);
            m->sinv[t * half + i] = sinf(t * freq);
        }
}

/* ------------------------------------------------------------- engram --- */

#define ENGRAM_SEED  0x9E3779B9u
#define ENGRAM_PRIME 0x01000193u

static void engram_step(Needle *m) {
    uint32_t C = m->d_model, T = m->n_tables, slots = m->slots;
    uint32_t heads = T / m->n_orders;
    uint32_t maxo = 0;
    for (uint32_t i = 0; i < m->n_orders; i++) if (m->orders[i] > maxo) maxo = m->orders[i];

    /* hash current-position indices */
    uint32_t idx[MAX_TABLES]; float ok[MAX_TABLES];
    uint32_t t = 0;
    for (uint32_t oi = 0; oi < m->n_orders; oi++) {
        uint32_t order = m->orders[oi];
        for (uint32_t hh = 0; hh < heads; hh++, t++) {
            uint32_t acc = (uint32_t)(ENGRAM_SEED * (oi * heads + hh + 1));
            int valid = 1;
            for (uint32_t j = 0; j < order; j++) {
                int hp = (int)m->hist_len - 1 - (int)j;
                uint32_t tok = hp >= 0 ? (uint32_t)m->hist[hp] : 0u;
                acc = (acc ^ tok) * ENGRAM_PRIME;
                if (j == order - 1) valid = hp >= 0;
            }
            acc ^= acc >> 15;
            idx[t] = acc % slots;
            ok[t] = valid ? 1.0f : 0.0f;
        }
    }

    for (uint32_t s = 0; s < m->n_sites; s++) {
        EngramSite *E = &m->engrams[s];
        /* e = concat of fetched (dequantized) table rows, masked */
        for (uint32_t ti = 0; ti < T; ti++) {
            cq_row(m, &E->tables, ti * slots + idx[ti], m->e + ti * m->sub_dim);
            if (ok[ti] == 0.0f)
                memset(m->e + ti * m->sub_dim, 0, m->sub_dim * 4);
        }
        cq_apply(m, &E->key_proj, m->e, m->ek + s * C, m->xh);
        float *v_now = m->ev + s * C;   /* reuse as scratch for v_now */
        cq_apply(m, &E->value_proj, m->e, v_now, m->xh);

        /* push into ring, then tap-mix */
        uint32_t depth = m->ering_depth;
        uint32_t slot = m->epos % depth;
        float *ring = m->ering + ((size_t)s * depth + slot) * C;
        memcpy(ring, v_now, C * 4);
        m->ering_valid[s * depth + slot] = 1;

        static float mixed[512];
        memset(mixed, 0, C * 4);
        for (uint32_t j = 0; j < m->taps; j++) {
            int p = (int)m->epos - (int)(j * maxo);
            if (p < 0) continue;
            uint32_t sl = (uint32_t)p % depth;
            if (!m->ering_valid[s * depth + sl]) continue;
            const float *vp = m->ering + ((size_t)s * depth + sl) * C;
            const float *tw = E->taps + (size_t)j * C;
            for (uint32_t c = 0; c < C; c++) mixed[c] += tw[c] * vp[c];
        }
        memcpy(m->ev + s * C, mixed, C * 4);
    }
    m->epos++;
}

/* ------------------------------------------------------------- profiling */

#ifdef NEEDLE_PROF
#ifdef ESP_PLATFORM
#include "esp_timer.h"
#define PROF_NOW() esp_timer_get_time()
#else
#include <time.h>
static int64_t PROF_NOW(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}
#endif
int64_t g_prof[8];
static const char *g_prof_names[8] = {"mhc", "qkv+gate", "attn", "out+hada", "engram", "logits", "misc", ""};
#define PROF_T(slot, expr) do { int64_t _t0 = PROF_NOW(); expr; g_prof[slot] += PROF_NOW() - _t0; } while (0)
void needle_prof_dump(void) {
    int64_t total = 0;
    for (int i = 0; i < 7; i++) total += g_prof[i];
    for (int i = 0; i < 7; i++) {
        if (g_prof[i])
            printf("[prof] %-9s %8lld us (%2d%%)\n", g_prof_names[i],
                   (long long)g_prof[i], (int)(g_prof[i] * 100 / (total ? total : 1)));
        g_prof[i] = 0;
    }
}
#else
#define PROF_T(slot, expr) expr
void needle_prof_dump(void) {}
#endif

/* --------------------------------------------------------------- forward */

static int g_trace = -1;
static void tr(const char *tag, int layer, const float *x, int n) {
    if (g_trace < 0) g_trace = getenv("NEEDLE_TRACE") != NULL;
    if (!g_trace) return;
    for (int i = 0; i < n; i++)
        if (isnan(x[i]) || isinf(x[i])) {
            fprintf(stderr, "[trace] %s L%d: non-finite at [%d] = %f\n", tag, layer, i, x[i]);
            exit(2);
        }
}

typedef struct {
    const Needle *m; uint32_t layer, lo, pos, kv_start, kv_end;
} AttnJob;

static void attn_job_fn(void *p) {
    AttnJob *j = (AttnJob *)p;
    const Needle *m = j->m;
    uint32_t hd = m->head_dim, KV = m->n_kv, H = m->n_heads, reps = H / KV;
    uint32_t i = j->layer, lo = j->lo, pos = j->pos;
    float aw[512];
    float inv_sq = 1.0f / sqrtf((float)hd);
    uint32_t ka = m->kv_alloc;
    for (uint32_t kk = j->kv_start; kk < j->kv_end; kk++) {
        size_t kbase = (((size_t)i * KV + kk) * ka) * hd;
        size_t sbase = ((size_t)i * KV + kk) * ka;
        for (uint32_t r = 0; r < reps; r++) {
            const float *qh = m->q + (kk * reps + r) * hd;
            uint32_t T = pos + 1 - lo;
            for (uint32_t tt = 0; tt < T; tt++) {
                uint32_t sl = (lo + tt) % ka;
                const int8_t *kp = m->k_cache + kbase + (size_t)sl * hd;
                float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                for (uint32_t d = 0; d < hd; d += 4) {
                    s0 += qh[d] * kp[d];         s1 += qh[d + 1] * kp[d + 1];
                    s2 += qh[d + 2] * kp[d + 2]; s3 += qh[d + 3] * kp[d + 3];
                }
                aw[tt] = ((s0 + s1) + (s2 + s3)) * m->k_scale[sbase + sl] * inv_sq;
            }
            softmax_(aw, (int)T);
            float *outp = m->att_out + (kk * reps + r) * hd;
            memset(outp, 0, hd * 4);
            for (uint32_t tt = 0; tt < T; tt++) {
                uint32_t sl = (lo + tt) % ka;
                const int8_t *vp = m->v_cache + kbase + (size_t)sl * hd;
                float w = aw[tt] * m->v_scale[sbase + sl];
                for (uint32_t d = 0; d < hd; d++) outp[d] += w * vp[d];
            }
        }
    }
}

const float *needle_step_ex(Needle *m, int token, uint32_t pos, int want_logits) {
    uint32_t C = m->d_model, n = m->lanes, L = m->n_layers;
    uint32_t H = m->n_heads, KV = m->n_kv, hd = m->head_dim;
    uint32_t A = H * hd, half = hd / 2;

    m->hist[m->hist_len++] = token;

    /* x lanes = embedding row * sqrt(C), broadcast */
    cq_row(m, &m->embedding, (uint32_t)token, m->x);
    float sc = sqrtf((float)C);
    for (uint32_t c = 0; c < C; c++) m->x[c] *= sc;
    for (uint32_t l = 1; l < n; l++) memcpy(m->x + l * C, m->x, C * 4);

    if (m->n_sites) PROF_T(4, engram_step(m));
    tr("emb", -1, m->x, (int)(n*C));
    tr("ekv", -1, m->ev, (int)(m->n_sites*C));

    const float *cos_p = m->cosv + (size_t)pos * half;
    const float *sin_p = m->sinv + (size_t)pos * half;

    for (uint32_t i = 0; i < L; i++) {
        Layer *lp = &m->layers[i];
        /* nx = rms_unit(x.flatten) */
        rms_unit(m->x, m->nx, (int)(n * C));

        /* hpre = sigmoid(a_pre*(phi_pre_i @ nx) + b_pre + pre_off) */
        float hpre[MAX_LANES], tmpl[MAX_LANES * MAX_LANES];
        {
            /* phi_pre rows for layer i: rows [i*n, (i+1)*n) of (L*n, n*C) */
            PROF_T(0, cq_prepare_x(m, &m->phi_pre, m->nx, m->xh2));
            CQMat sub = m->phi_pre;
            uint32_t row_bytes = sub.in_pad * sub.bits / 8;
            sub.packed += (uint64_t)i * n * row_bytes;
            sub.norms += (uint64_t)i * n * (sub.in_pad / sub.group);
            sub.out = n;
            PROF_T(0, cq_matvec(m, &sub, m->xh2, hpre));
        }
        uint32_t own = i % n;
        for (uint32_t l = 0; l < n; l++) {
            float off = (l == own) ? 4.0f : -4.0f;
            hpre[l] = sigmoidf_(m->a_pre[i] * hpre[l] + m->b_pre[i * n + l] + off);
        }
        for (uint32_t c = 0; c < C; c++) {
            float s = 0;
            for (uint32_t l = 0; l < n; l++) s += hpre[l] * m->x[l * C + c];
            m->u[c] = s;
        }
        tr("hpre", (int)i, hpre, (int)n);
        tr("u", (int)i, m->u, (int)C);

        /* engram injection */
        float *bx = m->u;
        static float bxbuf[512];
        for (uint32_t s = 0; s < m->n_sites; s++) {
            if (m->sites[s] == i) {
                static float un[512], ekn[512];
                rms_unit(m->u, un, (int)C);
                rms_unit(m->ek + s * C, ekn, (int)C);
                float dot = 0;
                for (uint32_t c = 0; c < C; c++) dot += un[c] * ekn[c];
                float alpha = sigmoidf_(dot / sqrtf((float)C));
                const float *ev = m->ev + s * C;
                for (uint32_t c = 0; c < C; c++) bxbuf[c] = m->u[c] + alpha * ev[c];
                bx = bxbuf;
                break;
            }
        }

        /* ---- attention ---- */
        zcrms(bx, lp->norm_in, m->h, (int)C);
        tr("h", (int)i, m->h, (int)C);
        PROF_T(1, cq_prepare_x(m, &lp->q_proj, m->h, m->xh));
        tr("xh", (int)i, m->xh, (int)lp->q_proj.in_pad);
        PROF_T(1, cq_matvec_mt(m, &lp->q_proj, m->xh, m->q));
        PROF_T(1, cq_matvec_mt(m, &lp->k_proj, m->xh, m->k));
        PROF_T(1, cq_matvec_mt(m, &lp->v_proj, m->xh, m->v));
        tr("bx", (int)i, bx, (int)C);
        tr("qkv", (int)i, m->q, (int)A);

        for (uint32_t hh = 0; hh < H; hh++)
            zcrms(m->q + hh * hd, lp->q_norm, m->q + hh * hd, (int)hd);
        for (uint32_t kk = 0; kk < KV; kk++)
            zcrms(m->k + kk * hd, lp->k_norm, m->k + kk * hd, (int)hd);
        /* rope */
        for (uint32_t hh = 0; hh < H; hh++) {
            float *qq = m->q + hh * hd;
            for (uint32_t d = 0; d < half; d++) {
                float x1 = qq[d], x2 = qq[d + half];
                qq[d] = x1 * cos_p[d] - x2 * sin_p[d];
                qq[d + half] = x2 * cos_p[d] + x1 * sin_p[d];
            }
        }
        for (uint32_t kk = 0; kk < KV; kk++) {
            float *kkp = m->k + kk * hd;
            for (uint32_t d = 0; d < half; d++) {
                float x1 = kkp[d], x2 = kkp[d + half];
                kkp[d] = x1 * cos_p[d] - x2 * sin_p[d];
                kkp[d + half] = x2 * cos_p[d] + x1 * sin_p[d];
            }
        }

        /* int8 KV store: per-vector absmax scale */
        uint32_t slot = pos % m->kv_alloc;
        for (uint32_t kk = 0; kk < KV; kk++) {
            size_t base = (((size_t)i * KV + kk) * m->kv_alloc + slot) * hd;
            size_t sbase = ((size_t)i * KV + kk) * m->kv_alloc + slot;
            float mx = 1e-12f;
            for (uint32_t d = 0; d < hd; d++) {
                float a = fabsf(m->k[kk * hd + d]);
                if (a > mx) mx = a;
            }
            m->k_scale[sbase] = mx / 127.0f;
            for (uint32_t d = 0; d < hd; d++)
                m->k_cache[base + d] = (int8_t)lrintf(m->k[kk * hd + d] / m->k_scale[sbase]);
            mx = 1e-12f;
            for (uint32_t d = 0; d < hd; d++) {
                float a = fabsf(m->v[kk * hd + d]);
                if (a > mx) mx = a;
            }
            m->v_scale[sbase] = mx / 127.0f;
            for (uint32_t d = 0; d < hd; d++)
                m->v_cache[base + d] = (int8_t)lrintf(m->v[kk * hd + d] / m->v_scale[sbase]);
        }

#ifdef NEEDLE_PROF
        int64_t _attn_t0 = PROF_NOW();
#endif
        uint32_t lo = 0;
        if (m->kv_window && pos + 1 > m->kv_window) lo = pos + 1 - m->kv_window;
        uint32_t reps = H / KV;
        AttnJob aj0 = { m, i, lo, pos, 0, KV / 2 };
        AttnJob aj1 = { m, i, lo, pos, KV / 2, KV };
        if (par_submit(attn_job_fn, &aj1)) {
            attn_job_fn(&aj0);
            par_wait();
        } else {
            aj0.kv_end = KV;
            attn_job_fn(&aj0);
        }
        (void)reps;

#ifdef NEEDLE_PROF
        g_prof[2] += PROF_NOW() - _attn_t0;
#endif
        /* gate + out_proj (gate_proj input is h, reuse prepared xh) */
        static float gate[2048];
        PROF_T(1, cq_matvec_mt(m, &lp->gate_proj, m->xh, gate));
        for (uint32_t d = 0; d < A; d++) m->att_out[d] *= sigmoidf_(gate[d]);
        tr("attout", (int)i, m->att_out, (int)A);
        static float attn[512];
        PROF_T(3, cq_apply(m, &lp->out_proj, m->att_out, attn, m->xh));
        static float attn_n[512];
        zcrms(attn, lp->post_norm, attn_n, (int)C);
        float gsc = sigmoidf_(lp->attn_gate);
        static float y[512];
        for (uint32_t c = 0; c < C; c++) y[c] = bx[c] + gsc * attn_n[c];

        /* ---- hadamard mlp ---- */
        static float hh2[512];
        zcrms(y, lp->pre_hada, hh2, (int)C);
        uint32_t N = m->hada_n;
        float invn = 1.0f / sqrtf((float)N);
        for (uint32_t c = 0; c < N; c++) m->z[c] = (c < C ? hh2[c] : 0.0f) * lp->d1[c];
        fwht(m->z, (int)N);
        for (uint32_t c = 0; c < N; c++) m->z[c] = siluf_(m->z[c] * invn * lp->d2[c]);
        fwht(m->z, (int)N);
        for (uint32_t c = 0; c < C; c++) y[c] += m->z[c] * invn * lp->d3[c];

        /* y -= u */
        for (uint32_t c = 0; c < C; c++) y[c] -= m->u[c];
        tr("y", (int)i, y, (int)C);

        /* hpost + sinkhorn res mix */
        float hpost[MAX_LANES];
        {   /* xh2 still holds FWHT(nx) from phi_pre — same layout */
            CQMat sub = m->phi_post;
            uint32_t row_bytes = sub.in_pad * sub.bits / 8;
            sub.packed += (uint64_t)i * n * row_bytes;
            sub.norms += (uint64_t)i * n * (sub.in_pad / sub.group);
            sub.out = n;
            PROF_T(0, cq_matvec(m, &sub, m->xh2, hpost));
        }
        for (uint32_t l = 0; l < n; l++) {
            float off = (l == own) ? 0.0f : -4.0f;
            hpost[l] = 2.0f * sigmoidf_(m->a_post[i] * hpost[l]
                                        + m->b_post[i * n + l] + off);
        }
        {
            CQMat sub = m->phi_res;
            uint32_t row_bytes = sub.in_pad * sub.bits / 8;
            sub.packed += (uint64_t)i * n * n * row_bytes;
            sub.norms += (uint64_t)i * n * n * (sub.in_pad / sub.group);
            sub.out = n * n;
            PROF_T(0, cq_matvec(m, &sub, m->xh2, tmpl));
        }
        for (uint32_t l = 0; l < n * n; l++)
            tmpl[l] = m->a_res[i] * tmpl[l] + m->b_res[i * n * n + l];
        PROF_T(0, sinkhorn(tmpl, (int)n, 20));

        /* x = hres @ x + hpost[:,None]*y */
        static float xn[MAX_LANES * 512];
        for (uint32_t l = 0; l < n; l++) {
            for (uint32_t c = 0; c < C; c++) {
                float s = 0;
                for (uint32_t j = 0; j < n; j++) s += tmpl[l * n + j] * m->x[j * C + c];
                xn[l * C + c] = s + hpost[l] * y[c];
            }
        }
        tr("hres", (int)i, tmpl, (int)(n*n));
        memcpy(m->x, xn, (size_t)n * C * 4);
        tr("xnew", (int)i, m->x, (int)(n*C));
    }

    if (!want_logits) return NULL;

    /* mean over lanes, final norm, logits via embedding (tied) */
    static float xm[512];
    for (uint32_t c = 0; c < C; c++) {
        float s = 0;
        for (uint32_t l = 0; l < n; l++) s += m->x[l * C + c];
        xm[c] = s / n;
    }
    static float xf[512];
    zcrms(xm, m->final_norm, xf, (int)C);
    PROF_T(5, cq_apply(m, &m->embedding, xf, m->logits, m->xh));
    return m->logits;
}

const float *needle_step(Needle *m, int token, uint32_t pos) {
    return needle_step_ex(m, token, pos, 1);
}

/* -------------------------------------------------------------- tokenizer */

#define SP_SPACE "\xe2\x96\x81"   /* U+2581 */

static int piece_id(const Needle *m, const char *s, int len) {
    uint32_t h = 2166136261u;
    for (int k = 0; k < len; k++) h = (h ^ (uint8_t)s[k]) * 16777619u;
    h &= m->p2id_cap - 1;
    while (m->p2id_slot[h] >= 0) {
        int32_t id = m->p2id_slot[h];
        if (m->piece_len[id] == len && memcmp(m->pieces[id], s, len) == 0)
            return id;
        h = (h + 1) & (m->p2id_cap - 1);
    }
    return -1;
}

/* greedy BPE over a char segment (RefTokenizer._bpe) */
static int bpe_segment(const Needle *m, const char *seg, int seglen,
                       int *out, int max_out) {
    /* start: one symbol per UTF-8 char */
    static int starts[8200];
    int n_sym = 0;
    for (int i = 0; i < seglen && n_sym < 8192;) {
        starts[n_sym++] = i;
        unsigned char c = (unsigned char)seg[i];
        i += (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
    }
    starts[n_sym] = seglen;
    while (n_sym > 1) {
        float best = -1e30f; int bj = -1, bid = -1;
        for (int j = 0; j < n_sym - 1; j++) {
            int a = starts[j], b = starts[j + 2];
            int id = piece_id(m, seg + a, b - a);
            if (id >= 0 && (bj < 0 || m->scores[id] > best)) {
                best = m->scores[id]; bj = j; bid = id;
            }
        }
        if (bj < 0) break;
        (void)bid;
        memmove(starts + bj + 1, starts + bj + 2, (n_sym - bj - 1) * sizeof(int));
        n_sym--;
    }
    int cnt = 0;
    for (int j = 0; j < n_sym && cnt < max_out; j++) {
        int a = starts[j], b = starts[j + 1];
        int id = piece_id(m, seg + a, b - a);
        if (id >= 0) out[cnt++] = id;
        else if (m->byte_fallback) {
            for (int i = a; i < b && cnt < max_out; i++)
                out[cnt++] = m->byte_id[(unsigned char)seg[i]];
        } else out[cnt++] = (int)m->unk_id;
    }
    return cnt;
}

int needle_encode(const Needle *m, const char *text, int *out, int max_out) {
    /* escape spaces to U+2581, optional dummy prefix */
    size_t tl = strlen(text);
    char *esc = (char *)malloc(tl * 3 + 4);
    size_t e = 0;
    if (m->add_dummy) { memcpy(esc + e, SP_SPACE, 3); e += 3; }
    for (size_t i = 0; i < tl; i++) {
        if (text[i] == ' ') { memcpy(esc + e, SP_SPACE, 3); e += 3; }
        else esc[e++] = text[i];
    }
    esc[e] = 0;

    int cnt = 0;
    size_t i = 0, seg_start = 0;
    while (i < e) {
        /* longest marker (USER_DEFINED piece) match at i */
        int mk = -1, mklen = 0;
        for (int p = 0; p < m->n_markers; p++) {
            int id = m->marker_ids[p];
            int ln = m->piece_len[id];
            if (i + (size_t)ln <= e && memcmp(esc + i, m->pieces[id], ln) == 0) {
                mk = id; mklen = ln;
                break;
            }
        }
        if (mk >= 0) {
            cnt += bpe_segment(m, esc + seg_start, (int)(i - seg_start),
                               out + cnt, max_out - cnt);
            if (cnt < max_out) out[cnt++] = mk;
            i += mklen;
            seg_start = i;
        } else i++;
    }
    cnt += bpe_segment(m, esc + seg_start, (int)(e - seg_start), out + cnt,
                       max_out - cnt);
    free(esc);
    return cnt;
}

int needle_decode_piece(const Needle *m, int id, char *buf, int buflen) {
    uint8_t t = m->types[id];
    if (t == 2 || t == 1) { buf[0] = 0; return 0; }   /* control/unknown */
    if (t == 4) {
        unsigned b;
        if (sscanf(m->pieces[id], "<0x%02X>", &b) == 1) {
            buf[0] = (char)b; buf[1] = 0; return 1;
        }
        buf[0] = 0; return 0;
    }
    int ln = m->piece_len[id], o = 0;
    for (int i = 0; i < ln && o < buflen - 1;) {
        if (i + 3 <= ln && memcmp(m->pieces[id] + i, SP_SPACE, 3) == 0) {
            buf[o++] = ' '; i += 3;
        } else buf[o++] = m->pieces[id][i++];
    }
    buf[o] = 0;
    return o;
}

/* ----------------------------------------------- constrained tool calling
 * Grammar-guided greedy decode for the fixed needle output shape:
 *   [{"name":"<tool>","arguments":{"<param>":<value>,...}}]
 * Structure text is teacher-forced; tool/param names are decoded with a
 * prefix-trie constraint over the schema; integer values are digit-masked.
 * This mirrors what the closed engine's grammar compiler does.            */

#ifdef ESP_PLATFORM
#include "esp_timer.h"
static int64_t needle_now_us(void) { return esp_timer_get_time(); }
#else
#include <time.h>
static int64_t needle_now_us(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}
#endif

typedef struct {
    int prefill_tok, decode_tok;
    int64_t prefill_us, decode_us;
} NeedleStats;
NeedleStats g_needle_stats;

#define NT_MAX_TOOLS 16
#define NT_MAX_PARAMS 12
typedef struct { char name[48]; uint8_t is_num; uint8_t required; } NParam;
typedef struct { char name[64]; NParam params[NT_MAX_PARAMS]; int n_params; } NTool;

static const char *js_find(const char *p, const char *key) {
    char pat[80];
    snprintf(pat, sizeof pat, "\"%s\":", key);
    return strstr(p, pat);
}

/* minimal parser for our flattened tools format:
 * [{"name":"x","description":"..","parameters":{"p":{"type":"integer",..},..}},..] */
static int needle_parse_tools(const char *json, NTool *tools, int max_tools) {
    int n = 0;
    const char *p = json;
    while (n < max_tools && (p = js_find(p, "name")) != NULL) {
        p += 7;
        while (*p && *p != '"') p++;
        if (*p) p++;
        const char *e = strchr(p, '"');
        if (!e) break;
        NTool *t = &tools[n];
        memset(t, 0, sizeof *t);
        size_t ln = (size_t)(e - p) < sizeof t->name - 1 ? (size_t)(e - p) : sizeof t->name - 1;
        memcpy(t->name, p, ln);
        p = e + 1;
        const char *params = js_find(p, "parameters");
        const char *next_tool = js_find(p, "name");
        if (params && (!next_tool || params < next_tool)) {
            const char *props = js_find(params, "properties");
            if (props && (!next_tool || props < next_tool)) params = props;
            params = strchr(params, '{');
            int depth = 1;
            const char *q = params + 1;
            while (*q && depth > 0) {
                if (*q == '{') depth++;
                else if (*q == '}') depth--;
                else if (*q == '"' && depth == 1 && t->n_params < NT_MAX_PARAMS) {
                    const char *pe = strchr(q + 1, '"');
                    if (!pe) break;
                    NParam *pr = &t->params[t->n_params];
                    size_t pl = (size_t)(pe - q - 1) < sizeof pr->name - 1
                              ? (size_t)(pe - q - 1) : sizeof pr->name - 1;
                    memcpy(pr->name, q + 1, pl);
                    pr->name[pl] = 0;
                    const char *ty = js_find(pe, "type");
                    pr->is_num = ty && (strncmp(ty + 8, "integer", 7) == 0
                                        || strncmp(ty + 8, "number", 6) == 0);
                    pr->required = 0;
                    t->n_params++;
                    /* skip this param's spec object */
                    const char *ob = strchr(pe, '{');
                    if (ob) {
                        int d2 = 1; q = ob + 1;
                        while (*q && d2 > 0) { if (*q=='{') d2++; else if (*q=='}') d2--; q++; }
                        continue;
                    }
                }
                q++;
            }
            p = q;
        }
        /* mark required params: "required":["a","b"] within this tool */
        {
            const char *rq = js_find(p - 1, "required");
            const char *nt = js_find(p, "name");
            if (rq && (!nt || rq < nt)) {
                const char *end = strchr(rq, ']');
                for (int i = 0; i < t->n_params; i++) {
                    char pat[64];
                    snprintf(pat, sizeof pat, "\"%s\"", t->params[i].name);
                    const char *hit = strstr(rq, pat);
                    if (hit && end && hit < end) t->params[i].required = 1;
                }
            }
        }
        n++;
    }
    return n;
}

/* encode without the leading dummy-prefix space (mid-sequence fragments) */
static int encode_raw(const Needle *m, const char *text, int *out, int max_out) {
    Needle tmp = *m;            /* shallow: shares tables, flips one flag */
    tmp.add_dummy = 0;
    return needle_encode(&tmp, text, out, max_out);
}

typedef struct { const float *logits; uint32_t pos; } DecCtx;

static void dc_feed(Needle *m, DecCtx *dc, int tok) {
    dc->logits = needle_step(m, tok, dc->pos++);
}

static float log_softmax_at(const float *logits, uint32_t n, int idx) {
    float mx = logits[0];
    for (uint32_t i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    float s = 0;
    for (uint32_t i = 0; i < n; i++) s += expf(logits[i] - mx);
    return logits[idx] - mx - logf(s);
}

/* Score each candidate string by mean token logprob (teacher-forced), using
 * cheap counter rewind: the KV cache and engram ring are simply overwritten
 * when decoding resumes from the saved position. Feeds the winner for real
 * and appends it to outbuf. Returns winner index. */
static int dc_pick_scored(Needle *m, DecCtx *dc, const char **cand, int n_cand,
                          char *outbuf, size_t *olen) {
    uint32_t pos0 = dc->pos, hist0 = m->hist_len, epos0 = m->epos;
    /* stash logits at the branch point (needle_step overwrites m->logits);
     * heap (PSRAM on ESP32) — cold data, don't burn internal SRAM */
    static float *base = NULL;
    if (!base) base = (float *)malloc(8192 * 4);
    memcpy(base, dc->logits, m->vocab * 4);
    /* Pre-rank by first-token logprob — that costs nothing (we already hold
     * the logits) and lets us teacher-force only the few plausible names
     * instead of all of them; each skipped candidate is a saved forward pass. */
    static int order[NT_MAX_TOOLS + NT_MAX_PARAMS];
    static float first_lp[NT_MAX_TOOLS + NT_MAX_PARAMS];
    for (int i = 0; i < n_cand; i++) {
        int ids[24];
        int n = encode_raw(m, cand[i], ids, 24);
        first_lp[i] = (n > 0) ? log_softmax_at(base, m->vocab, ids[0]) : -1e30f;
        order[i] = i;
    }
    for (int a = 0; a < n_cand; a++)
        for (int b = a + 1; b < n_cand; b++)
            if (first_lp[order[b]] > first_lp[order[a]]) {
                int t = order[a]; order[a] = order[b]; order[b] = t;
            }
    int n_score = n_cand < 3 ? n_cand : 3;

    float best_score = -1e30f;
    int best = -1;
    for (int oi = 0; oi < n_score; oi++) {
        int i = order[oi];
        int ids[24];
        int n = encode_raw(m, cand[i], ids, 24);
        if (n <= 0) continue;
        float lp = log_softmax_at(base, m->vocab, ids[0]);
        const float *lg = NULL;
        for (int k = 0; k < n - 1; k++) {
            lg = needle_step(m, ids[k], dc->pos + (uint32_t)k);
            lp += log_softmax_at(lg, m->vocab, ids[k + 1]);
        }
        m->hist_len = hist0; m->epos = epos0;   /* rewind */
        lp /= (float)n;
        if (getenv("NEEDLE_DEBUG")) fprintf(stderr, "[cand] %-14s %.3f (%d tok)\n", cand[i], lp, n);
        if (lp > best_score) { best_score = lp; best = i; }
    }
    if (best < 0) return -1;
    {
        int ids[24];
        int n = encode_raw(m, cand[best], ids, 24);
        dc->pos = pos0;
        for (int k = 0; k < n; k++) dc_feed(m, dc, ids[k]);
    }
    size_t cl = strlen(cand[best]);
    memcpy(outbuf + *olen, cand[best], cl);
    *olen += cl;
    return best;
}

/* Emit exactly `text`, but let the model pick its own tokenization: at each
 * step only tokens whose piece is a prefix of the remaining text are allowed,
 * and the argmax among those is fed. Keeps the context canonical. */
static void dc_force(Needle *m, DecCtx *dc, const char *text, char *outbuf, size_t *olen) {
    size_t off = 0, tl = strlen(text);
    while (off < tl) {
        int best = -1;
        float bl = -1e30f;
        for (uint32_t t = 0; t < m->n_pieces; t++) {
            if (m->types[t] == 1 || m->types[t] == 2) continue;
            int pl = m->piece_len[t];
            if (pl == 0 || (size_t)pl > tl - off) continue;
            if (memcmp(m->pieces[t], text + off, pl) != 0) continue;
            if (dc->logits[t] > bl) { bl = dc->logits[t]; best = (int)t; }
        }
        if (best < 0) {   /* byte fallback */
            best = m->byte_id[(unsigned char)text[off]];
            if (best < 0) return;
        }
        off += m->piece_len[best];
        dc_feed(m, dc, best);
    }
    memcpy(outbuf + *olen, text, tl);
    *olen += tl;
}

/* greedy decode constrained to a set of candidate strings; returns index of
 * the completed candidate or -1 */
static int dc_trie(Needle *m, DecCtx *dc, const char **cand, int n_cand,
                   char *outbuf, size_t *olen) {
    static size_t off[NT_MAX_TOOLS + NT_MAX_PARAMS];
    static int live[NT_MAX_TOOLS + NT_MAX_PARAMS];
    for (int i = 0; i < n_cand; i++) { off[i] = 0; live[i] = 1; }
    for (int step = 0; step < 24; step++) {
        int best = -1;
        float bl = -1e30f;
        for (uint32_t t = 0; t < m->n_pieces; t++) {
            if (m->types[t] == 2 || m->types[t] == 1) continue;
            const char *pc = m->pieces[t];
            int pl = m->piece_len[t];
            if (pl == 0) continue;
            int ok = 0;
            for (int i = 0; i < n_cand && !ok; i++) {
                if (!live[i]) continue;
                size_t rem = strlen(cand[i]) - off[i];
                if ((size_t)pl <= rem && memcmp(pc, cand[i] + off[i], pl) == 0) ok = 1;
            }
            if (ok && dc->logits[t] > bl) { bl = dc->logits[t]; best = (int)t; }
        }
        if (best < 0) return -1;
        const char *pc = m->pieces[best];
        int pl = m->piece_len[best];
        for (int i = 0; i < n_cand; i++) {
            if (!live[i]) continue;
            size_t rem = strlen(cand[i]) - off[i];
            if ((size_t)pl <= rem && memcmp(pc, cand[i] + off[i], pl) == 0) off[i] += pl;
            else live[i] = 0;
        }
        memcpy(outbuf + *olen, pc, pl); *olen += pl;
        dc_feed(m, dc, best);
        for (int i = 0; i < n_cand; i++)
            if (live[i] && off[i] == strlen(cand[i])) return i;
    }
    return -1;
}

static int piece_all_digits(const Needle *m, uint32_t t) {
    if (m->types[t] != 0) return 0;
    for (int i = 0; i < m->piece_len[t]; i++)
        if (m->pieces[t][i] < '0' || m->pieces[t][i] > '9') return 0;
    return m->piece_len[t] > 0;
}

/* Run one constrained tool call. Returns length of JSON written to out. */
int needle_toolcall(Needle *m, const char *query, const char *tools_json,
                    char *out, size_t outsz) {
    static NTool tools[NT_MAX_TOOLS];
    int n_tools = needle_parse_tools(tools_json, tools, NT_MAX_TOOLS);
    if (n_tools <= 0) return -1;

    /* Split at the </tools> marker: markers are atomic tokens, so the prefix
     * tokenization is always a true prefix of the whole prompt's. */
    static char prefix[8192], rest[4096];
    snprintf(prefix, sizeof prefix, "<|im_start|>user\n<tools>%s</tools>", tools_json);
    snprintf(rest, sizeof rest, "\n%s<|im_end|>\n<|im_start|>assistant\n", query);

    static int pids[4096], rids[1024];
    pids[0] = (int)m->bos_id;
    int n_pids = 1 + needle_encode(m, prefix, pids + 1, 4095);
    int n_rids = encode_raw(m, rest, rids, 1024);
    int n_ids = n_pids + n_rids;

    uint32_t th = 2166136261u;
    for (const char *c = tools_json; *c; c++) th = (th ^ (uint8_t)*c) * 16777619u;

    size_t ering_bytes = (size_t)m->n_sites * m->ering_depth * m->d_model * 4;
    size_t evalid_bytes = (size_t)m->n_sites * m->ering_depth;

    DecCtx dc = { NULL, 0 };
    int64_t t_pre = needle_now_us();
    int reused = (m->px_valid && m->px_hash == th && m->px_n == (uint32_t)n_pids
                  && m->max_len >= (uint32_t)(n_ids + 200));
    if (reused) {
        /* KV rows for the tools block are still live in the ring */
        m->hist_len = m->px_hist_len;
        m->epos = m->px_epos;
        memcpy(m->ering, m->px_ering, ering_bytes);
        memcpy(m->ering_valid, m->px_ering_valid, evalid_bytes);
    } else {
        needle_reset(m, (uint32_t)(n_ids + 200));
        if (m->max_len == 0) return -1;
        ering_bytes = (size_t)m->n_sites * m->ering_depth * m->d_model * 4;
        evalid_bytes = (size_t)m->n_sites * m->ering_depth;
        for (int p = 0; p < n_pids; p++) needle_step_ex(m, pids[p], (uint32_t)p, 0);
        m->px_hash = th;
        m->px_n = (uint32_t)n_pids;
        m->px_hist_len = m->hist_len;
        m->px_epos = m->epos;
        memcpy(m->px_ering, m->ering, ering_bytes);
        memcpy(m->px_ering_valid, m->ering_valid, evalid_bytes);
        m->px_valid = 1;
    }
    dc.pos = (uint32_t)n_pids;
    for (int k = 0; k < n_rids; k++)
        dc.logits = needle_step_ex(m, rids[k], dc.pos++, k == n_rids - 1);
    int64_t t_dec = needle_now_us();
    g_needle_stats.prefill_tok = reused ? n_rids : n_ids;
    g_needle_stats.prefill_us = t_dec - t_pre;

    size_t olen = 0;
    (void)outsz;
    /* free-run the reasoning section until the model opens <tool_call>
     * (capped); the think text is not part of the returned JSON */
    for (int s = 0; s < 90; s++) {
        int best = 0;
        for (uint32_t t = 1; t < m->n_pieces; t++)
            if (dc.logits[t] > dc.logits[best]) best = (int)t;
        if (m->types[best] == 3 && strcmp(m->pieces[best], "<tool_call>") == 0) {
            dc_feed(m, &dc, best);
            break;
        }
        if (best == (int)m->eos_id ||
            (m->types[best] == 3 && strcmp(m->pieces[best], "<|im_end|>") == 0)) {
            /* model refuses to call a tool */
            memcpy(out, "[]", 3);
            return 2;
        }
        dc_feed(m, &dc, best);
    }
    dc_force(m, &dc, "[{\"name\":\"", out, &olen);

    const char *names[NT_MAX_TOOLS];
    for (int i = 0; i < n_tools; i++) names[i] = tools[i].name;
    int ti = dc_pick_scored(m, &dc, names, n_tools, out, &olen);
    if (ti < 0) return -1;
    NTool *T = &tools[ti];

    dc_force(m, &dc, "\",\"arguments\":{", out, &olen);
    int used[NT_MAX_PARAMS] = {0};
    int emitted = 0;
    int next_quote_consumed = 0;
    for (int round = 0; round < T->n_params; round++) {
        /* choose param name among unused */
        const char *cands[NT_MAX_PARAMS];
        int map[NT_MAX_PARAMS], nc = 0;
        for (int i = 0; i < T->n_params; i++)
            if (!used[i]) { cands[nc] = T->params[i].name; map[nc++] = i; }
        if (!nc) break;
        if (next_quote_consumed) { out[olen++] = '"'; next_quote_consumed = 0; }
        else dc_force(m, &dc, "\"", out, &olen);
        int pi = dc_trie(m, &dc, cands, nc, out, &olen);
        if (pi < 0) return -1;
        NParam *P = &T->params[map[pi]];
        used[map[pi]] = 1;
        emitted++;
        dc_force(m, &dc, "\":", out, &olen);
        int spill_comma = 0, spill_quote = 0, spill_close = 0;
        if (P->is_num) {
            int got = 0;
            for (int s = 0; s < 8; s++) {
                int best = -1;
                float bl = -1e30f;
                for (uint32_t t = 0; t < m->n_pieces; t++)
                    if (piece_all_digits(m, t) && dc.logits[t] > bl) { bl = dc.logits[t]; best = (int)t; }
                if (best < 0) break;
                /* stop when a structural token outranks continuing digits */
                if (got) {
                    float best_any = -1e30f;
                    int any = 0;
                    for (uint32_t t = 0; t < m->n_pieces; t++)
                        if (dc.logits[t] > best_any) { best_any = dc.logits[t]; any = (int)t; }
                    if (!piece_all_digits(m, any)) break;
                }
                memcpy(out + olen, m->pieces[best], m->piece_len[best]);
                olen += m->piece_len[best];
                dc_feed(m, &dc, best);
                got = 1;
            }
            if (!got) { out[olen++] = '0'; }
        } else {
            dc_force(m, &dc, "\"", out, &olen);
            int quote_done = 0;
            for (int s = 0; s < 48; s++) {
                int best = 0;
                for (uint32_t t = 1; t < m->n_pieces; t++)
                    if (dc.logits[t] > dc.logits[best]) best = (int)t;
                if (m->types[best] == 3 || best == (int)m->eos_id) break;
                char piece[64];
                needle_decode_piece(m, best, piece, sizeof piece);
                if (piece[0] == '}' || piece[0] == ']') break;  /* structural drift */
                char *qm = strchr(piece, '"');
                if (qm) {
                    /* token contains the closing quote: keep the prefix, feed
                     * the whole token so the model state stays canonical.
                     * The token may also swallow following structure:
                     *   e"    -> just the close quote
                     *   ","   -> close + comma + next param's open quote
                     *   "}    -> close + end of arguments                 */
                    memcpy(out + olen, piece, (size_t)(qm - piece));
                    olen += (size_t)(qm - piece);
                    dc_feed(m, &dc, best);
                    quote_done = 1;
                    if (getenv("NEEDLE_DEBUG"))
                        fprintf(stderr, "[val-end] piece=%s\n", piece);
                    spill_comma = strchr(qm + 1, ',') != NULL;
                    spill_quote = strchr(qm + 1, '"') != NULL;
                    spill_close = strchr(qm + 1, '}') != NULL;
                    break;
                }
                memcpy(out + olen, piece, strlen(piece));
                olen += strlen(piece);
                dc_feed(m, &dc, best);
            }
            out[olen++] = '"';
            if (!quote_done) {
                olen--;
                dc_force(m, &dc, "\"", out, &olen);
            }
        }
        /* continue or close? */
        int more_params = 0, more_required = 0;
        for (int i = 0; i < T->n_params; i++)
            if (!used[i]) { more_params = 1; if (T->params[i].required) more_required = 1; }
        if (getenv("NEEDLE_DEBUG"))
            fprintf(stderr, "[branch] spill c=%d q=%d cl=%d more=%d req=%d\n",
                    spill_comma, spill_quote, spill_close, more_params, more_required);
        if (!more_params) break;
        if (spill_close && !more_required) break;
        if (spill_comma) {
            out[olen++] = ',';
            next_quote_consumed = spill_quote;
            continue;
        }
        if (!more_required) {
            float lc = -1e30f, lb = -1e30f;
            for (uint32_t t = 0; t < m->n_pieces; t++) {
                if (m->types[t] != 0 || m->piece_len[t] == 0) continue;
                char c0 = m->pieces[t][0];
                if (c0 == ',' && dc.logits[t] > lc) lc = dc.logits[t];
                if (c0 == '}' && dc.logits[t] > lb) lb = dc.logits[t];
            }
            if (lb >= lc) break;
        }
        dc_force(m, &dc, ",", out, &olen);
    }
    (void)emitted;
    olen -= 0;
    memcpy(out + olen, "}}]", 3); olen += 3;
    out[olen] = 0;
    g_needle_stats.decode_tok = (int)(dc.pos - (uint32_t)n_ids);
    g_needle_stats.decode_us = needle_now_us() - t_dec;
    return (int)olen;
}

/* ------------------------------------------------------------------ main */

#ifndef NEEDLE_NO_MAIN
#include <sys/stat.h>

static double now_ms(void);
#include <time.h>
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "model/needle2.cact";
    const char *query = argc > 2 ? argv[2] : "What's the weather in Paris right now?";
    const char *tools = argc > 3 ? argv[3] :
        "[{\"name\":\"get_weather\",\"description\":\"Get current weather for a city\","
        "\"parameters\":{\"city\":{\"type\":\"string\",\"description\":\"City name\","
        "\"required\":true}}}]";
    int max_new = argc > 4 ? atoi(argv[4]) : 96;

    FILE *f = fopen(path, "rb");
    if (!f) { perror("open"); return 1; }
    struct stat st; stat(path, &st);
    uint8_t *blob = (uint8_t *)malloc(st.st_size);
    if (fread(blob, 1, st.st_size, f) != (size_t)st.st_size) { perror("read"); return 1; }
    fclose(f);

    Needle m;
    if (needle_load(&m, blob, st.st_size) != 0) return 1;
    fprintf(stderr, "loaded: %u layers, d_model %u, vocab %u, kv_window %u\n",
            m.n_layers, m.d_model, m.vocab, m.kv_window);

    if (!getenv("NEEDLE_FREE")) {
        static char callbuf[2048];
        int reps = getenv("NEEDLE_REPEAT") ? atoi(getenv("NEEDLE_REPEAT")) : 1;
        for (int rep = 0; rep < reps - 1; rep++) {
            double ta = now_ms();
            needle_toolcall(&m, query, tools, callbuf, sizeof callbuf);
            fprintf(stderr, "call %d: %.0f ms | prefill %d tok | %s\n", rep + 1,
                    now_ms() - ta, g_needle_stats.prefill_tok, callbuf);
        }
        double t0 = now_ms();
        int cl = needle_toolcall(&m, query, tools, callbuf, sizeof callbuf);
        double t1 = now_ms();
        if (cl >= 0) {
            printf("%s\n", callbuf);
            fprintf(stderr, "constrained call in %.0f ms | prefill %d tok %.1f tok/s | decode %d tok %.1f tok/s\n",
                    t1 - t0,
                    g_needle_stats.prefill_tok,
                    g_needle_stats.prefill_tok * 1e6 / (double)g_needle_stats.prefill_us,
                    g_needle_stats.decode_tok,
                    g_needle_stats.decode_tok * 1e6 / (double)g_needle_stats.decode_us);
            return 0;
        }
        fprintf(stderr, "constrained decode failed, falling back\n");
    }

    char prompt[8192];
    snprintf(prompt, sizeof prompt,
             "<|im_start|>user\n<tools>%s</tools>\n%s<|im_end|>\n<|im_start|>assistant\n",
             tools, query);
    int ids[4096];
    ids[0] = (int)m.bos_id;
    int n_ids = 1 + needle_encode(&m, prompt, ids + 1, 4095);
    fprintf(stderr, "prompt tokens: %d\n", n_ids);
    if (getenv("NEEDLE_DEBUG")) {
        fprintf(stderr, "ids:");
        for (int p = 0; p < n_ids; p++) fprintf(stderr, " %d", ids[p]);
        fprintf(stderr, "\n");
    }

    needle_reset(&m, (uint32_t)(n_ids + max_new));
    double t0 = now_ms();
    const float *logits = NULL;
    FILE *dump = getenv("NEEDLE_DUMP") ? fopen(getenv("NEEDLE_DUMP"), "wb") : NULL;
    for (int p = 0; p < n_ids; p++) {
        logits = needle_step_ex(&m, ids[p], (uint32_t)p, dump != NULL || p == n_ids - 1);
        if (dump) fwrite(logits, 4, m.vocab, dump);
    }
    if (dump) fclose(dump);
    double t1 = now_ms();

    int pos = n_ids, produced = 0;
    char piece[64];
    while (produced < max_new) {
        int best = 0;
        for (uint32_t i = 1; i < m.vocab; i++)
            if (logits[i] > logits[best]) best = (int)i;
        if (best == (int)m.eos_id) break;
        needle_decode_piece(&m, best, piece, sizeof piece);
        fputs(piece, stdout);
        fflush(stdout);
        /* stop on <|im_end|> marker */
        if (m.types[best] == 3 && strcmp(m.pieces[best], "<|im_end|>") == 0) break;
        logits = needle_step(&m, best, (uint32_t)pos++);
        produced++;
    }
    double t2 = now_ms();
    fputc('\n', stdout);
    fprintf(stderr, "prefill: %d tok in %.0f ms (%.1f tok/s)\n",
            n_ids, t1 - t0, n_ids * 1000.0 / (t1 - t0));
    fprintf(stderr, "decode:  %d tok in %.0f ms (%.1f tok/s)\n",
            produced, t2 - t1, produced * 1000.0 / (t2 - t1));
    return 0;
}
#endif
