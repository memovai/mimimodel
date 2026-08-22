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
#define cold_malloc(sz) heap_caps_malloc((sz), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
#define aligned_alloc16(sz) \
    heap_caps_aligned_alloc(16, (((sz) + 15) & ~(size_t)15), \
                            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)
#else
#define fast_malloc malloc
#define cold_malloc malloc
static inline void *aligned_alloc16(size_t sz) {
    void *p = NULL;
    if (posix_memalign(&p, 16, (sz + 15) & ~(size_t)15) != 0) return NULL;
    return p;
}

#endif

#define NEEDLE_KV_SLACK 160   /* rows beyond the window: one call's decode */
#define NEEDLE_QUERY_ROOM 512 /* query tokens the prefix cache stays valid for */
#define NEEDLE_DECODE_ROOM 200
#ifndef NEEDLE_PREFIX_SINK_DEFAULT
#define NEEDLE_PREFIX_SINK_DEFAULT 160
#endif
#ifndef NEEDLE_REASON_MAX_DEFAULT
#define NEEDLE_REASON_MAX_DEFAULT 256
#endif
#ifndef NEEDLE_BYTE_GRAMMAR_DEFAULT
#define NEEDLE_BYTE_GRAMMAR_DEFAULT 1
#endif
#ifndef NEEDLE_MHC_OVERLAP_DEFAULT
#define NEEDLE_MHC_OVERLAP_DEFAULT 0
#endif
#ifndef NEEDLE_MT_MIN_ROWS_DEFAULT
#define NEEDLE_MT_MIN_ROWS_DEFAULT 64
#endif
#ifndef NEEDLE_MHC_Q_LEAD_ROWS_DEFAULT
#define NEEDLE_MHC_Q_LEAD_ROWS_DEFAULT 128
#endif
#ifndef NEEDLE_GATE_OVERLAP_DEFAULT
#define NEEDLE_GATE_OVERLAP_DEFAULT 0
#endif
#ifndef NEEDLE_GATE_LEAD_ROWS_DEFAULT
#define NEEDLE_GATE_LEAD_ROWS_DEFAULT 32
#endif
#ifndef NEEDLE_DYNAMIC_WEIGHT_CACHE_DEFAULT
#define NEEDLE_DYNAMIC_WEIGHT_CACHE_DEFAULT 0
#endif
#ifndef NEEDLE_DYNAMIC_CACHE_RESERVE_KB_DEFAULT
#define NEEDLE_DYNAMIC_CACHE_RESERVE_KB_DEFAULT 256
#endif
#ifndef NEEDLE_SINKHORN_ITERS
#define NEEDLE_SINKHORN_ITERS 20
#endif
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
    __attribute__((aligned(16))) float lut2[256][4];
                                      /* byte -> 4 decoded 2-bit values */
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
    uint32_t max_len, kv_alloc;   /* prefix sink followed by a recent-token ring */
    uint32_t sink_len;
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
    float *x, *nx, *u, *h, *q, *k, *v, *att_out, *xh, *xh2, *z, *logits,
          *ek, *ev, *e;
    /* quantized companions of xh / xh2, filled by cq_prepare_x. Indexed by
     * which prep buffer a caller passed (see prep_slot()). */
    int16_t *xq[2];
    float   *xs[2];
    float *cosv, *sinv;           /* rope tables (max_len, head_dim/2) */
    float *attn_scores;            /* overflow buffers when kv_alloc > d_model */
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

/* 2-bit quad-row kernel: four rows share each activation load. */
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

#if defined(ESP_PLATFORM) && defined(__XTENSA__)
extern void needle_cq2_dot2_tie728(float out[2], const uint8_t *row0,
                                    const uint8_t *row1, const float *x,
                                    const float *lut, int nbytes);

static void cq_matvec_2b_tie728(const Needle *m, const CQMat *W,
                                const float *xh, float *y) {
    uint32_t G = W->group, n_groups = W->in_pad / G;
    uint32_t row_bytes = W->in_pad / 4;
    const float *lut = (const float *)m->lut2;
    uint32_t r = 0;
    for (; r + 2 <= W->out; r += 2) {
        const uint8_t *r0 = W->packed + (uint64_t)r * row_bytes;
        const uint8_t *r1 = r0 + row_bytes;
        const uint16_t *n0 = W->norms + (uint64_t)r * n_groups;
        const uint16_t *n1 = n0 + n_groups;
        float a0 = 0.0f, a1 = 0.0f;
        for (uint32_t g = 0; g < n_groups; g++) {
            float s[2];
            uint32_t boff = g * G / 4;
            needle_cq2_dot2_tie728(s, r0 + boff, r1 + boff,
                                    xh + g * G, lut, G / 4);
            a0 += fp16_to_f32(n0[g]) * s[0];
            a1 += fp16_to_f32(n1[g]) * s[1];
        }
        y[r] = a0;
        y[r + 1] = a1;
    }
    if (r < W->out) {
        CQMat tail = *W;
        tail.packed += (uint64_t)r * row_bytes;
        tail.norms += (uint64_t)r * n_groups;
        tail.out = 1;
        cq_matvec_2b(m, &tail, xh, y + r);
    }
}
#endif

static void cq_matvec(const Needle *m, const CQMat *W, const float *xh,
                      float *y) {
    /* The integer path only wins where a SIMD MAC exists: on scalar cores the
     * weight-decode round trip costs more than it saves, and the host's
     * auto-vectorized float loops are faster still. */
#ifdef NEEDLE_HAVE_PIE
    if (W->bits == 2 || W->bits == 4) { cq_matvec_i16(m, W, xh, y); return; }
#endif
    if (W->bits == 2) {
#if defined(ESP_PLATFORM) && defined(__XTENSA__)
        cq_matvec_2b_tie728(m, W, xh, y);
#else
        cq_matvec_2b(m, W, xh, y);
#endif
        return;
    }
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
    if (!g_mv_go || W->out < NEEDLE_MT_MIN_ROWS_DEFAULT) {
        cq_matvec(m, W, xh, y);
        return;
    }
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

typedef struct {
    const Needle *m;
    uint32_t layer, lanes;
    const float *xh;
    float *hpost, *res;
} MhcProjJob;

static inline float sigmoidf_(float x);
static void sinkhorn(float *lg, int n, int iters);

/* phi_post and phi_res depend only on the layer input prepared for phi_pre.
 * Computing them on the worker while core 0 builds the attention input fills
 * a previously idle-core region without changing their arithmetic order. */
static void mhc_proj_job_fn(void *p) {
    MhcProjJob *j = (MhcProjJob *)p;
    CQMat post = cq_rows(&j->m->phi_post, j->layer * j->lanes, j->lanes);
    CQMat res = cq_rows(&j->m->phi_res, j->layer * j->lanes * j->lanes,
                       j->lanes * j->lanes);
    cq_matvec(j->m, &post, j->xh, j->hpost);
    cq_matvec(j->m, &res, j->xh, j->res);
    uint32_t own = j->layer % j->lanes;
    for (uint32_t l = 0; l < j->lanes; l++) {
        float off = (l == own) ? 0.0f : -4.0f;
        j->hpost[l] = 2.0f * sigmoidf_(j->m->a_post[j->layer] * j->hpost[l]
                         + j->m->b_post[j->layer * j->lanes + l] + off);
    }
    for (uint32_t l = 0; l < j->lanes * j->lanes; l++)
        j->res[l] = j->m->a_res[j->layer] * j->res[l]
                  + j->m->b_res[j->layer * j->lanes * j->lanes + l];
    sinkhorn(j->res, (int)j->lanes, NEEDLE_SINKHORN_ITERS);
}

/* Copy a CQ matrix (packed indices + norms are contiguous in the blob) into
 * PSRAM to reduce flash-cache churn. Returns bytes used. */
#ifdef ESP_PLATFORM
#include "esp_heap_caps.h"
#include "esp_partition.h"

/* set by the app so cached weights can be read via the SPI driver (fast DMA
 * path) instead of memcpy through the mmap cache */
const esp_partition_t *g_needle_part = NULL;
const uint8_t *g_needle_mmap_base = NULL;

typedef struct {
    CQMat *matrix;
    const uint8_t *packed;
    const uint16_t *norms;
    uint8_t *copy;
} DynamicCQCache;

#define MAX_DYNAMIC_CQ_CACHE (8 + MAX_LAYERS * 5)
static DynamicCQCache *g_dynamic_cq_cache;
static int g_dynamic_cq_count;

static int cq_points_into_model(const CQMat *W) {
    if (!g_needle_mmap_base || !g_needle_part) return 1;
    uintptr_t p = (uintptr_t)W->packed;
    uintptr_t lo = (uintptr_t)g_needle_mmap_base;
    return p >= lo && p < lo + g_needle_part->size;
}

static size_t cq_cache_psram(CQMat *W, size_t budget, int dynamic) {
    /* A prior cache tier already owns this matrix. */
    if (!cq_points_into_model(W)) return 0;
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
    if (dynamic) {
        if (!g_dynamic_cq_cache
            || g_dynamic_cq_count >= MAX_DYNAMIC_CQ_CACHE) {
            heap_caps_free(buf);
            return 0;
        }
        DynamicCQCache *entry = &g_dynamic_cq_cache[g_dynamic_cq_count++];
        entry->matrix = W;
        entry->packed = W->packed;
        entry->norms = W->norms;
        entry->copy = buf;
    }
    W->packed = buf;
    W->norms = (const uint16_t *)(buf + (size_t)W->out * row_bytes);
    return total;
}

static size_t needle_cache_psram_tier(Needle *m, size_t budget, int dynamic) {
    size_t used = 0, got;
    CQMat *order[8 + MAX_LAYERS * 5];
    int n = 0;
    order[n++] = &m->phi_pre; order[n++] = &m->phi_post; order[n++] = &m->phi_res;
    for (uint32_t s = 0; s < m->n_sites; s++) {
        order[n++] = &m->engrams[s].key_proj;
        order[n++] = &m->engrams[s].value_proj;
    }
    if (dynamic) {
        /* Profile-guided ordering: q/gate/out dominate projection time.  The
         * fixed boot tier keeps the original layer-local order; only the
         * request-sized opportunistic tier prefers the hotter matrices. */
        for (uint32_t i = 0; i < m->n_layers; i++) {
            Layer *L = &m->layers[i];
            order[n++] = &L->q_proj;
            order[n++] = &L->gate_proj;
            order[n++] = &L->out_proj;
        }
        for (uint32_t i = 0; i < m->n_layers; i++) {
            Layer *L = &m->layers[i];
            order[n++] = &L->k_proj;
            order[n++] = &L->v_proj;
        }
    } else {
        for (uint32_t i = 0; i < m->n_layers; i++) {
            Layer *L = &m->layers[i];
            order[n++] = &L->q_proj;
            order[n++] = &L->k_proj;
            order[n++] = &L->v_proj;
            order[n++] = &L->gate_proj;
            order[n++] = &L->out_proj;
        }
    }
    for (int i = 0; i < n; i++) {
        got = cq_cache_psram(order[i], budget - used, dynamic);
        /* A large projection may not fit while a later K/V or mHC matrix does.
         * Keep filling the remaining budget instead of abandoning it. */
        if (!got) continue;
        used += got;
    }
    return used;
}

size_t needle_cache_psram(Needle *m, size_t budget) {
    return needle_cache_psram_tier(m, budget, 0);
}

static void needle_release_dynamic_cache(void) {
    while (g_dynamic_cq_count > 0) {
        DynamicCQCache *entry = &g_dynamic_cq_cache[--g_dynamic_cq_count];
        entry->matrix->packed = entry->packed;
        entry->matrix->norms = entry->norms;
        heap_caps_free(entry->copy);
    }
}

static size_t needle_cache_psram_dynamic(Needle *m, size_t reserve) {
    if (!g_dynamic_cq_cache) {
        g_dynamic_cq_cache = (DynamicCQCache *)heap_caps_calloc(
            MAX_DYNAMIC_CQ_CACHE, sizeof(*g_dynamic_cq_cache), MALLOC_CAP_SPIRAM);
        if (!g_dynamic_cq_cache) return 0;
    }
    size_t free = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    if (free <= reserve) return 0;
    return needle_cache_psram_tier(m, free - reserve, 1);
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
#if defined(ESP_PLATFORM) && NEEDLE_DYNAMIC_WEIGHT_CACHE_DEFAULT
    needle_release_dynamic_cache();
#endif
    m->max_len = max_len;
    uint32_t C = m->d_model, hd = m->head_dim, L = m->n_layers, KV = m->n_kv;
    uint32_t n = m->lanes;
    /* A row is overwritten kv_alloc positions later, so kv_alloc > kv_window
     * keeps every in-window row intact — and leaves the prefix rows valid for
     * the next call as long as the slack covers one call's decode. */
    uint32_t want = m->kv_window
                  ? (m->sink_len ? m->sink_len + m->kv_window
                                 : m->kv_window + NEEDLE_KV_SLACK)
                  : max_len;
    m->kv_alloc = want < max_len ? want : max_len;
    free(m->k_cache); free(m->v_cache); free(m->k_scale); free(m->v_scale);
    free(m->attn_scores);
    free(m->ering); free(m->ering_valid); free(m->hist);
    m->k_cache = (int8_t *)calloc((size_t)L * KV * m->kv_alloc * hd, 1);
    m->v_cache = (int8_t *)calloc((size_t)L * KV * m->kv_alloc * hd, 1);
    m->k_scale = (float *)calloc((size_t)L * KV * m->kv_alloc, 4);
    m->v_scale = (float *)calloc((size_t)L * KV * m->kv_alloc, 4);
    m->attn_scores = NULL;
    if (m->kv_alloc > C) {
        size_t bytes = (size_t)2 * m->kv_alloc * sizeof(float);
        m->attn_scores = (float *)fast_malloc(bytes);
#ifdef ESP_PLATFORM
        if (!m->attn_scores) m->attn_scores = (float *)malloc(bytes);
#endif
    }
    if (!m->k_cache || !m->v_cache || !m->k_scale || !m->v_scale
        || (m->kv_alloc > C && !m->attn_scores)) {
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
        m->h = (float *)fast_malloc(C * 4);
        m->q = (float *)fast_malloc(A * 4);
        m->k = (float *)fast_malloc((size_t)KV * hd * 4);
        m->v = (float *)fast_malloc((size_t)KV * hd * 4);
        m->att_out = (float *)fast_malloc(A * 4);
        m->xh = (float *)aligned_alloc16((size_t)n * C * 4);  /* >= biggest in_pad */
        m->xh2 = (float *)aligned_alloc16((size_t)n * C * 4); /* phi-prepared nx */
        for (int s = 0; s < 2; s++) {
            /* ee.vld.128 needs 16-byte alignment; group offsets are multiples
             * of 256 bytes so aligning the base is enough */
            m->xq[s] = (int16_t *)aligned_alloc16((size_t)n * C * sizeof(int16_t));
            m->xs[s] = (float *)fast_malloc((size_t)n * C / 8 * sizeof(float));
        }
        m->z = (float *)fast_malloc(m->hada_n * 4);
        m->logits = (float *)malloc(m->vocab * 4);         /* cold: PSRAM ok */
        m->ek = (float *)fast_malloc((size_t)m->n_sites * C * 4);
        m->ev = (float *)cold_malloc((size_t)m->n_sites * C * 4);
        m->e = (float *)cold_malloc(C * 4);
    }
    if ((m->n_sites && (!m->ering || !m->ering_valid
                        || !m->px_ering || !m->px_ering_valid))
        || !m->hist || !m->x || !m->nx || !m->u || !m->h || !m->q || !m->k
        || !m->v || !m->att_out || !m->xh || !m->xh2 || !m->xq[0]
        || !m->xq[1] || !m->xs[0] || !m->xs[1] || !m->z || !m->logits
        || !m->ek || !m->ev || !m->e) {
        fprintf(stderr, "needle_reset: runtime scratch allocation failed"
                " x=%p nx=%p u=%p h=%p q=%p k=%p v=%p ao=%p xh=%p xh2=%p"
                " xq=%p/%p xs=%p/%p z=%p logits=%p ek=%p ev=%p e=%p\n",
                (void *)m->x, (void *)m->nx, (void *)m->u, (void *)m->h,
                (void *)m->q, (void *)m->k, (void *)m->v, (void *)m->att_out,
                (void *)m->xh, (void *)m->xh2, (void *)m->xq[0],
                (void *)m->xq[1], (void *)m->xs[0], (void *)m->xs[1],
                (void *)m->z, (void *)m->logits, (void *)m->ek,
                (void *)m->ev, (void *)m->e);
        m->max_len = 0;
        return;
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
enum {
    PROF_EMBED,
    PROF_ENGRAM,
    PROF_MHC_PRE,
    PROF_QKV_PREP,
    PROF_Q,
    PROF_K,
    PROF_V,
    PROF_QKV_POST,
    PROF_ATTN,
    PROF_GATE,
    PROF_OUT,
    PROF_POST_ATTN,
    PROF_HADA,
    PROF_MHC_POST,
    PROF_LOGITS,
    PROF_COUNT
};
static int64_t g_prof[2][PROF_COUNT];
static int64_t g_prof_wall[2];
static int g_prof_mode;
static const char *g_prof_names[PROF_COUNT] = {
    "embedding", "engram", "mhc-pre", "qkv-prep", "q-proj", "k-proj",
    "v-proj", "qkv-post", "attention", "gate-proj",
    "out-proj", "post-attn", "hada-mlp", "mhc-post", "logits"
};
#define PROF_SPLIT(slot) do { \
    int64_t _prof_now = PROF_NOW(); \
    g_prof[g_prof_mode][slot] += _prof_now - _prof_mark; \
    _prof_mark = _prof_now; \
} while (0)
#define PROF_SET_MODE(mode) do { g_prof_mode = (mode); } while (0)
#define PROF_ADD_WALL(mode, us) do { g_prof_wall[mode] += (us); } while (0)
void needle_prof_dump(void) {
    static const char *mode_names[2] = {"prefill", "decode"};
    for (int mode = 0; mode < 2; mode++) {
        int64_t model = 0;
        for (int i = 0; i < PROF_COUNT; i++) model += g_prof[mode][i];
        int64_t wall = g_prof_wall[mode] ? g_prof_wall[mode] : model;
        printf("[prof:%s] model %lld us | wall %lld us | outside %lld us\n",
               mode_names[mode], (long long)model, (long long)wall,
               (long long)(wall > model ? wall - model : 0));
        for (int i = 0; i < PROF_COUNT; i++) {
            if (g_prof[mode][i])
                printf("[prof:%s] %-10s %8lld us (%2d%% wall)\n",
                       mode_names[mode], g_prof_names[i],
                       (long long)g_prof[mode][i],
                       (int)(g_prof[mode][i] * 100 / (wall ? wall : 1)));
            g_prof[mode][i] = 0;
        }
        g_prof_wall[mode] = 0;
    }
}
#else
#define PROF_SPLIT(slot) do {} while (0)
#define PROF_SET_MODE(mode) do { (void)(mode); } while (0)
#define PROF_ADD_WALL(mode, us) do { (void)(mode); (void)(us); } while (0)
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
    const Needle *m;
    uint32_t layer, prefix_n, recent_lo, pos, kv_start, kv_end;
    float *scores;
} AttnJob;

static inline uint32_t kv_slot(const Needle *m, uint32_t pos) {
    if (!m->sink_len) return pos % m->kv_alloc;
    if (pos < m->sink_len) return pos;
    return m->sink_len + (pos - m->sink_len) % m->kv_window;
}

static void attn_job_fn(void *p) {
    AttnJob *j = (AttnJob *)p;
    const Needle *m = j->m;
    uint32_t hd = m->head_dim, KV = m->n_kv, H = m->n_heads, reps = H / KV;
    uint32_t i = j->layer, pos = j->pos;
    float *aw = j->scores;
    float inv_sq = 1.0f / sqrtf((float)hd);
    uint32_t ka = m->kv_alloc;
    for (uint32_t kk = j->kv_start; kk < j->kv_end; kk++) {
        size_t kbase = (((size_t)i * KV + kk) * ka) * hd;
        size_t sbase = ((size_t)i * KV + kk) * ka;
        for (uint32_t r = 0; r < reps; r++) {
            const float *qh = m->q + (kk * reps + r) * hd;
            uint32_t recent_n = pos >= j->recent_lo ? pos + 1 - j->recent_lo : 0;
            uint32_t T = j->prefix_n + recent_n;
            for (uint32_t tt = 0; tt < T; tt++) {
                uint32_t logical = tt < j->prefix_n ? tt
                                                    : j->recent_lo + tt - j->prefix_n;
                uint32_t sl = kv_slot(m, logical);
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
                uint32_t logical = tt < j->prefix_n ? tt
                                                    : j->recent_lo + tt - j->prefix_n;
                uint32_t sl = kv_slot(m, logical);
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
#ifdef NEEDLE_PROF
    int64_t _prof_mark = PROF_NOW();
#endif

    m->hist[m->hist_len++] = token;

    /* x lanes = embedding row * sqrt(C), broadcast */
    cq_row(m, &m->embedding, (uint32_t)token, m->x);
    float sc = sqrtf((float)C);
    for (uint32_t c = 0; c < C; c++) m->x[c] *= sc;
    for (uint32_t l = 1; l < n; l++) memcpy(m->x + l * C, m->x, C * 4);
    PROF_SPLIT(PROF_EMBED);

    if (m->n_sites) engram_step(m);
    PROF_SPLIT(PROF_ENGRAM);
    tr("emb", -1, m->x, (int)(n*C));
    tr("ekv", -1, m->ev, (int)(m->n_sites*C));

    const float *cos_p = m->cosv + (size_t)pos * half;
    const float *sin_p = m->sinv + (size_t)pos * half;

    for (uint32_t i = 0; i < L; i++) {
        Layer *lp = &m->layers[i];
        /* nx = rms_unit(x.flatten) */
        rms_unit(m->x, m->nx, (int)(n * C));

        /* hpre = sigmoid(a_pre*(phi_pre_i @ nx) + b_pre + pre_off) */
        float hpre[MAX_LANES], hpost[MAX_LANES];
        float tmpl[MAX_LANES * MAX_LANES];
        MhcProjJob mhc_job = {m, i, n, m->xh2, hpost, tmpl};
        int mhc_overlap = 0;
        {
            /* phi_pre rows for layer i: rows [i*n, (i+1)*n) of (L*n, n*C) */
            cq_prepare_x(m, &m->phi_pre, m->nx, m->xh2);
#if defined(ESP_PLATFORM) && NEEDLE_MHC_OVERLAP_DEFAULT
            mhc_overlap = par_submit(mhc_proj_job_fn, &mhc_job);
#endif
            CQMat sub = m->phi_pre;
            uint32_t row_bytes = sub.in_pad * sub.bits / 8;
            sub.packed += (uint64_t)i * n * row_bytes;
            sub.norms += (uint64_t)i * n * (sub.in_pad / sub.group);
            sub.out = n;
            cq_matvec(m, &sub, m->xh2, hpre);
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
        PROF_SPLIT(PROF_MHC_PRE);

        /* ---- attention ---- */
        zcrms(bx, lp->norm_in, m->h, (int)C);
        tr("h", (int)i, m->h, (int)C);
        cq_prepare_x(m, &lp->q_proj, m->h, m->xh);
        PROF_SPLIT(PROF_QKV_PREP);
        tr("xh", (int)i, m->xh, (int)lp->q_proj.in_pad);
        static float gate[2048];
        if (mhc_overlap && NEEDLE_MHC_Q_LEAD_ROWS_DEFAULT > 0
                        && NEEDLE_MHC_Q_LEAD_ROWS_DEFAULT < (int)lp->q_proj.out) {
            uint32_t lead = NEEDLE_MHC_Q_LEAD_ROWS_DEFAULT;
            CQMat q_head = cq_rows(&lp->q_proj, 0, lead);
            cq_matvec(m, &q_head, m->xh, m->q);
            par_wait();
            CQMat q_tail = cq_rows(&lp->q_proj, lead, lp->q_proj.out - lead);
            cq_matvec_mt(m, &q_tail, m->xh, m->q + lead);
        } else {
            if (mhc_overlap) par_wait();
            cq_matvec_mt(m, &lp->q_proj, m->xh, m->q);
        }
        PROF_SPLIT(PROF_Q);
        cq_matvec_mt(m, &lp->k_proj, m->xh, m->k);
        PROF_SPLIT(PROF_K);
        cq_matvec_mt(m, &lp->v_proj, m->xh, m->v);
        PROF_SPLIT(PROF_V);
#ifdef ESP_PLATFORM
        MvJob gate_job;
#endif
        int gate_overlap = 0;
#if defined(ESP_PLATFORM) && NEEDLE_GATE_OVERLAP_DEFAULT
        if (NEEDLE_GATE_LEAD_ROWS_DEFAULT > 0
            && NEEDLE_GATE_LEAD_ROWS_DEFAULT < (int)lp->gate_proj.out) {
            gate_job.m = m;
            gate_job.W = cq_rows(&lp->gate_proj, 0,
                                 NEEDLE_GATE_LEAD_ROWS_DEFAULT);
            gate_job.xh = m->xh;
            gate_job.y = gate;
            gate_job.r0 = 0;
            gate_overlap = par_submit(mv_job_fn, &gate_job);
        }
#endif
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
        uint32_t slot = kv_slot(m, pos);
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
        PROF_SPLIT(PROF_QKV_POST);

        uint32_t prefix_n = m->sink_len < pos + 1 ? m->sink_len : pos + 1;
        uint32_t recent_lo = prefix_n;
        if (m->kv_window && pos + 1 > m->kv_window) {
            uint32_t window_lo = pos + 1 - m->kv_window;
            if (window_lo > recent_lo) recent_lo = window_lo;
        }
        uint32_t reps = H / KV;
        /* nx and h are dead until the next layer. Reuse them for the common
         * <= d_model case instead of spending scarce ESP32 internal SRAM on
         * dedicated attention buffers. */
        float *score0 = m->attn_scores ? m->attn_scores : m->nx;
        float *score1 = m->attn_scores ? m->attn_scores + m->kv_alloc : m->h;
        AttnJob aj0 = { m, i, prefix_n, recent_lo, pos, 0, KV / 2, score0 };
        AttnJob aj1 = { m, i, prefix_n, recent_lo, pos, KV / 2, KV,
                        score1 };
        if (gate_overlap) par_wait();
        if (par_submit(attn_job_fn, &aj1)) {
            attn_job_fn(&aj0);
            par_wait();
        } else {
            aj0.kv_end = KV;
            attn_job_fn(&aj0);
        }
        (void)reps;
        PROF_SPLIT(PROF_ATTN);

        /* gate + out_proj (gate_proj input is h, reuse prepared xh) */
        if (gate_overlap) {
            uint32_t lead = NEEDLE_GATE_LEAD_ROWS_DEFAULT;
            CQMat gate_tail = cq_rows(&lp->gate_proj, lead,
                                      lp->gate_proj.out - lead);
            cq_matvec_mt(m, &gate_tail, m->xh, gate + lead);
        } else {
            cq_matvec_mt(m, &lp->gate_proj, m->xh, gate);
        }
        for (uint32_t d = 0; d < A; d++) m->att_out[d] *= sigmoidf_(gate[d]);
        PROF_SPLIT(PROF_GATE);
        tr("attout", (int)i, m->att_out, (int)A);
        static float attn[512];
        cq_apply(m, &lp->out_proj, m->att_out, attn, m->xh);
        PROF_SPLIT(PROF_OUT);
        static float attn_n[512];
        zcrms(attn, lp->post_norm, attn_n, (int)C);
        float gsc = sigmoidf_(lp->attn_gate);
        static float y[512];
        for (uint32_t c = 0; c < C; c++) y[c] = bx[c] + gsc * attn_n[c];
        PROF_SPLIT(PROF_POST_ATTN);

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
        PROF_SPLIT(PROF_HADA);

        /* y -= u */
        for (uint32_t c = 0; c < C; c++) y[c] -= m->u[c];
        tr("y", (int)i, y, (int)C);

        /* hpost + sinkhorn res mix */
        if (!mhc_overlap) mhc_proj_job_fn(&mhc_job);
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
        PROF_SPLIT(PROF_MHC_POST);
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
    cq_apply(m, &m->embedding, xf, m->logits, m->xh);
    PROF_SPLIT(PROF_LOGITS);
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
 * Byte-level grammar-guided greedy decode for the fixed Needle output shape:
 *   [{"name":"<tool>","arguments":{"<param>":<value>,...}}]
 * Every candidate token is accepted only when all of its decoded bytes keep
 * the schema valid. This preserves the model's natural tokenization across
 * structural boundaries instead of teacher-forcing fragments separately. */

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
    int reason_tok, reason_opened;
    int prefix_reused;
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

/* The reference engine applies one byte-level grammar to the entire JSON
 * value. Keep this path separate from the historical segmented decoder so the
 * two can be ablated on identical logits. A token may cross several grammar
 * states (for example `},{"name":"`), which is exactly what segmented
 * teacher-forcing cannot preserve. */
typedef enum {
    BG_ARRAY_OPEN,
    BG_CALL_OPEN,
    BG_NAME_LITERAL,
    BG_TOOL_NAME,
    BG_ARGS_LITERAL,
    BG_PARAM_START,
    BG_PARAM_QUOTE,
    BG_PARAM_NAME,
    BG_PARAM_COLON,
    BG_STRING_OPEN,
    BG_STRING_VALUE,
    BG_NUMBER_VALUE,
    BG_AFTER_VALUE,
    BG_CALL_CLOSE,
    BG_AFTER_CALL,
    BG_DONE,
} ByteGrammarState;

typedef struct {
    ByteGrammarState state;
    NTool *tools;
    int n_tools, max_calls, calls;
    uint32_t emitted_tools;
    int tool, param;
    uint32_t used_params;
    int literal_off;
    int escape;
    char choice[80];
    int choice_len;
    char number[64];
    int number_len;
} ByteGrammar;

static int bg_required_done(const ByteGrammar *g) {
    const NTool *tool = &g->tools[g->tool];
    for (int i = 0; i < tool->n_params; i++)
        if (tool->params[i].required && !(g->used_params & (1u << i))) return 0;
    return 1;
}

static int bg_choice_prefix(const ByteGrammar *g, int tools) {
    if (tools) {
        for (int i = 0; i < g->n_tools; i++) {
            if (g->emitted_tools & (1u << i)) continue;
            if (strncmp(g->tools[i].name, g->choice, (size_t)g->choice_len) == 0)
                return 1;
        }
    } else {
        const NTool *tool = &g->tools[g->tool];
        for (int i = 0; i < tool->n_params; i++) {
            if (g->used_params & (1u << i)) continue;
            if (strncmp(tool->params[i].name, g->choice, (size_t)g->choice_len) == 0)
                return 1;
        }
    }
    return 0;
}

static int bg_finish_choice(ByteGrammar *g, int tools) {
    g->choice[g->choice_len] = 0;
    if (tools) {
        for (int i = 0; i < g->n_tools; i++) {
            if (!(g->emitted_tools & (1u << i))
                && strcmp(g->tools[i].name, g->choice) == 0) {
                g->tool = i;
                g->param = -1;
                g->used_params = 0;
                return 1;
            }
        }
    } else {
        NTool *tool = &g->tools[g->tool];
        for (int i = 0; i < tool->n_params; i++) {
            if (!(g->used_params & (1u << i))
                && strcmp(tool->params[i].name, g->choice) == 0) {
                g->param = i;
                return 1;
            }
        }
    }
    return 0;
}

static int bg_number_done(ByteGrammar *g) {
    if (g->number_len <= 0 || g->number_len >= (int)sizeof g->number) return 0;
    g->number[g->number_len] = 0;
    char *end = NULL;
    (void)strtod(g->number, &end);
    return end && *end == 0;
}

static int bg_consume_byte(ByteGrammar *g, unsigned char byte) {
    static const char name_literal[] = "\"name\":\"";
    static const char args_literal[] = ",\"arguments\":{";
    int again = 1;
    while (again) {
        again = 0;
        switch (g->state) {
        case BG_ARRAY_OPEN:
            if (byte != '[') return 0;
            g->state = BG_CALL_OPEN;
            break;
        case BG_CALL_OPEN:
            if (byte != '{') return 0;
            g->state = BG_NAME_LITERAL;
            g->literal_off = 0;
            break;
        case BG_NAME_LITERAL:
            if (byte != (unsigned char)name_literal[g->literal_off]) return 0;
            if (++g->literal_off == (int)sizeof name_literal - 1) {
                g->state = BG_TOOL_NAME;
                g->choice_len = 0;
            }
            break;
        case BG_TOOL_NAME:
            if (byte == '"') {
                if (!bg_finish_choice(g, 1)) return 0;
                g->state = BG_ARGS_LITERAL;
                g->literal_off = 0;
            } else {
                if (g->choice_len >= (int)sizeof g->choice - 1) return 0;
                g->choice[g->choice_len++] = (char)byte;
                if (!bg_choice_prefix(g, 1)) return 0;
            }
            break;
        case BG_ARGS_LITERAL:
            if (byte != (unsigned char)args_literal[g->literal_off]) return 0;
            if (++g->literal_off == (int)sizeof args_literal - 1)
                g->state = BG_PARAM_START;
            break;
        case BG_PARAM_START:
            if (byte == '}' && bg_required_done(g)) {
                g->state = BG_CALL_CLOSE;
            } else if (byte == '"') {
                g->state = BG_PARAM_NAME;
                g->choice_len = 0;
            } else return 0;
            break;
        case BG_PARAM_QUOTE:
            if (byte != '"') return 0;
            g->state = BG_PARAM_NAME;
            g->choice_len = 0;
            break;
        case BG_PARAM_NAME:
            if (byte == '"') {
                if (!bg_finish_choice(g, 0)) return 0;
                g->state = BG_PARAM_COLON;
            } else {
                if (g->choice_len >= (int)sizeof g->choice - 1) return 0;
                g->choice[g->choice_len++] = (char)byte;
                if (!bg_choice_prefix(g, 0)) return 0;
            }
            break;
        case BG_PARAM_COLON:
            if (byte != ':') return 0;
            if (g->tools[g->tool].params[g->param].is_num) {
                g->state = BG_NUMBER_VALUE;
                g->number_len = 0;
            } else g->state = BG_STRING_OPEN;
            break;
        case BG_STRING_OPEN:
            if (byte != '"') return 0;
            g->state = BG_STRING_VALUE;
            g->escape = 0;
            break;
        case BG_STRING_VALUE:
            if (g->escape) {
                if (!(byte == '"' || byte == '\\' || byte == '/' || byte == 'b'
                      || byte == 'f' || byte == 'n' || byte == 'r' || byte == 't'
                      || byte == 'u'))
                    return 0;
                g->escape = 0;
            } else if (byte == '\\') {
                g->escape = 1;
            } else if (byte == '"') {
                g->used_params |= 1u << g->param;
                g->state = BG_AFTER_VALUE;
            } else if (byte < 0x20) return 0;
            break;
        case BG_NUMBER_VALUE:
            if ((byte >= '0' && byte <= '9') || byte == '-' || byte == '+'
                || byte == '.' || byte == 'e' || byte == 'E') {
                if (g->number_len >= (int)sizeof g->number - 1) return 0;
                g->number[g->number_len++] = (char)byte;
            } else {
                if (!bg_number_done(g)) return 0;
                g->used_params |= 1u << g->param;
                g->state = BG_AFTER_VALUE;
                again = 1;
            }
            break;
        case BG_AFTER_VALUE:
            if (byte == ',') {
                if (g->used_params == (uint32_t)((1u << g->tools[g->tool].n_params) - 1u))
                    return 0;
                g->state = BG_PARAM_QUOTE;
            } else if (byte == '}' && bg_required_done(g)) {
                g->state = BG_CALL_CLOSE;
            } else return 0;
            break;
        case BG_CALL_CLOSE:
            if (byte != '}') return 0;
            g->emitted_tools |= 1u << g->tool;
            g->calls++;
            g->state = BG_AFTER_CALL;
            break;
        case BG_AFTER_CALL:
            if (byte == ']') {
                g->state = BG_DONE;
            } else if (byte == ',' && g->calls < g->max_calls
                       && g->emitted_tools != (uint32_t)((1u << g->n_tools) - 1u)) {
                g->state = BG_CALL_OPEN;
            } else return 0;
            break;
        case BG_DONE:
            return 0;
        }
    }
    return 1;
}

static int bg_consume_piece(ByteGrammar *g, const char *piece, int length) {
    for (int i = 0; i < length; i++)
        if (!bg_consume_byte(g, (unsigned char)piece[i])) return 0;
    return 1;
}

static int dc_byte_grammar(Needle *m, DecCtx *dc, NTool *tools, int n_tools,
                           int max_calls, char *out, size_t outsz) {
    ByteGrammar grammar;
    memset(&grammar, 0, sizeof grammar);
    grammar.state = BG_ARRAY_OPEN;
    grammar.tools = tools;
    grammar.n_tools = n_tools;
    grammar.max_calls = max_calls;
    grammar.tool = grammar.param = -1;
    size_t olen = 0;
    for (int step = 0; step < NEEDLE_DECODE_ROOM; step++) {
        int best = -1, best_len = 0;
        float best_logit = -1e30f;
        ByteGrammar best_grammar;
        char best_piece[64];
        for (uint32_t token = 0; token < m->n_pieces; token++) {
            if (m->types[token] != 0 && m->types[token] != 4) continue;
            char piece[64];
            int length = needle_decode_piece(m, (int)token, piece, sizeof piece);
            if (length <= 0 || olen + (size_t)length >= outsz) continue;
            ByteGrammar candidate = grammar;
            if (!bg_consume_piece(&candidate, piece, length)) continue;
            if (m->logits[token] > best_logit) {
                best = (int)token;
                best_len = length;
                best_logit = m->logits[token];
                best_grammar = candidate;
                memcpy(best_piece, piece, (size_t)length + 1);
            }
        }
        if (best < 0) return -1;
        memcpy(out + olen, best_piece, (size_t)best_len);
        olen += (size_t)best_len;
        grammar = best_grammar;
        if (grammar.state == BG_DONE) {
            out[olen] = 0;
            return (int)olen;
        }
        dc_feed(m, dc, best);
    }
    return -1;
}

/* ------------------------------------------------------- tool retrieval ---
 * Needle attends over a 256-token sliding window. A tools block larger than
 * that pushes most of the tools out of view and selection collapses — measured
 * on google/mobile-actions: 22% tool-name accuracy with a 417-token block of 7
 * tools, 77% with a 191-token block of 3.
 *
 * The official Needle package documents a learned top-5 retrieval head when
 * more than five tools are declared. This engine uses BM25 because it needs no
 * additional forward pass; it reaches 97% recall@3 on this set. Pruning only
 * kicks in when the block would not fit the budget, so small tool sets pass
 * through untouched and keep the prefix cache warm. */

#define NEEDLE_TOOLS_BUDGET 180   /* tokens; leaves room for the query */
#define NEEDLE_RAG_MIN_K    2
#define NEEDLE_MAX_CALLS    4   /* mimiclaw's llm_response_t caps a turn at 4 */

/* Legacy segmented decoder: logit margin the comma must clear before another
 * call is opened. The default continuous byte grammar does not use this.
 *
 * The model is decisive when another call is genuinely wanted (`,{"` beats `]`
 * by ~2.7 nats) and nearly tied when it is not (~0.05 nats), so a threshold
 * separates the two cleanly. Swept on 300 cases: overall accuracy 39.0% at 0,
 * 46.7% at 1.0, 49.3% at 2.0, then 42.0% at 3.0 and 40.7% at 4.0 where it has
 * collapsed back to single-call behaviour. Both buckets improve together —
 * a spurious second call breaks the one-call cases too. */
#define NEEDLE_CONT_MARGIN_DEFAULT 2.0f

static int is_word_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
}
static char lower_c(char c) { return (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c; }

/* Split a JSON array into its top-level object elements. */
static int split_json_array(const char *json, const char **starts, int *lens, int max) {
    const char *p = strchr(json, '[');
    if (!p) return 0;
    p++;
    int n = 0;
    while (*p && n < max) {
        while (*p && *p != '{') { if (*p == ']') return n; p++; }
        if (!*p) break;
        const char *s = p;
        int depth = 0, instr = 0, esc = 0;
        for (; *p; p++) {
            if (esc) { esc = 0; continue; }
            if (*p == '\\') { esc = 1; continue; }
            if (*p == '"') { instr = !instr; continue; }
            if (instr) continue;
            if (*p == '{') depth++;
            else if (*p == '}') { depth--; if (!depth) { p++; break; } }
        }
        starts[n] = s;
        lens[n] = (int)(p - s);
        n++;
    }
    return n;
}

/* A tool's retrieval document is its name plus its description, matching the
 * reference implementation's tool_to_text(). */
static void tool_doc(const char *elem, int len, char *out, size_t outsz) {
    size_t o = 0;
    for (int pass = 0; pass < 2; pass++) {
        const char *key = pass ? "\"description\":\"" : "\"name\":\"";
        const char *q = NULL;
        for (const char *c = elem; c + strlen(key) <= elem + len; c++)
            if (strncmp(c, key, strlen(key)) == 0) { q = c + strlen(key); break; }
        if (!q) continue;
        if (o && o + 1 < outsz) out[o++] = ' ';
        for (; q < elem + len && *q != '"' && o + 1 < outsz; q++) out[o++] = *q;
    }
    out[o] = 0;
}

/* BM25 over a handful of short documents. */
static void bm25_rank(const char *query, char docs[][512], int n, int *order) {
    static char qbuf[2048];
    size_t ql = strlen(query);
    if (ql >= sizeof qbuf) ql = sizeof qbuf - 1;
    for (size_t i = 0; i < ql; i++) qbuf[i] = lower_c(query[i]);
    qbuf[ql] = 0;

    int dlen[NT_MAX_TOOLS];
    double avgdl = 0;
    for (int i = 0; i < n; i++) {
        int c = 0;
        for (const char *p = docs[i]; *p; ) {
            if (is_word_char(*p)) { c++; while (*p && is_word_char(*p)) p++; }
            else p++;
        }
        dlen[i] = c ? c : 1;
        avgdl += dlen[i];
    }
    avgdl /= (n ? n : 1);

    double score[NT_MAX_TOOLS];
    for (int i = 0; i < n; i++) score[i] = 0.0;

    for (const char *qp = qbuf; *qp; ) {
        if (!is_word_char(*qp)) { qp++; continue; }
        const char *ws = qp;
        while (*qp && is_word_char(*qp)) qp++;
        int wl = (int)(qp - ws);
        if (wl < 2) continue;

        int tf[NT_MAX_TOOLS], df = 0;
        for (int i = 0; i < n; i++) {
            int c = 0;
            for (const char *p = docs[i]; *p; ) {
                if (!is_word_char(*p)) { p++; continue; }
                const char *ds = p;
                while (*p && is_word_char(*p)) p++;
                if ((int)(p - ds) == wl) {
                    int same = 1;
                    for (int k = 0; k < wl; k++)
                        if (lower_c(ds[k]) != ws[k]) { same = 0; break; }
                    if (same) c++;
                }
            }
            tf[i] = c;
            if (c) df++;
        }
        if (!df) continue;
        double idf = log(1.0 + ((double)n - df + 0.5) / (df + 0.5));
        const double k1 = 1.5, b = 0.75;
        for (int i = 0; i < n; i++) {
            if (!tf[i]) continue;
            score[i] += idf * tf[i] * (k1 + 1.0)
                      / (tf[i] + k1 * (1.0 - b + b * dlen[i] / avgdl));
        }
    }
    for (int i = 0; i < n; i++) order[i] = i;
    for (int a = 0; a < n; a++)
        for (int b2 = a + 1; b2 < n; b2++)
            if (score[order[b2]] > score[order[a]]) {
                int t = order[a]; order[a] = order[b2]; order[b2] = t;
            }
}

/* Returns 1 and fills `out` when the tool list was pruned, 0 to use it as-is. */
static int prune_tools(Needle *m, const char *query, const char *tools_json,
                       char *out, size_t outsz) {
    int budget = NEEDLE_TOOLS_BUDGET;
    if (getenv("NEEDLE_TOOLS_BUDGET")) budget = atoi(getenv("NEEDLE_TOOLS_BUDGET"));
    if (budget <= 0) return 0;

    static int probe[4096];
    int full = needle_encode(m, tools_json, probe, 4096);
    if (full <= budget) {
        if (getenv("NEEDLE_DEBUG_RETRIEVAL"))
            fprintf(stderr, "[retrieval] full=%d budget=%d keep=all\n", full, budget);
        return 0;                               /* already fits; keep cache warm */
    }

    const char *starts[NT_MAX_TOOLS];
    int lens[NT_MAX_TOOLS];
    int n = split_json_array(tools_json, starts, lens, NT_MAX_TOOLS);
    if (n <= NEEDLE_RAG_MIN_K) return 0;

    static char docs[NT_MAX_TOOLS][512];
    for (int i = 0; i < n; i++) tool_doc(starts[i], lens[i], docs[i], sizeof docs[i]);

    int order[NT_MAX_TOOLS];
    bm25_rank(query, docs, n, order);

    /* take the largest prefix of the ranking that still fits the budget */
    int keep = n;
    for (; keep > NEEDLE_RAG_MIN_K; keep--) {
        size_t o = 0;
        out[o++] = '[';
        for (int i = 0; i < keep; i++) {
            if (i) out[o++] = ',';
            memcpy(out + o, starts[order[i]], lens[order[i]]);
            o += lens[order[i]];
        }
        out[o++] = ']';
        out[o] = 0;
        if ((size_t)needle_encode(m, out, probe, 4096) <= (size_t)budget) break;
    }
    if (keep <= NEEDLE_RAG_MIN_K) {
        size_t o = 0;
        out[o++] = '[';
        for (int i = 0; i < NEEDLE_RAG_MIN_K && i < n; i++) {
            if (i) out[o++] = ',';
            memcpy(out + o, starts[order[i]], lens[order[i]]);
            o += lens[order[i]];
        }
        out[o++] = ']';
        out[o] = 0;
    }
    if (getenv("NEEDLE_DEBUG_RETRIEVAL")) {
        fprintf(stderr, "[retrieval] full=%d budget=%d keep=%d", full, budget, keep);
        for (int i = 0; i < keep; i++) fprintf(stderr, " | %s", docs[order[i]]);
        fprintf(stderr, "\n");
    }
    (void)outsz;
    return 1;
}

/* Append raw model bytes into a JSON string value. Control characters must be
 * escaped or the result is not valid JSON — and a raw newline additionally
 * splits the batch protocol's one-line-per-query contract. */
static size_t json_append(char *out, size_t o, const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '\\')      { out[o++] = '\\'; out[o++] = '\\'; }
        else if (c == '\n')  { out[o++] = '\\'; out[o++] = 'n'; }
        else if (c == '\r')  { out[o++] = '\\'; out[o++] = 'r'; }
        else if (c == '\t')  { out[o++] = '\\'; out[o++] = 't'; }
        else if (c < 0x20)   { continue; }
        else                 { out[o++] = (char)c; }
    }
    return o;
}

static int text_join(char *out, size_t outsz, const char *const *parts, int n_parts) {
    size_t used = 0;
    for (int i = 0; i < n_parts; i++) {
        size_t length = strlen(parts[i]);
        if (length >= outsz - used) return 0;
        memcpy(out + used, parts[i], length);
        used += length;
    }
    out[used] = 0;
    return 1;
}

/* Run one constrained tool call. Returns length of JSON written to out. */
int needle_toolcall_sys(Needle *m, const char *system, const char *query,
                        const char *tools_json, char *out, size_t outsz) {
    static char pruned[8192];
    if (prune_tools(m, query, tools_json, pruned, sizeof pruned)) tools_json = pruned;

    static NTool tools[NT_MAX_TOOLS];
    int n_tools = needle_parse_tools(tools_json, tools, NT_MAX_TOOLS);
    if (n_tools <= 0) return -1;

    /* Split at the </tools> marker: markers are atomic tokens, so the prefix
     * tokenization is always a true prefix of the whole prompt's. */
    static char prefix[8192], rest[8192];
    const char *prefix_parts_with_system[] = {
        "<|im_start|>system\n", system, "<|im_end|>\n<|im_start|>user\n<tools>",
        tools_json, "</tools>"
    };
    const char *prefix_parts[] = {
        "<|im_start|>user\n<tools>", tools_json, "</tools>"
    };
    const char *rest_parts[] = {
        "\n", query, "<|im_end|>\n<|im_start|>assistant\n"
    };
    int prefix_ok = system && system[0]
                  ? text_join(prefix, sizeof prefix, prefix_parts_with_system, 5)
                  : text_join(prefix, sizeof prefix, prefix_parts, 3);
    if (!prefix_ok || !text_join(rest, sizeof rest, rest_parts, 3)) return -1;

    static int pids[4096], rids[1024];
    pids[0] = (int)m->bos_id;
    int n_pids = 1 + needle_encode(m, prefix, pids + 1, 4095);
    int n_rids = encode_raw(m, rest, rids, 1024);
    int n_ids = n_pids + n_rids;
    if (getenv("NEEDLE_DEBUG_IDS")) {
        fprintf(stderr, "[debug] prefix ids:");
        for (int i = 0; i < n_pids; i++) fprintf(stderr, " %d", pids[i]);
        fprintf(stderr, "\n[debug] turn ids:");
        for (int i = 0; i < n_rids; i++) fprintf(stderr, " %d", rids[i]);
        fprintf(stderr, "\n");
    }

    uint32_t th = 2166136261u;
    for (const char *c = tools_json; *c; c++) th = (th ^ (uint8_t)*c) * 16777619u;
    if (system) for (const char *c = system; *c; c++) th = (th ^ (uint8_t)*c) * 16777619u;

    size_t ering_bytes = (size_t)m->n_sites * m->ering_depth * m->d_model * 4;
    size_t evalid_bytes = (size_t)m->n_sites * m->ering_depth;

    DecCtx dc = { NULL, 0 };
    int64_t t_pre = needle_now_us();
    PROF_SET_MODE(0);
    int reused = (m->px_valid && m->px_hash == th && m->px_n == (uint32_t)n_pids
                  && m->max_len >= (uint32_t)(n_ids + NEEDLE_DECODE_ROOM));
    if (getenv("NEEDLE_NO_PREFIX_CACHE")) reused = 0;
    if (reused) {
        /* KV rows for the tools block are still live in the ring */
        m->hist_len = m->px_hist_len;
        m->epos = m->px_epos;
        memcpy(m->ering, m->px_ering, ering_bytes);
        memcpy(m->ering_valid, m->px_ering_valid, evalid_bytes);
    } else {
        /* Size for the tools prefix plus generous query room, not just this
         * query: the KV ring is separately capped at kv_window + slack, so a
         * larger max_len costs only the rope tables (256 B/position) and keeps
         * the prefix cache alive when a later query is longer than the first. */
        const char *sink_mode = getenv("NEEDLE_PREFIX_SINK");
        m->sink_len = (uint32_t)n_pids;
        if (sink_mode && strcmp(sink_mode, "system") == 0) {
            if (system && system[0]) {
                static char system_prefix[4096];
                static int system_ids[1024];
                snprintf(system_prefix, sizeof system_prefix,
                         "<|im_start|>system\n%s<|im_end|>\n", system);
                system_ids[0] = (int)m->bos_id;
                m->sink_len = 1u + (uint32_t)needle_encode(
                    m, system_prefix, system_ids + 1, 1023);
            } else m->sink_len = 0;
        } else if (!sink_mode || strcmp(sink_mode, "1") != 0) {
            int cap = sink_mode ? atoi(sink_mode) : NEEDLE_PREFIX_SINK_DEFAULT;
            if (cap <= 0) m->sink_len = 0;
            else if (m->sink_len > (uint32_t)cap) m->sink_len = (uint32_t)cap;
        }
        needle_reset(m, (uint32_t)(n_pids + NEEDLE_QUERY_ROOM + NEEDLE_DECODE_ROOM));
        if (m->max_len == 0) return -1;
#if defined(ESP_PLATFORM) && NEEDLE_DYNAMIC_WEIGHT_CACHE_DEFAULT
        {
            int64_t cache_start = needle_now_us();
            size_t cached = needle_cache_psram_dynamic(
                m, (size_t)NEEDLE_DYNAMIC_CACHE_RESERVE_KB_DEFAULT * 1024);
            fprintf(stderr, "[needle] dynamic weights cached: %u KB in %lld ms\n",
                    (unsigned)(cached / 1024),
                    (long long)((needle_now_us() - cache_start) / 1000));
        }
#endif
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
    PROF_ADD_WALL(0, t_dec - t_pre);
    PROF_SET_MODE(1);
    g_needle_stats.prefill_tok = reused ? n_rids : n_ids;
    g_needle_stats.prefix_reused = reused;
    g_needle_stats.prefill_us = t_dec - t_pre;

    size_t olen = 0;
    (void)outsz;
    /* free-run the reasoning section until the model opens <tool_call>
     * (capped); the think text is not part of the returned JSON */
    int debug_reason = getenv("NEEDLE_DEBUG_REASON") != NULL;
    int reason_max = NEEDLE_REASON_MAX_DEFAULT;
    const char *reason_env = getenv("NEEDLE_REASON_MAX");
    if (reason_env) reason_max = atoi(reason_env);
    if (reason_max < 1) reason_max = 1;
    if (reason_max > 512) reason_max = 512;
    g_needle_stats.reason_tok = 0;
    g_needle_stats.reason_opened = 0;
    if (getenv("NEEDLE_DEBUG_TOP5")) {
        int top[5];
        for (int rank = 0; rank < 5; rank++) {
            int best = -1;
            for (uint32_t token = 0; token < m->n_pieces; token++) {
                int duplicate = 0;
                for (int prior = 0; prior < rank; prior++)
                    if (top[prior] == (int)token) duplicate = 1;
                if (!duplicate && (best < 0 || dc.logits[token] > dc.logits[best]))
                    best = (int)token;
            }
            top[rank] = best;
        }
        fprintf(stderr, "[debug] top5:");
        for (int rank = 0; rank < 5; rank++)
            fprintf(stderr, " %d:%.4f", top[rank], dc.logits[top[rank]]);
        fprintf(stderr, "\n");
    }
    if (debug_reason) fprintf(stderr, "[reason] ");
    for (int s = 0; s < reason_max; s++) {
        int best = 0;
        for (uint32_t t = 1; t < m->n_pieces; t++)
            if (dc.logits[t] > dc.logits[best]) best = (int)t;
        if (m->types[best] == 3 && strcmp(m->pieces[best], "<tool_call>") == 0) {
            dc_feed(m, &dc, best);
            g_needle_stats.reason_opened = 1;
            break;
        }
        if (best == (int)m->eos_id ||
            (m->types[best] == 3 && strcmp(m->pieces[best], "<|im_end|>") == 0)) {
            /* model refuses to call a tool */
            memcpy(out, "[]", 3);
            if (debug_reason) fprintf(stderr, "\n");
            return 2;
        }
        if (debug_reason) {
            char piece[64];
            needle_decode_piece(m, best, piece, sizeof piece);
            fputs(piece, stderr);
        }
        dc_feed(m, &dc, best);
        g_needle_stats.reason_tok++;
    }
    if (debug_reason) fprintf(stderr, "\n");
    const char *grammar_env = getenv("NEEDLE_BYTE_GRAMMAR");
    int byte_grammar = grammar_env ? atoi(grammar_env) != 0
                                   : NEEDLE_BYTE_GRAMMAR_DEFAULT;
    if (byte_grammar) {
        if (!g_needle_stats.reason_opened) {
            memcpy(out, "[]", 3);
            g_needle_stats.decode_tok = (int)(dc.pos - (uint32_t)n_ids);
            g_needle_stats.decode_us = needle_now_us() - t_dec;
            PROF_ADD_WALL(1, g_needle_stats.decode_us);
            return 2;
        }
        int max_calls = NEEDLE_MAX_CALLS;
        if (getenv("NEEDLE_MAX_CALLS")) max_calls = atoi(getenv("NEEDLE_MAX_CALLS"));
        int length = dc_byte_grammar(m, &dc, tools, n_tools, max_calls, out, outsz);
        g_needle_stats.decode_tok = (int)(dc.pos - (uint32_t)n_ids);
        g_needle_stats.decode_us = needle_now_us() - t_dec;
        PROF_ADD_WALL(1, g_needle_stats.decode_us);
        return length;
    }
    /* The dataset's answers are often two calls (33% of google/mobile-actions),
     * so emit an array and let the model decide after each closing brace whether
     * to open another call or finish. The structural bytes are fed through
     * dc_force rather than memcpy'd so the model actually sees them before it
     * makes that choice. */
    int emitted_calls = 0;
    int emitted[NT_MAX_TOOLS];
    for (int i = 0; i < n_tools; i++) emitted[i] = 0;
    int max_calls = NEEDLE_MAX_CALLS;
    if (getenv("NEEDLE_MAX_CALLS")) max_calls = atoi(getenv("NEEDLE_MAX_CALLS"));
    for (int call = 0; call < max_calls; call++) {
        /* Work out whether this call can happen BEFORE emitting anything for
         * it. Bailing out after the opening was already forced leaves a
         * dangling `,{"name":"` and the whole array fails to parse — that was
         * 172 of the 320 two-call cases scoring zero.
         *
         * A tool already used this turn is not a candidate again: no ground
         * truth in google/mobile-actions repeats one, and without this the
         * model happily emits the same call four times. Retrieval usually
         * leaves only 2-3 tools, so this is what ends the loop in practice. */
        const char *names[NT_MAX_TOOLS];
        int nmap[NT_MAX_TOOLS], n_cand = 0;
        for (int i = 0; i < n_tools; i++)
            if (!emitted[i]) { names[n_cand] = tools[i].name; nmap[n_cand++] = i; }
        if (n_cand == 0) break;

        /* Force the opening as ONE segment. dc_force masks logits per segment,
         * so splitting "[{\"name\":\"" into "[" + "{\"name\":\"" changes which
         * tokens the model walks through and measurably degrades the first
         * call (1-call exact 61.1% -> 50.8% on the full eval when split). */
        dc_force(m, &dc, call ? ",{\"name\":\"" : "[{\"name\":\"", out, &olen);
        /* Greedy prefix-constrained walk beats scoring each candidate by mean token
         * logprob (measured: 64% vs 58% tool-name accuracy on mobile-actions). The
         * scored variant also teacher-forces ~21 tokens through engram_step while
         * the engram ring is only 10 deep, so it destroys ring history that the
         * real decode still needs — rewinding epos does not restore the contents. */
        int ti = getenv("NEEDLE_NAME_SCORED")
               ? dc_pick_scored(m, &dc, names, n_cand, out, &olen)
               : dc_trie(m, &dc, names, n_cand, out, &olen);
        if (ti < 0) return -1;
        emitted[nmap[ti]] = 1;
        NTool *T = &tools[nmap[ti]];

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
                size_t val_start = olen;
                for (int s = 0; s < 48; s++) {
                    int best = 0;
                    for (uint32_t t = 1; t < m->n_pieces; t++)
                        if (dc.logits[t] > dc.logits[best]) best = (int)t;
                    if (s == 0 && !getenv("NEEDLE_NO_VALMASK")) {
                        /* At the first token of a value the model often prefers to
                         * close out (`,"` or `}`) rather than fill a key we forced
                         * on it — the intended content sits a few nats lower. Taking
                         * the close literally writes an empty value, which is half of
                         * all argument errors on mobile-actions. Skip tokens that are
                         * pure JSON structure and let the best real content win. */
                        int alt = -1;
                        for (uint32_t t = 1; t < m->n_pieces; t++) {
                            if (m->types[t] == 1 || m->types[t] == 2) continue;
                            int pl = m->piece_len[t];
                            if (pl <= 0) continue;
                            int structural = 1;
                            for (int k2 = 0; k2 < pl; k2++) {
                                char c2 = m->pieces[t][k2];
                                if (!(c2 == ',' || c2 == '"' || c2 == '}' || c2 == ']' ||
                                      c2 == '{' || c2 == ':' || c2 == ' ')) { structural = 0; break; }
                            }
                            if (structural) continue;
                            if (alt < 0 || dc.logits[t] > dc.logits[alt]) alt = (int)t;
                        }
                        if (alt >= 0) best = alt;   /* fall back to argmax if nothing survives */
                    }
                    if (s == 0 && getenv("NEEDLE_DEBUG_VAL")) {
                        int top[5];
                        for (int a = 0; a < 5; a++) {
                            int b2 = -1;
                            for (uint32_t t = 1; t < m->n_pieces; t++) {
                                int dup = 0;
                                for (int c2 = 0; c2 < a; c2++) if (top[c2] == (int)t) dup = 1;
                                if (dup) continue;
                                if (b2 < 0 || dc.logits[t] > dc.logits[b2]) b2 = (int)t;
                            }
                            top[a] = b2;
                        }
                        fprintf(stderr, "[val-start %s]", P->name);
                        for (int a = 0; a < 5; a++) {
                            char pc[64];
                            needle_decode_piece(m, top[a], pc, sizeof pc);
                            fprintf(stderr, "  %s(%.2f)", pc[0] ? pc : "<ctl>", dc.logits[top[a]]);
                        }
                        fprintf(stderr, "\n");
                    }
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
                        olen = json_append(out, olen, piece, (size_t)(qm - piece));
                        dc_feed(m, &dc, best);
                        quote_done = 1;
                        if (getenv("NEEDLE_DEBUG"))
                            fprintf(stderr, "[val-end] piece=%s\n", piece);
                        spill_comma = strchr(qm + 1, ',') != NULL;
                        spill_quote = strchr(qm + 1, '"') != NULL;
                        spill_close = strchr(qm + 1, '}') != NULL;
                        break;
                    }
                    olen = json_append(out, olen, piece, strlen(piece));
                    dc_feed(m, &dc, best);
                }
                /* a trailing comma is the model's field separator leaking into
                 * the value, never part of the intended string */
                while (olen > val_start && (out[olen-1] == ',' || out[olen-1] == ' '))
                    olen--;
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
        dc_force(m, &dc, "}}", out, &olen);   /* close arguments, close the call */
        emitted_calls++;
        /* another call, or done? compare the model's own preference */
        {
            if (getenv("NEEDLE_DEBUG_STOP")) {
                int top[6];
                for (int a = 0; a < 6; a++) {
                    int b2 = -1;
                    for (uint32_t t = 1; t < m->n_pieces; t++) {
                        int dup = 0;
                        for (int c2 = 0; c2 < a; c2++) if (top[c2] == (int)t) dup = 1;
                        if (dup) continue;
                        if (b2 < 0 || dc.logits[t] > dc.logits[b2]) b2 = (int)t;
                    }
                    top[a] = b2;
                }
                fprintf(stderr, "[stop? after call %d]", call + 1);
                for (int a = 0; a < 6; a++) {
                    char pc[64];
                    needle_decode_piece(m, top[a], pc, sizeof pc);
                    fprintf(stderr, "  %s(%.2f)", pc[0] ? pc : "<mk>", dc.logits[top[a]]);
                }
                fprintf(stderr, "\n");
            }
            float lc = -1e30f, lb = -1e30f;
            for (uint32_t t = 0; t < m->n_pieces; t++) {
                if (m->types[t] != 0 || m->piece_len[t] == 0) continue;
                char c0 = m->pieces[t][0];
                if (c0 == ',' && dc.logits[t] > lc) lc = dc.logits[t];
                if (c0 == ']' && dc.logits[t] > lb) lb = dc.logits[t];
            }
            /* Continuing costs a whole spurious call when the model is only
             * mildly in favour, and a spurious call fails ordered strict match
             * outright. Require the comma to win by a margin. */
            float margin = NEEDLE_CONT_MARGIN_DEFAULT;
            const char *mv = getenv("NEEDLE_CONT_MARGIN");
            if (mv) margin = (float)atof(mv);
            if (lc < lb + margin) break;
        }
    }
    (void)emitted_calls;
    out[olen++] = ']';
    out[olen] = 0;
    g_needle_stats.decode_tok = (int)(dc.pos - (uint32_t)n_ids);
    g_needle_stats.decode_us = needle_now_us() - t_dec;
    PROF_ADD_WALL(1, g_needle_stats.decode_us);
    return (int)olen;
}

int needle_toolcall(Needle *m, const char *query, const char *tools_json,
                    char *out, size_t outsz) {
    return needle_toolcall_sys(m, NULL, query, tools_json, out, outsz);
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

    /* Batch mode: query argument of the form @path reads one query per line and
     * emits one JSON result per line. Input is system<TAB>query with an optional
     * third tools-JSON field. 0x1e encodes a newline inside either text field. */
    if (query[0] == '@') {
        FILE *qf = strcmp(query + 1, "-") == 0 ? stdin : fopen(query + 1, "r");
        if (!qf) { perror("open query file"); return 1; }
        static char line[4096], callbuf[4096];
        int n = 0;
        double t_start = now_ms();
        while (fgets(line, sizeof line, qf)) {
            size_t ln = strlen(line);
            while (ln && (line[ln-1] == '\n' || line[ln-1] == '\r')) line[--ln] = 0;
            if (ln == 0) continue;
            /* A bare line has no system turn. Per-row tools preserve benchmark
             * tool ordering instead of silently reusing the first case's order. */
            char *sys_part = NULL, *q_part = line;
            const char *row_tools = tools;
            char *tabp = strchr(line, '\t');
            if (tabp) {
                *tabp = 0; sys_part = line; q_part = tabp + 1;
                char *tab2 = strchr(q_part, '\t');
                if (tab2) { *tab2 = 0; row_tools = tab2 + 1; }
            }
            for (char *p = sys_part; p && *p; p++) if ((uint8_t)*p == 0x1e) *p = '\n';
            for (char *p = q_part; *p; p++) if ((uint8_t)*p == 0x1e) *p = '\n';
            double t0 = now_ms();
            int cl = needle_toolcall_sys(&m, sys_part, q_part, row_tools, callbuf, sizeof callbuf);
            double dt = now_ms() - t0;
            printf("%s\t%.0f\t%d\t%d\t%lld\t%lld\t%d\t%d\n",
                   cl >= 0 ? callbuf : "[]", dt,
                   g_needle_stats.prefill_tok, g_needle_stats.decode_tok,
                   (long long)g_needle_stats.prefill_us,
                   (long long)g_needle_stats.decode_us,
                   g_needle_stats.reason_tok, g_needle_stats.reason_opened);
            fflush(stdout);
            n++;
        }
        if (qf != stdin) fclose(qf);
        fprintf(stderr, "batch: %d queries in %.1f s (%.2f s/query)\n",
                n, (now_ms() - t_start) / 1000.0, (now_ms() - t_start) / 1000.0 / (n ? n : 1));
        return 0;
    }

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
