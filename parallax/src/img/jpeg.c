/* JPEG decoder (ITU T.81). Every scan is entropy-decoded into a per-component coefficient array;
 * after EOI the blocks are dequantised, inverse transformed, upsampled and colour converted.
 * Holding all coefficients costs 2 bytes per sample but lets baseline, multi-scan sequential and
 * progressive files share one output path, which is what keeps them all bit-exact with libjpeg. */
#include "img/jpeg.h"

#include <stdlib.h>
#include <string.h>

/* Zig-zag index -> natural (row-major) index. The 16 trailing entries absorb run lengths that
 * overshoot coefficient 63 in corrupt data, exactly as libjpeg's padded table does, so such
 * files neither write out of bounds nor decode differently from libjpeg. */
static const uint8_t k_natural[64 + 16] = {
    0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,  12, 19, 26, 33, 40, 48,
    41, 34, 27, 20, 13, 6,  7,  14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23,
    30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63, 63, 63,
    63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63};

/* ---- Annex K.3 tables, used when a scan references table 0/1 that was never defined ---- */
static const uint8_t k_dc_lum_bits[16] = {0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
static const uint8_t k_dc_chr_bits[16] = {0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
static const uint8_t k_dc_vals[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
static const uint8_t k_ac_lum_bits[16] = {0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d};
static const uint8_t k_ac_lum_vals[162] = {
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61,
    0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52,
    0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25,
    0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45,
    0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64,
    0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83,
    0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99,
    0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
    0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3,
    0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8,
    0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa};
static const uint8_t k_ac_chr_bits[16] = {0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77};
static const uint8_t k_ac_chr_vals[162] = {
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61,
    0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33,
    0x52, 0xf0, 0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18,
    0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44,
    0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63,
    0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a,
    0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
    0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
    0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca,
    0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7,
    0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa};

/* ---- Huffman tables ------------------------------------------------------------------- */
#define FAST_BITS 9

typedef struct {
    uint8_t fast_len[1 << FAST_BITS]; /* 0: code longer than FAST_BITS, take the slow path */
    uint8_t fast_sym[1 << FAST_BITS];
    int32_t maxcode[17];              /* largest code of each length (mincode - 1 if none) */
    int32_t delta[17];                /* index into vals = code + delta[len] */
    uint8_t vals[256];
    int     nvals;
    bool    defined;
} huff;

static int huff_build(huff *t, const uint8_t counts[16], const uint8_t *vals) {
    int n = 0;
    for (int l = 0; l < 16; l++) n += counts[l];
    if (n > 256) return PLX_EFORMAT;
    memset(t, 0, sizeof *t);
    memcpy(t->vals, vals, (size_t)n);
    t->nvals = n;
    int32_t code = 0;
    int k = 0;
    for (int l = 1; l <= 16; l++) {
        int cnt = counts[l - 1];
        t->delta[l] = k - code;
        for (int i = 0; i < cnt; i++, k++, code++) {
            if (code >= (1 << l)) return PLX_EFORMAT; /* over-subscribed: not a prefix code */
            if (l <= FAST_BITS) {
                int shift = FAST_BITS - l;
                for (int f = 0; f < (1 << shift); f++) {
                    t->fast_len[(code << shift) | f] = (uint8_t)l;
                    t->fast_sym[(code << shift) | f] = vals[k];
                }
            }
        }
        t->maxcode[l] = code - 1;
        code <<= 1;
    }
    t->defined = true;
    return PLX_OK;
}

/* ---- entropy-coded segment bit reader ---------------------------------------------------
 * MSB-first 64-bit accumulator. At a marker or end of data it feeds zero bits like libjpeg,
 * but counts them in `fake`: consuming any of them (nbits < fake) means the scan was truncated. */
typedef struct {
    const uint8_t *p, *end;
    uint64_t acc;
    int nbits, fake;
    bool at_marker;
} bitrd;

static void bits_fill(bitrd *b) {
    while (b->nbits <= 56) {
        int byte = -1;
        if (!b->at_marker && b->p < b->end) {
            if (*b->p != 0xFF) {
                byte = *b->p++;
            } else {
                const uint8_t *q = b->p + 1;
                while (q < b->end && *q == 0xFF) q++; /* fill bytes: FF FF .. 00 is one stuffed FF */
                if (q < b->end && *q == 0) {
                    byte = 0xFF;
                    b->p = q + 1;
                } else {
                    b->at_marker = true; /* p stays on the marker for the caller */
                }
            }
        }
        if (byte < 0) {
            byte = 0;
            b->fake += 8;
        }
        b->acc |= (uint64_t)byte << (56 - b->nbits);
        b->nbits += 8;
    }
}

/* Every symbol needs at most 16 code bits + 16 value bits, so 32 buffered bits always suffice. */
#define REFILL(b) do { if ((b)->nbits < 32) bits_fill(b); } while (0)

static inline int get_bits(bitrd *b, int n) {
    if (n == 0) return 0;
    int v = (int)(b->acc >> (64 - n));
    b->acc <<= n;
    b->nbits -= n;
    return v;
}

static inline int extend(int v, int s) { return v < (1 << (s - 1)) ? v - (1 << s) + 1 : v; }

static inline int16_t wrap16(int v) { return (int16_t)(uint16_t)v; }

static inline int huff_decode(bitrd *b, const huff *t) {
    unsigned look = (unsigned)(b->acc >> (64 - FAST_BITS));
    int len = t->fast_len[look];
    if (len) {
        b->acc <<= len;
        b->nbits -= len;
        return t->fast_sym[look];
    }
    uint32_t code16 = (uint32_t)(b->acc >> 48);
    for (int l = FAST_BITS + 1; l <= 16; l++) {
        int32_t code = (int32_t)(code16 >> (16 - l));
        if (code <= t->maxcode[l]) {
            int idx = code + t->delta[l];
            if (idx < 0 || idx >= t->nvals) return -1;
            b->acc <<= l;
            b->nbits -= l;
            return t->vals[idx];
        }
    }
    return -1;
}

/* ---- decoder state ---------------------------------------------------------------------- */
typedef struct {
    int id, h, v, tq;
    int cw, ch;          /* sample dimensions: ceil(image * factor / max factor) */
    int bw, bh;          /* allocated blocks: the full interleaved-MCU grid */
    int16_t *coef;       /* bw * bh blocks of 64 coefficients, natural order */
    uint16_t q[64];      /* quant table latched at the component's first scan, natural order */
    bool scanned;
    int pred;            /* DC predictor, kept modulo 2^16 like the int16 coefficients it feeds */
    const huff *dct, *act;
} jcomp;

typedef struct {
    const uint8_t *data;
    size_t len;
    int w, h, nc, hmax, vmax, mcux, mcuy;
    bool frame, progressive, jfif, adobe;
    int adobe_transform, restart_interval;
    bool qdef[4];
    uint16_t qt[4][64];
    huff dc[4], ac[4];
    jcomp c[3];
    bitrd br;
    int eobrun, ss, se, ah, al;
} jdec;

typedef int (*block_fn)(jdec *d, jcomp *c, int16_t *blk);

static int blk_sequential(jdec *d, jcomp *c, int16_t *blk) {
    bitrd *b = &d->br;
    memset(blk, 0, 64 * sizeof *blk);
    REFILL(b);
    int s = huff_decode(b, c->dct);
    if (s < 0 || s > 15) return PLX_EFORMAT;
    c->pred = wrap16(c->pred + (s ? extend(get_bits(b, s), s) : 0));
    blk[0] = (int16_t)c->pred;
    for (int k = 1; k < 64; k++) {
        REFILL(b);
        int rs = huff_decode(b, c->act);
        if (rs < 0) return PLX_EFORMAT;
        int r = rs >> 4, sz = rs & 15;
        if (sz) {
            k += r;
            blk[k_natural[k]] = (int16_t)extend(get_bits(b, sz), sz);
        } else {
            if (r != 15) break;
            k += 15;
        }
    }
    return PLX_OK;
}

static int blk_dc_first(jdec *d, jcomp *c, int16_t *blk) {
    bitrd *b = &d->br;
    REFILL(b);
    int s = huff_decode(b, c->dct);
    if (s < 0 || s > 15) return PLX_EFORMAT;
    c->pred = wrap16(c->pred + (s ? extend(get_bits(b, s), s) : 0));
    blk[0] = wrap16(c->pred * (1 << d->al));
    return PLX_OK;
}

static int blk_dc_refine(jdec *d, jcomp *c, int16_t *blk) {
    REFILL(&d->br);
    if (get_bits(&d->br, 1)) blk[0] = (int16_t)(blk[0] | (1 << d->al));
    return PLX_OK;
}

static int blk_ac_first(jdec *d, jcomp *c, int16_t *blk) {
    if (d->eobrun > 0) {
        d->eobrun--;
        return PLX_OK;
    }
    bitrd *b = &d->br;
    for (int k = d->ss; k <= d->se; k++) {
        REFILL(b);
        int rs = huff_decode(b, c->act);
        if (rs < 0) return PLX_EFORMAT;
        int r = rs >> 4, s = rs & 15;
        if (s) {
            k += r;
            blk[k_natural[k]] = wrap16(extend(get_bits(b, s), s) * (1 << d->al));
        } else if (r == 15) {
            k += 15;
        } else {
            d->eobrun = (1 << r) + get_bits(b, r) - 1;
            break;
        }
    }
    return PLX_OK;
}

/* One correction bit for an already-nonzero coefficient: it moves away from zero by p1. */
static inline void refine_coef(bitrd *b, int16_t *coef, int p1) {
    REFILL(b);
    if (get_bits(b, 1) && (*coef & p1) == 0) *coef = wrap16(*coef >= 0 ? *coef + p1 : *coef - p1);
}

/* Successive-approximation AC refinement (G.1.2.3), following libjpeg's control flow so that
 * run lengths overshooting Se land exactly where libjpeg puts them. */
static int blk_ac_refine(jdec *d, jcomp *c, int16_t *blk) {
    bitrd *b = &d->br;
    int p1 = 1 << d->al;
    int k = d->ss;
    if (d->eobrun == 0) {
        for (; k <= d->se; k++) {
            REFILL(b);
            int rs = huff_decode(b, c->act);
            if (rs < 0) return PLX_EFORMAT;
            int r = rs >> 4, s = rs & 15, val = 0;
            if (s) {
                val = get_bits(b, 1) ? p1 : -p1; /* s must be 1; libjpeg treats any s as 1 */
            } else if (r != 15) {
                d->eobrun = (1 << r) + get_bits(b, r);
                break;
            }
            do {
                int16_t *coef = blk + k_natural[k];
                if (*coef != 0) refine_coef(b, coef, p1);
                else if (--r < 0) break;
                k++;
            } while (k <= d->se);
            if (val) blk[k_natural[k]] = (int16_t)val;
        }
    }
    if (d->eobrun > 0) {
        for (; k <= d->se; k++) {
            int16_t *coef = blk + k_natural[k];
            if (*coef != 0) refine_coef(b, coef, p1);
        }
        d->eobrun--;
    }
    return PLX_OK;
}

/* Returns the next marker code with *pos just past it, or -1 at end of data. Bytes before the
 * marker are skipped, as libjpeg does with a warning, which makes stray padding harmless. */
static int next_marker(const uint8_t *data, size_t len, size_t *pos) {
    size_t p = *pos;
    for (;;) {
        while (p < len && data[p] != 0xFF) p++;
        while (p < len && data[p] == 0xFF) p++;
        if (p >= len) {
            *pos = len;
            return -1;
        }
        int m = data[p++];
        if (m != 0) {
            *pos = p;
            return m;
        }
    }
}

static void bits_reset(jdec *d, size_t pos) {
    memset(&d->br, 0, sizeof d->br);
    d->br.p = d->data + pos;
    d->br.end = d->data + d->len;
}

static const huff *pick_table(jdec *d, bool dc, int idx) {
    huff *t = dc ? &d->dc[idx] : &d->ac[idx];
    if (!t->defined && idx < 2) {
        if (dc) huff_build(t, idx ? k_dc_chr_bits : k_dc_lum_bits, k_dc_vals);
        else huff_build(t, idx ? k_ac_chr_bits : k_ac_lum_bits, idx ? k_ac_chr_vals : k_ac_lum_vals);
    }
    return t->defined ? t : NULL;
}

static int parse_sof(jdec *d, const uint8_t *s, size_t n, bool progressive) {
    if (d->frame || n < 6) return PLX_EFORMAT;
    int prec = s[0], h = s[1] << 8 | s[2], w = s[3] << 8 | s[4], nc = s[5];
    if (n != 6 + 3 * (size_t)nc || nc == 0) return PLX_EFORMAT;
    if (prec != 8 || (nc != 1 && nc != 3)) return PLX_EUNSUPPORTED;
    if (w == 0) return PLX_EFORMAT;
    if (h == 0 || (int64_t)w * h > PLX_JPEG_MAX_PIXELS) return PLX_EUNSUPPORTED;
    d->w = w;
    d->h = h;
    d->nc = nc;
    d->progressive = progressive;
    d->hmax = d->vmax = 1;
    for (int i = 0; i < nc; i++) {
        jcomp *c = &d->c[i];
        c->id = s[6 + 3 * i];
        c->h = s[7 + 3 * i] >> 4;
        c->v = s[7 + 3 * i] & 15;
        c->tq = s[8 + 3 * i];
        if (c->h < 1 || c->h > 4 || c->v < 1 || c->v > 4 || c->tq > 3) return PLX_EFORMAT;
        for (int j = 0; j < i; j++)
            if (d->c[j].id == c->id) return PLX_EFORMAT;
        d->hmax = PLX_MAX(d->hmax, c->h);
        d->vmax = PLX_MAX(d->vmax, c->v);
    }
    d->mcux = (w + 8 * d->hmax - 1) / (8 * d->hmax);
    d->mcuy = (h + 8 * d->vmax - 1) / (8 * d->vmax);
    for (int i = 0; i < nc; i++) {
        jcomp *c = &d->c[i];
        c->cw = (int)(((int64_t)w * c->h + d->hmax - 1) / d->hmax);
        c->ch = (int)(((int64_t)h * c->v + d->vmax - 1) / d->vmax);
        c->bw = d->mcux * c->h;
        c->bh = d->mcuy * c->v;
        c->coef = PLX_NEW(int16_t, (size_t)c->bw * c->bh * 64);
    }
    d->frame = true;
    return PLX_OK;
}

static int parse_dqt(jdec *d, const uint8_t *s, size_t n) {
    while (n > 0) {
        int pq = s[0] >> 4, tq = s[0] & 15;
        size_t need = 1 + (pq ? 128 : 64);
        if (pq > 1 || tq > 3 || n < need) return PLX_EFORMAT;
        for (int k = 0; k < 64; k++)
            d->qt[tq][k_natural[k]] = pq ? (uint16_t)(s[1 + 2 * k] << 8 | s[2 + 2 * k]) : s[1 + k];
        d->qdef[tq] = true;
        s += need;
        n -= need;
    }
    return PLX_OK;
}

static int parse_dht(jdec *d, const uint8_t *s, size_t n) {
    while (n > 0) {
        if (n < 17) return PLX_EFORMAT;
        int tc = s[0] >> 4, th = s[0] & 15;
        if (tc > 1 || th > 3) return PLX_EFORMAT;
        size_t cnt = 0;
        for (int l = 0; l < 16; l++) cnt += s[1 + l];
        if (n < 17 + cnt) return PLX_EFORMAT;
        int st = huff_build(tc ? &d->ac[th] : &d->dc[th], s + 1, s + 17);
        if (st < 0) return st;
        s += 17 + cnt;
        n -= 17 + cnt;
    }
    return PLX_OK;
}

/* Expect RSTn at the end of a restart interval. Unconsumed bits are padding and are dropped. */
static int restart(jdec *d, int *next_rst) {
    size_t pos = (size_t)(d->br.p - d->data);
    if (next_marker(d->data, d->len, &pos) != 0xD0 + *next_rst) return PLX_EFORMAT;
    *next_rst = (*next_rst + 1) & 7;
    bits_reset(d, pos);
    for (int i = 0; i < d->nc; i++) d->c[i].pred = 0;
    d->eobrun = 0;
    return PLX_OK;
}

/* Parses an SOS header at s and decodes the scan data that follows; *pos ends at the byte
 * where the search for the next marker should resume. */
static int decode_scan(jdec *d, const uint8_t *s, size_t n, size_t *pos) {
    if (!d->frame || n < 1) return PLX_EFORMAT;
    int ns = s[0];
    if (ns < 1 || ns > d->nc || n != 4 + 2 * (size_t)ns) return PLX_EFORMAT;
    jcomp *sc[4];
    int blocks = 0;
    for (int i = 0; i < ns; i++) {
        int id = s[1 + 2 * i], td = s[2 + 2 * i] >> 4, ta = s[2 + 2 * i] & 15;
        sc[i] = NULL;
        for (int j = 0; j < d->nc; j++)
            if (d->c[j].id == id) sc[i] = &d->c[j];
        if (!sc[i] || td > 3 || ta > 3) return PLX_EFORMAT;
        for (int j = 0; j < i; j++)
            if (sc[j] == sc[i]) return PLX_EFORMAT;
        sc[i]->dct = pick_table(d, true, td);
        sc[i]->act = pick_table(d, false, ta);
        blocks += sc[i]->h * sc[i]->v;
    }
    if (ns > 1 && blocks > 10) return PLX_EFORMAT;
    d->ss = s[1 + 2 * ns];
    d->se = s[2 + 2 * ns];
    d->ah = s[3 + 2 * ns] >> 4;
    d->al = s[3 + 2 * ns] & 15;

    block_fn fn;
    bool need_dc = false, need_ac = false;
    if (!d->progressive) {
        fn = blk_sequential; /* Ss/Se/Ah/Al are meaningless here; libjpeg only warns */
        need_dc = need_ac = true;
    } else {
        bool dc_band = d->ss == 0;
        if (dc_band ? d->se != 0 : (d->ss > d->se || d->se > 63 || ns != 1)) return PLX_EFORMAT;
        if ((d->ah != 0 && d->al != d->ah - 1) || d->al > 13) return PLX_EFORMAT;
        if (dc_band) fn = d->ah ? blk_dc_refine : blk_dc_first;
        else fn = d->ah ? blk_ac_refine : blk_ac_first;
        need_dc = dc_band && d->ah == 0;
        need_ac = !dc_band;
    }
    for (int i = 0; i < ns; i++) {
        jcomp *c = sc[i];
        if ((need_dc && !c->dct) || (need_ac && !c->act)) return PLX_EFORMAT;
        if (!c->scanned) {
            if (!d->qdef[c->tq]) return PLX_EFORMAT;
            memcpy(c->q, d->qt[c->tq], sizeof c->q);
            c->scanned = true;
        }
        c->pred = 0;
    }

    /* A single-component scan is non-interleaved: one block per MCU, covering only the blocks
     * that hold real samples rather than the padded interleaved grid. */
    int mx_n = ns == 1 ? (sc[0]->cw + 7) / 8 : d->mcux;
    int my_n = ns == 1 ? (sc[0]->ch + 7) / 8 : d->mcuy;
    bits_reset(d, *pos);
    d->eobrun = 0;
    int ri = d->restart_interval, todo = ri, next_rst = 0;
    for (int my = 0; my < my_n; my++) {
        for (int mx = 0; mx < mx_n; mx++) {
            if (ri) {
                if (todo == 0) {
                    int st = restart(d, &next_rst);
                    if (st < 0) return st;
                    todo = ri;
                }
                todo--;
            }
            for (int i = 0; i < ns; i++) {
                jcomp *c = sc[i];
                int nh = ns == 1 ? 1 : c->h, nv = ns == 1 ? 1 : c->v;
                for (int by = 0; by < nv; by++) {
                    for (int bx = 0; bx < nh; bx++) {
                        size_t row = (size_t)my * nv + by, col = (size_t)mx * nh + bx;
                        int st = fn(d, c, c->coef + (row * c->bw + col) * 64);
                        if (st < 0) return st;
                    }
                }
            }
            if (d->br.nbits < d->br.fake) return PLX_EFORMAT; /* consumed bits past the data */
        }
    }
    *pos = (size_t)(d->br.p - d->data);
    return PLX_OK;
}

static void parse_app(jdec *d, int marker, const uint8_t *s, size_t n) {
    if (marker == 0xE0 && n >= 14 && memcmp(s, "JFIF", 5) == 0) d->jfif = true;
    if (marker == 0xEE && n >= 12 && memcmp(s, "Adobe", 5) == 0) {
        d->adobe = true;
        d->adobe_transform = s[11];
    }
}

/* Walks the marker stream. header_only stops after SOF. */
static int jpeg_run(jdec *d, bool header_only) {
    const uint8_t *data = d->data;
    size_t len = d->len, pos = 2;
    if (len < 2 || data[0] != 0xFF || data[1] != 0xD8) return PLX_EFORMAT;
    for (;;) {
        int m = next_marker(data, len, &pos);
        if (m < 0 || m == 0xD9) break;
        if (m == 0xD8) return PLX_EFORMAT;
        if ((m >= 0xD0 && m <= 0xD7) || m == 0x01) continue; /* stray RSTn / TEM carry no length */
        if (pos + 2 > len) return PLX_EFORMAT;
        size_t seglen = (size_t)data[pos] << 8 | data[pos + 1];
        if (seglen < 2 || pos + seglen > len) return PLX_EFORMAT;
        const uint8_t *s = data + pos + 2;
        size_t n = seglen - 2;
        pos += seglen;
        int st = PLX_OK;
        switch (m) {
        case 0xC0: case 0xC1: case 0xC2:
            st = parse_sof(d, s, n, m == 0xC2);
            if (header_only) return st;
            break;
        case 0xC3: case 0xC5: case 0xC6: case 0xC7: case 0xC9: case 0xCA: case 0xCB:
        case 0xCD: case 0xCE: case 0xCF:
            return PLX_EUNSUPPORTED; /* lossless, hierarchical, arithmetic */
        case 0xC4: st = parse_dht(d, s, n); break;
        case 0xDB: st = parse_dqt(d, s, n); break;
        case 0xDD:
            if (n != 2) return PLX_EFORMAT;
            d->restart_interval = s[0] << 8 | s[1];
            break;
        case 0xDA:
            if (header_only) return PLX_EFORMAT;
            st = decode_scan(d, s, n, &pos);
            break;
        default:
            parse_app(d, m, s, n);
            break;
        }
        if (st < 0) return st;
    }
    if (!d->frame) return PLX_EFORMAT;
    for (int i = 0; i < d->nc; i++)
        if (!d->c[i].scanned) return PLX_EFORMAT;
    return PLX_OK;
}

/* ---- inverse DCT -----------------------------------------------------------------------
 * libjpeg's jpeg_idct_islow: the Loeffler-Ligtenberg-Moschytz 1-D factorisation applied to
 * columns then rows, constants scaled by 2^13, pass-1 outputs kept with 2 extra fraction bits.
 * 64-bit intermediates (libjpeg's JLONG on LP64) so corrupt coefficients cannot overflow.
 * DESCALE relies on >> of a negative value being an arithmetic (flooring) shift: implementation-
 * defined in C, true of every compiler we target, and the same assumption libjpeg makes. */
#define CONST_BITS 13
#define PASS1_BITS 2
#define FIX(x) ((int64_t)((x) * (1 << CONST_BITS) + 0.5))
#define DESCALE(x, n) (((x) + ((int64_t)1 << ((n) - 1))) >> (n))

/* The shared butterfly: in[0..7] at stride `st` -> out[0..7] at stride `os`, descaled by `sh`. */
#define IDCT_1D(in, st, out, os, sh)                                                         \
    do {                                                                                     \
        int64_t z2 = in[2 * st], z3 = in[6 * st];                                            \
        int64_t z1 = (z2 + z3) * FIX(0.541196100);                                           \
        int64_t t2 = z1 + z3 * -FIX(1.847759065), t3 = z1 + z2 * FIX(0.765366865);           \
        int64_t t0 = (in[0] + in[4 * st]) * (1 << CONST_BITS);                               \
        int64_t t1 = (in[0] - in[4 * st]) * (1 << CONST_BITS);                               \
        int64_t t10 = t0 + t3, t13 = t0 - t3, t11 = t1 + t2, t12 = t1 - t2;                  \
        int64_t o0 = in[7 * st], o1 = in[5 * st], o2 = in[3 * st], o3 = in[1 * st];          \
        int64_t q1 = o0 + o3, q2 = o1 + o2, q3 = o0 + o2, q4 = o1 + o3;                      \
        int64_t q5 = (q3 + q4) * FIX(1.175875602);                                           \
        o0 *= FIX(0.298631336); o1 *= FIX(2.053119869);                                      \
        o2 *= FIX(3.072711026); o3 *= FIX(1.501321110);                                      \
        q1 *= -FIX(0.899976223); q2 *= -FIX(2.562915447);                                    \
        q3 = q3 * -FIX(1.961570560) + q5; q4 = q4 * -FIX(0.390180644) + q5;                  \
        o0 += q1 + q3; o1 += q2 + q4; o2 += q2 + q3; o3 += q1 + q4;                          \
        out[0 * os] = DESCALE(t10 + o3, sh); out[7 * os] = DESCALE(t10 - o3, sh);            \
        out[1 * os] = DESCALE(t11 + o2, sh); out[6 * os] = DESCALE(t11 - o2, sh);            \
        out[2 * os] = DESCALE(t12 + o1, sh); out[5 * os] = DESCALE(t12 - o1, sh);            \
        out[3 * os] = DESCALE(t13 + o0, sh); out[4 * os] = DESCALE(t13 - o0, sh);            \
    } while (0)

static void idct_block(const int16_t *coef, const uint16_t *q, uint8_t *out, size_t stride) {
    int64_t ws[64], in[8], res[8];
    for (int x = 0; x < 8; x++) {
        /* The all-zero-AC shortcut libjpeg takes gives the same values as the full path. */
        for (int k = 0; k < 8; k++) in[k] = (int64_t)coef[k * 8 + x] * q[k * 8 + x];
        IDCT_1D(in, 1, (ws + x), 8, CONST_BITS - PASS1_BITS);
    }
    for (int y = 0; y < 8; y++) {
        const int64_t *row = ws + y * 8;
        IDCT_1D(row, 1, res, 1, CONST_BITS + PASS1_BITS + 3);
        uint8_t *o = out + y * stride;
        for (int x = 0; x < 8; x++) o[x] = (uint8_t)PLX_CLAMP(res[x] + 128, 0, 255);
    }
}

/* ---- upsampling ------------------------------------------------------------------------
 * libjpeg picks the method per component from the ratio to the max sampling factor: fancy
 * (triangle) for 2x1 and 2x2 when the component is wider than 2 samples, fancy for 1x2,
 * replication otherwise. Edges replicate the last real sample/row, which is what libjpeg's
 * special-cased first/last columns and its duplicated context rows amount to. */
static void upsample(const jdec *d, const jcomp *c, const uint8_t *src, size_t stride, uint8_t *dst) {
    int W = d->w, H = d->h, cw = c->cw, ch = c->ch;
    bool integral = d->hmax % c->h == 0 && d->vmax % c->v == 0;
    int rh = d->hmax / c->h, rv = d->vmax / c->v;
    if (integral && rh == 2 && rv == 1 && cw > 2) {
        for (int y = 0; y < H; y++) {
            const uint8_t *in = src + (size_t)y * stride;
            uint8_t *o = dst + (size_t)y * W;
            for (int x = 0; x < cw; x++) {
                int v3 = in[x] * 3, l = in[x > 0 ? x - 1 : 0], r = in[x < cw - 1 ? x + 1 : x];
                o[2 * x] = (uint8_t)((v3 + l + 1) >> 2);
                if (2 * x + 1 < W) o[2 * x + 1] = (uint8_t)((v3 + r + 2) >> 2);
            }
        }
    } else if (integral && rh == 1 && rv == 2) {
        for (int y = 0; y < H; y++) {
            int r = y >> 1, far = (y & 1) ? PLX_MIN(r + 1, ch - 1) : PLX_MAX(r - 1, 0);
            const uint8_t *a = src + (size_t)r * stride, *b = src + (size_t)far * stride;
            uint8_t *o = dst + (size_t)y * W;
            int bias = (y & 1) ? 2 : 1;
            for (int x = 0; x < W; x++) o[x] = (uint8_t)((a[x] * 3 + b[x] + bias) >> 2);
        }
    } else if (integral && rh == 2 && rv == 2 && cw > 2) {
        int *cs = PLX_NEW(int, cw);
        for (int y = 0; y < H; y++) {
            int r = y >> 1, far = (y & 1) ? PLX_MIN(r + 1, ch - 1) : PLX_MAX(r - 1, 0);
            const uint8_t *a = src + (size_t)r * stride, *b = src + (size_t)far * stride;
            uint8_t *o = dst + (size_t)y * W;
            for (int x = 0; x < cw; x++) cs[x] = a[x] * 3 + b[x];
            for (int x = 0; x < cw; x++) {
                int t3 = cs[x] * 3, l = cs[x > 0 ? x - 1 : 0], rr = cs[x < cw - 1 ? x + 1 : x];
                o[2 * x] = (uint8_t)((t3 + l + 8) >> 4);
                if (2 * x + 1 < W) o[2 * x + 1] = (uint8_t)((t3 + rr + 7) >> 4);
            }
        }
        free(cs);
    } else {
        /* Box replication; for integral ratios this is libjpeg's int_upsample. Fractional ratios
         * (which libjpeg refuses) get the same nearest-sample rule. */
        for (int y = 0; y < H; y++) {
            const uint8_t *in = src + (size_t)((int64_t)y * c->v / d->vmax) * stride;
            uint8_t *o = dst + (size_t)y * W;
            for (int x = 0; x < W; x++) o[x] = in[(int64_t)x * c->h / d->hmax];
        }
    }
}

/* ---- colour conversion: libjpeg's 16-bit fixed-point tables ------------------------------ */
#define CBITS 16
#define CFIX(x) ((int32_t)((x) * (1 << CBITS) + 0.5))
#define CHALF ((int32_t)1 << (CBITS - 1))
#define GBIAS (256 << CBITS) /* keeps the green sum positive so >> is a true floor */

typedef struct { int cr_r[256], cb_b[256]; int32_t cr_g[256], cb_g[256]; } ycc_tables;

static int floor_shift(int64_t v) { return (int)(v >= 0 ? v >> CBITS : -((-v + (1 << CBITS) - 1) >> CBITS)); }

static void ycc_init(ycc_tables *t) {
    for (int i = 0; i < 256; i++) {
        int x = i - 128;
        t->cr_r[i] = floor_shift((int64_t)CFIX(1.40200) * x + CHALF);
        t->cb_b[i] = floor_shift((int64_t)CFIX(1.77200) * x + CHALF);
        t->cr_g[i] = -CFIX(0.71414) * x;
        t->cb_g[i] = -CFIX(0.34414) * x + CHALF + GBIAS;
    }
}

static int jpeg_output(jdec *d, uint8_t **pixels, int *nc_out) {
    int W = d->w, H = d->h, nc = d->nc;
    uint8_t *plane[3] = {NULL}, *full[3] = {NULL};
    size_t fstride[3];
    for (int i = 0; i < nc; i++) {
        jcomp *c = &d->c[i];
        size_t stride = (size_t)c->bw * 8;
        plane[i] = plx_malloc(stride * c->bh * 8);
        int nbx = (c->cw + 7) / 8, nby = (c->ch + 7) / 8;
        for (int by = 0; by < nby; by++)
            for (int bx = 0; bx < nbx; bx++)
                idct_block(c->coef + ((size_t)by * c->bw + bx) * 64, c->q,
                           plane[i] + (size_t)by * 8 * stride + (size_t)bx * 8, stride);
        if (c->h == d->hmax && c->v == d->vmax) {
            full[i] = plane[i];
            fstride[i] = stride;
        } else {
            full[i] = plx_malloc((size_t)W * H);
            upsample(d, c, plane[i], stride, full[i]);
            fstride[i] = (size_t)W;
        }
    }
    uint8_t *out = plx_malloc((size_t)W * H * nc);
    bool rgb = nc == 3 && !d->jfif &&
               (d->adobe ? d->adobe_transform == 0
                         : d->c[0].id == 'R' && d->c[1].id == 'G' && d->c[2].id == 'B');
    if (nc == 1) {
        for (int y = 0; y < H; y++) memcpy(out + (size_t)y * W, full[0] + y * fstride[0], (size_t)W);
    } else if (rgb) {
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                for (int k = 0; k < 3; k++) out[((size_t)y * W + x) * 3 + k] = full[k][y * fstride[k] + x];
    } else {
        ycc_tables t;
        ycc_init(&t);
        for (int y = 0; y < H; y++) {
            const uint8_t *py = full[0] + y * fstride[0], *pb = full[1] + y * fstride[1],
                          *pr = full[2] + y * fstride[2];
            uint8_t *o = out + (size_t)y * W * 3;
            for (int x = 0; x < W; x++) {
                int Y = py[x], cb = pb[x], cr = pr[x];
                int g = Y + ((t.cb_g[cb] + t.cr_g[cr]) >> CBITS) - 256;
                o[3 * x + 0] = (uint8_t)PLX_CLAMP(Y + t.cr_r[cr], 0, 255);
                o[3 * x + 1] = (uint8_t)PLX_CLAMP(g, 0, 255);
                o[3 * x + 2] = (uint8_t)PLX_CLAMP(Y + t.cb_b[cb], 0, 255);
            }
        }
    }
    for (int i = 0; i < nc; i++) {
        if (full[i] != plane[i]) free(full[i]);
        free(plane[i]);
    }
    *pixels = out;
    *nc_out = nc;
    return PLX_OK;
}

static jdec *jdec_new(const uint8_t *data, size_t len) {
    jdec *d = PLX_NEW(jdec, 1);
    d->data = data;
    d->len = len;
    return d;
}

static void jdec_free(jdec *d) {
    for (int i = 0; i < 3; i++) free(d->c[i].coef);
    free(d);
}

int plx_jpeg_info(const uint8_t *data, size_t len, int *w, int *h, int *c) {
    jdec *d = jdec_new(data, len);
    int st = jpeg_run(d, true);
    if (st == PLX_OK) {
        *w = d->w;
        *h = d->h;
        *c = d->nc;
    }
    jdec_free(d);
    return st;
}

int plx_jpeg_decode_u8(const uint8_t *data, size_t len, int *w, int *h, int *c, uint8_t **pixels) {
    *pixels = NULL;
    jdec *d = jdec_new(data, len);
    int st = jpeg_run(d, false);
    if (st == PLX_OK) st = jpeg_output(d, pixels, c);
    if (st == PLX_OK) {
        *w = d->w;
        *h = d->h;
    }
    jdec_free(d);
    return st;
}

int plx_jpeg_decode(const uint8_t *data, size_t len, plx_img *out) {
    int w, h, c;
    uint8_t *px;
    int st = plx_jpeg_decode_u8(data, len, &w, &h, &c, &px);
    if (st < 0) return st;
    *out = plx_img_from_u8(px, w, h, c, 1.0f / 255.0f);
    free(px);
    return PLX_OK;
}
