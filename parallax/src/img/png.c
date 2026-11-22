/* PNG writer: row filtering, a deflate compressor and the PNG/zlib framing around it. */
#include "img/png.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

/* ---- checksums -------------------------------------------------------------------------- */
static uint32_t g_crc_table[256];
static pthread_once_t g_crc_once = PTHREAD_ONCE_INIT;

static void crc_init(void) {
    for (uint32_t n = 0; n < 256; n++) {
        uint32_t c = n;
        for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1; /* reflected 0x04C11DB7 */
        g_crc_table[n] = c;
    }
}

uint32_t plx_crc32(uint32_t crc, const void *data, size_t len) {
    pthread_once(&g_crc_once, crc_init);
    const uint8_t *p = data;
    crc = ~crc;
    for (size_t i = 0; i < len; i++) crc = g_crc_table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

uint32_t plx_adler32(uint32_t adler, const void *data, size_t len) {
    const uint8_t *p = data;
    uint32_t a = adler & 0xFFFF, b = adler >> 16;
    while (len > 0) {
        size_t n = PLX_MIN(len, (size_t)5552); /* largest run whose sums cannot overflow 32 bits */
        len -= n;
        while (n--) {
            a += *p++;
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    return b << 16 | a;
}

/* ---- growable byte buffer and LSB-first bit writer -------------------------------------- */
typedef struct {
    uint8_t *p;
    size_t n, cap;
} bytebuf;

static void bb_reserve(bytebuf *b, size_t extra) {
    if (b->n + extra <= b->cap) return;
    b->cap = PLX_MAX(PLX_MAX(b->cap * 2, b->n + extra), (size_t)4096);
    b->p = plx_realloc(b->p, b->cap);
}

static void bb_put(bytebuf *b, const void *data, size_t n) {
    bb_reserve(b, n);
    memcpy(b->p + b->n, data, n);
    b->n += n;
}

static void bb_be32(bytebuf *b, uint32_t v) {
    uint8_t x[4] = {(uint8_t)(v >> 24), (uint8_t)(v >> 16), (uint8_t)(v >> 8), (uint8_t)v};
    bb_put(b, x, 4);
}

typedef struct {
    bytebuf out;
    uint64_t acc;
    int nbits;
} bitwr;

static void bw_put(bitwr *w, uint32_t v, int n) {
    w->acc |= (uint64_t)v << w->nbits;
    w->nbits += n;
    if (w->nbits >= 32) {
        bb_reserve(&w->out, 4);
        for (int k = 0; k < 4; k++) w->out.p[w->out.n++] = (uint8_t)(w->acc >> (8 * k));
        w->acc >>= 32;
        w->nbits -= 32;
    }
}

static void bw_align(bitwr *w) {
    bw_put(w, 0, (8 - (w->nbits & 7)) & 7);
    while (w->nbits > 0) {
        bb_put(&w->out, &(uint8_t){(uint8_t)w->acc}, 1);
        w->acc >>= 8;
        w->nbits -= 8;
    }
}

/* ---- Huffman code construction ----------------------------------------------------------
 * Length-limited optimal code lengths by package-merge. At every level the merged list is a
 * prefix of lightest leaves interleaved with packages, so instead of tracking which leaves each
 * package holds we only count, level by level, how many leaves the selected prefix contains:
 * each such leaf gains one bit of code length. */
#define MAX_SYMS 288

static void huff_lengths(const uint32_t *freq, int n, int maxlen, uint8_t *len) {
    int sym[MAX_SYMS], m = 0;
    memset(len, 0, (size_t)n);
    for (int i = 0; i < n; i++) {
        if (!freq[i]) continue;
        int j = m++;
        for (; j > 0 && freq[sym[j - 1]] > freq[i]; j--) sym[j] = sym[j - 1]; /* insertion sort */
        sym[j] = i;
    }
    if (m == 0) return;
    if (m == 1) {
        len[sym[0]] = 1;
        return;
    }
    /* level l list: weight and whether the item is a leaf; at most 2m items per level */
    uint64_t *wt = plx_malloc(sizeof *wt * (size_t)maxlen * 2 * m);
    uint8_t *leaf = plx_malloc((size_t)maxlen * 2 * m);
    int size[16];
    for (int i = 0; i < m; i++) {
        wt[i] = freq[sym[i]];
        leaf[i] = 1;
    }
    size[0] = m;
    for (int l = 1; l < maxlen; l++) {
        const uint64_t *pw = wt + (size_t)(l - 1) * 2 * m;
        uint64_t *cw = wt + (size_t)l * 2 * m;
        uint8_t *cl = leaf + (size_t)l * 2 * m;
        int npk = size[l - 1] / 2, li = 0, pi = 0, k = 0;
        while (li < m || pi < npk) {
            uint64_t pkw = pi < npk ? pw[2 * pi] + pw[2 * pi + 1] : 0;
            if (pi >= npk || (li < m && wt[li] <= pkw)) {
                cw[k] = wt[li++];
                cl[k++] = 1;
            } else {
                cw[k] = pkw;
                cl[k++] = 0;
                pi++;
            }
        }
        size[l] = k;
    }
    int take = 2 * m - 2;
    for (int l = maxlen - 1; l >= 0 && take > 0; l--) {
        const uint8_t *cl = leaf + (size_t)l * 2 * m;
        int nleaf = 0;
        for (int k = 0; k < take; k++) nleaf += cl[k];
        for (int k = 0; k < nleaf; k++) len[sym[k]]++;
        take = 2 * (take - nleaf);
    }
    free(wt);
    free(leaf);
}

/* Canonical codes (RFC 1951 3.2.2), bit-reversed because deflate sends codes MSB-first
 * through an LSB-first bit stream. */
static void huff_codes(const uint8_t *len, int n, uint16_t *code) {
    int count[16] = {0}, next[16];
    for (int i = 0; i < n; i++) count[len[i]]++;
    count[0] = 0;
    int c = 0;
    for (int b = 1; b < 16; b++) {
        c = (c + count[b - 1]) << 1;
        next[b] = c;
    }
    for (int i = 0; i < n; i++) {
        if (!len[i]) continue;
        int v = next[len[i]]++, r = 0;
        for (int b = 0; b < len[i]; b++) r |= ((v >> b) & 1) << (len[i] - 1 - b);
        code[i] = (uint16_t)r;
    }
}

/* ---- deflate ---------------------------------------------------------------------------- */
static const uint16_t k_len_base[29] = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                                        31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
static const uint8_t k_len_extra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                        2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
static const uint16_t k_dist_base[30] = {1,   2,   3,   4,   5,   7,    9,    13,   17,   25,
                                         33,  49,  65,  97,  129, 193,  257,  385,  513,  769,
                                         1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
static const uint8_t k_dist_extra[30] = {0, 0, 0, 0, 1, 1, 2, 2,  3,  3,  4,  4,  5,  5,  6,
                                         6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
/* RFC 1951 3.2.6: order in which code-length code lengths are sent */
static const uint8_t k_cl_order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

#define WSIZE 32768
#define HASH_BITS 15
#define MAX_CHAIN 128   /* candidates examined per position: zlib level ~6-7 effort */
#define NICE_LEN 192    /* a match this long is good enough to stop searching */
#define BLOCK_TOKENS 65536

typedef struct {
    uint16_t len;  /* literal byte when dist == 0, else match length */
    uint16_t dist;
} token;

typedef struct {
    const uint8_t *src;
    size_t n, block_start, emitted;
    int64_t *head, *prev;
    token *tok;
    int ntok;
    uint32_t lfreq[286], dfreq[30];
    uint8_t len_code[259], dist_code[512];
    bitwr bw;
} deflater;

static int dist_sym(const deflater *z, int d) { return d <= 256 ? z->dist_code[d - 1] : z->dist_code[256 + ((d - 1) >> 7)]; }

static void fixed_lengths(uint8_t *ll, uint8_t *dl) {
    for (int i = 0; i < 288; i++) ll[i] = i < 144 ? 8 : i < 256 ? 9 : i < 280 ? 7 : 8;
    for (int i = 0; i < 30; i++) dl[i] = 5;
}

/* Bits the block's tokens cost under the given lengths, extra bits included. */
static uint64_t data_bits(const deflater *z, const uint8_t *ll, const uint8_t *dl) {
    uint64_t bits = 0;
    for (int i = 0; i < 286; i++) bits += (uint64_t)z->lfreq[i] * (ll[i] + (i > 256 ? k_len_extra[i - 257] : 0));
    for (int i = 0; i < 30; i++) bits += (uint64_t)z->dfreq[i] * (dl[i] + k_dist_extra[i]);
    return bits;
}

static void emit_tokens(deflater *z, const uint8_t *ll, const uint8_t *dl) {
    uint16_t lc[288], dc[30];
    huff_codes(ll, 288, lc);
    huff_codes(dl, 30, dc);
    for (int i = 0; i < z->ntok; i++) {
        token t = z->tok[i];
        if (t.dist == 0) {
            bw_put(&z->bw, lc[t.len], ll[t.len]);
            continue;
        }
        int ls = z->len_code[t.len], ds = dist_sym(z, t.dist);
        bw_put(&z->bw, lc[257 + ls], ll[257 + ls]);
        bw_put(&z->bw, t.len - k_len_base[ls], k_len_extra[ls]);
        bw_put(&z->bw, dc[ds], dl[ds]);
        bw_put(&z->bw, t.dist - k_dist_base[ds], k_dist_extra[ds]);
    }
    bw_put(&z->bw, lc[256], ll[256]);
}

/* Code-length sequence for the dynamic header, run-length coded with symbols 16/17/18. */
static int rle_lengths(const uint8_t *lens, int n, uint8_t *sym, uint8_t *extra) {
    int k = 0;
    for (int i = 0; i < n;) {
        int v = lens[i], run = 1;
        while (i + run < n && lens[i + run] == v) run++;
        int left = run;
        if (v == 0) {
            while (left >= 11) { int r = PLX_MIN(left, 138); sym[k] = 18; extra[k++] = (uint8_t)(r - 11); left -= r; }
            if (left >= 3) { sym[k] = 17; extra[k++] = (uint8_t)(left - 3); left = 0; }
        } else {
            sym[k] = (uint8_t)v;
            extra[k++] = 0;
            left--;
            while (left >= 3) { int r = PLX_MIN(left, 6); sym[k] = 16; extra[k++] = (uint8_t)(r - 3); left -= r; }
        }
        while (left-- > 0) { sym[k] = (uint8_t)v; extra[k++] = 0; }
        i += run;
    }
    return k;
}

/* Gives a frequency table at least two used symbols: inflaters (zlib's included) reject an
 * incomplete code, and a lone symbol would get a 1-bit code with an unused sibling. */
static void ensure_two(uint32_t *f, int n) {
    int used = 0;
    for (int i = 0; i < n; i++) used += f[i] != 0;
    for (int i = 0; i < n && used < 2; i++)
        if (!f[i]) { f[i] = 1; used++; }
}

static void flush_block(deflater *z, bool final) {
    z->lfreq[256] = 1;
    size_t raw = z->emitted - z->block_start;

    uint8_t fl[288], fd[30];
    fixed_lengths(fl, fd);
    uint64_t fixed_bits = 3 + data_bits(z, fl, fd);

    uint32_t lf[286], df[30];
    memcpy(lf, z->lfreq, sizeof lf);
    memcpy(df, z->dfreq, sizeof df);
    ensure_two(lf, 286);
    ensure_two(df, 30);
    uint8_t ll[288] = {0}, dl[30], lens[286 + 30], cls[286 + 30], clx[286 + 30], cll[19];
    huff_lengths(lf, 286, 15, ll);
    huff_lengths(df, 30, 15, dl);
    int hlit = 286, hdist = 30;
    while (hlit > 257 && !ll[hlit - 1]) hlit--;
    while (hdist > 1 && !dl[hdist - 1]) hdist--;
    memcpy(lens, ll, (size_t)hlit);
    memcpy(lens + hlit, dl, (size_t)hdist);
    int ncl = rle_lengths(lens, hlit + hdist, cls, clx);
    uint32_t clf[19] = {0};
    for (int i = 0; i < ncl; i++) clf[cls[i]]++;
    ensure_two(clf, 19);
    huff_lengths(clf, 19, 7, cll);
    int hclen = 19;
    while (hclen > 4 && !cll[k_cl_order[hclen - 1]]) hclen--;
    uint64_t dyn_bits = 3 + 14 + 3 * (uint64_t)hclen + data_bits(z, ll, dl);
    for (int i = 0; i < ncl; i++) dyn_bits += cll[cls[i]] + (cls[i] == 16 ? 2 : cls[i] == 17 ? 3 : cls[i] == 18 ? 7 : 0);

    size_t nstored = PLX_MAX((raw + 65534) / 65535, (size_t)1);
    uint64_t stored_bits = 8 * ((uint64_t)raw + 5 * nstored) + 7;

    if (stored_bits <= fixed_bits && stored_bits <= dyn_bits) {
        const uint8_t *p = z->src + z->block_start;
        for (size_t s = 0; s < nstored; s++) {
            uint32_t n = (uint32_t)PLX_MIN(raw - s * 65535, (size_t)65535);
            bw_put(&z->bw, final && s + 1 == nstored, 1);
            bw_put(&z->bw, 0, 2);
            bw_align(&z->bw);
            uint8_t hdr[4] = {(uint8_t)n, (uint8_t)(n >> 8), (uint8_t)~n, (uint8_t)(~n >> 8)};
            bb_put(&z->bw.out, hdr, 4);
            bb_put(&z->bw.out, p + s * 65535, n);
        }
    } else if (fixed_bits <= dyn_bits) {
        bw_put(&z->bw, final, 1);
        bw_put(&z->bw, 1, 2);
        emit_tokens(z, fl, fd);
    } else {
        bw_put(&z->bw, final, 1);
        bw_put(&z->bw, 2, 2);
        bw_put(&z->bw, (uint32_t)(hlit - 257), 5);
        bw_put(&z->bw, (uint32_t)(hdist - 1), 5);
        bw_put(&z->bw, (uint32_t)(hclen - 4), 4);
        for (int i = 0; i < hclen; i++) bw_put(&z->bw, cll[k_cl_order[i]], 3);
        uint16_t clc[19];
        huff_codes(cll, 19, clc);
        for (int i = 0; i < ncl; i++) {
            int s = cls[i];
            bw_put(&z->bw, clc[s], cll[s]);
            if (s >= 16) bw_put(&z->bw, clx[i], s == 16 ? 2 : s == 17 ? 3 : 7);
        }
        emit_tokens(z, ll, dl);
    }
    z->ntok = 0;
    z->block_start = z->emitted;
    memset(z->lfreq, 0, sizeof z->lfreq);
    memset(z->dfreq, 0, sizeof z->dfreq);
}

static void emit(deflater *z, int len, int dist) {
    if (dist == 0) {
        z->tok[z->ntok++] = (token){(uint16_t)len, 0};
        z->lfreq[len]++;
        z->emitted++;
    } else {
        z->tok[z->ntok++] = (token){(uint16_t)len, (uint16_t)dist};
        z->lfreq[257 + z->len_code[len]]++;
        z->dfreq[dist_sym(z, dist)]++;
        z->emitted += (size_t)len;
    }
    if (z->ntok == BLOCK_TOKENS) flush_block(z, false);
}

static inline uint32_t hash3(const uint8_t *p) {
    return ((uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16) * 2654435761u >> (32 - HASH_BITS);
}

static void insert(deflater *z, size_t i) {
    uint32_t h = hash3(z->src + i);
    z->prev[i & (WSIZE - 1)] = z->head[h];
    z->head[h] = (int64_t)i;
}

/* Longest match for position i among the chain of earlier positions with the same hash. */
static int longest(const deflater *z, size_t i, int *dist) {
    const uint8_t *s = z->src;
    int limit = (int)PLX_MIN(z->n - i, (size_t)258), best = 0;
    int64_t cand = z->head[hash3(s + i)];
    for (int chain = MAX_CHAIN; cand >= 0 && (int64_t)i - cand <= WSIZE && chain > 0; chain--) {
        const uint8_t *a = s + cand, *b = s + i;
        if (a[best] == b[best]) {
            int l = 0;
            while (l < limit && a[l] == b[l]) l++;
            if (l > best) {
                best = l;
                *dist = (int)((int64_t)i - cand);
                if (l >= NICE_LEN || l == limit) break;
            }
        }
        cand = z->prev[cand & (WSIZE - 1)];
    }
    return best >= 3 ? best : 0;
}

size_t plx_zlib_compress(const uint8_t *src, size_t n, uint8_t **out) {
    deflater *z = PLX_NEW(deflater, 1);
    z->src = src;
    z->n = n;
    z->head = plx_malloc(sizeof *z->head << HASH_BITS);
    for (int i = 0; i < 1 << HASH_BITS; i++) z->head[i] = -1;
    z->prev = plx_malloc(sizeof *z->prev * WSIZE);
    z->tok = plx_malloc(sizeof *z->tok * BLOCK_TOKENS);
    for (int c = 0; c < 29; c++)
        for (int l = k_len_base[c]; l < k_len_base[c] + (1 << k_len_extra[c]) && l <= 258; l++) z->len_code[l] = (uint8_t)c;
    for (int c = 0; c < 30; c++)
        for (int d = k_dist_base[c]; d < k_dist_base[c] + (1 << k_dist_extra[c]); d++)
            z->dist_code[d <= 256 ? d - 1 : 256 + ((d - 1) >> 7)] = (uint8_t)c;

    bb_put(&z->bw.out, (const uint8_t[]){0x78, 0x9C}, 2); /* deflate, 32K window, default level */

    /* One-step lazy matching: a match found at i is emitted only if i+1 does not start a
     * longer one; otherwise i becomes a literal and the i+1 match is reconsidered. */
    bool have_prev = false;
    int prev_len = 0, prev_dist = 0;
    for (size_t i = 0; i < n;) {
        int cur_len = 0, cur_dist = 0;
        if (n - i >= 3) {
            cur_len = longest(z, i, &cur_dist);
            insert(z, i);
        }
        if (have_prev) {
            if (prev_len && cur_len <= prev_len) {
                emit(z, prev_len, prev_dist);
                size_t end = i - 1 + (size_t)prev_len;
                for (size_t p = i + 1; p < end; p++)
                    if (n - p >= 3) insert(z, p);
                i = end;
                have_prev = false;
                continue;
            }
            emit(z, z->src[i - 1], 0);
        }
        prev_len = cur_len;
        prev_dist = cur_dist;
        have_prev = true;
        i++;
    }
    if (have_prev) emit(z, z->src[n - 1], 0); /* the last position cannot start a match */
    flush_block(z, true);
    bw_align(&z->bw);
    bb_be32(&z->bw.out, plx_adler32(1, src, n));

    size_t len = z->bw.out.n;
    *out = z->bw.out.p;
    free(z->head);
    free(z->prev);
    free(z->tok);
    free(z);
    return len;
}

/* ---- PNG framing ------------------------------------------------------------------------ */
static int paeth(int a, int b, int c) {
    int p = a + b - c, pa = abs(p - a), pb = abs(p - b), pc = abs(p - c);
    return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

/* Filters every row with the type minimising the sum of |residual| read as signed bytes,
 * the heuristic the PNG specification recommends. */
static uint8_t *filter_rows(const uint8_t *raw, size_t rowbytes, int h, int bpp) {
    uint8_t *out = plx_malloc((rowbytes + 1) * (size_t)h), *cand = plx_malloc(rowbytes * 5);
    uint8_t *zero = PLX_NEW(uint8_t, rowbytes);
    for (int y = 0; y < h; y++) {
        const uint8_t *cur = raw + (size_t)y * rowbytes, *up = y ? cur - rowbytes : zero;
        uint64_t best = UINT64_MAX;
        int best_f = 0;
        for (int f = 0; f < 5; f++) {
            uint8_t *o = cand + (size_t)f * rowbytes;
            uint64_t score = 0;
            for (size_t x = 0; x < rowbytes; x++) {
                int a = x >= (size_t)bpp ? cur[x - bpp] : 0, b = up[x], c = x >= (size_t)bpp ? up[x - bpp] : 0;
                int pred = f == 0 ? 0 : f == 1 ? a : f == 2 ? b : f == 3 ? (a + b) >> 1 : paeth(a, b, c);
                uint8_t r = (uint8_t)(cur[x] - pred);
                o[x] = r;
                score += r < 128 ? r : 256 - r;
            }
            if (score < best) {
                best = score;
                best_f = f;
            }
        }
        uint8_t *dst = out + (size_t)y * (rowbytes + 1);
        dst[0] = (uint8_t)best_f;
        memcpy(dst + 1, cand + (size_t)best_f * rowbytes, rowbytes);
    }
    free(cand);
    free(zero);
    return out;
}

static void put_chunk(bytebuf *b, const char *type, const uint8_t *data, size_t n) {
    bb_be32(b, (uint32_t)n);
    size_t start = b->n;
    bb_put(b, type, 4);
    if (n) bb_put(b, data, n);
    bb_be32(b, plx_crc32(0, b->p + start, n + 4));
}

#define IDAT_MAX ((size_t)1 << 18)

static int png_write(const char *path, const uint8_t *raw, int w, int h, int bpp, int depth, int ctype) {
    size_t rowbytes = (size_t)w * bpp;
    uint8_t *filtered = filter_rows(raw, rowbytes, h, bpp), *z;
    size_t zn = plx_zlib_compress(filtered, (rowbytes + 1) * (size_t)h, &z);
    free(filtered);

    bytebuf b = {0};
    bb_put(&b, (const uint8_t[]){0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'}, 8);
    uint8_t ihdr[13] = {(uint8_t)(w >> 24), (uint8_t)(w >> 16), (uint8_t)(w >> 8), (uint8_t)w,
                        (uint8_t)(h >> 24), (uint8_t)(h >> 16), (uint8_t)(h >> 8), (uint8_t)h,
                        (uint8_t)depth, (uint8_t)ctype, 0, 0, 0};
    put_chunk(&b, "IHDR", ihdr, sizeof ihdr);
    for (size_t off = 0; off < zn; off += IDAT_MAX) put_chunk(&b, "IDAT", z + off, PLX_MIN(zn - off, IDAT_MAX));
    put_chunk(&b, "IEND", NULL, 0);
    free(z);
    int st = plx_write_file(path, b.p, b.n);
    free(b.p);
    return st;
}

int plx_png_write_u8(const char *path, const uint8_t *pixels, int w, int h, int c) {
    static const int ctype[5] = {0, 0, 4, 2, 6};
    if (w <= 0 || h <= 0 || c < 1 || c > 4) return PLX_EINVAL;
    return png_write(path, pixels, w, h, c, 8, ctype[c]);
}

int plx_png_write_gray16(const char *path, const uint16_t *pixels, int w, int h) {
    if (w <= 0 || h <= 0) return PLX_EINVAL;
    size_t n = (size_t)w * h;
    uint8_t *be = plx_malloc(2 * n);
    for (size_t i = 0; i < n; i++) {
        be[2 * i] = (uint8_t)(pixels[i] >> 8);
        be[2 * i + 1] = (uint8_t)pixels[i];
    }
    int st = png_write(path, be, w, h, 2, 16, 0);
    free(be);
    return st;
}
