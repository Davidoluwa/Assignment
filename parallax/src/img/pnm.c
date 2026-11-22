#include "img/pnm.h"

#include <stdlib.h>
#include <string.h>

#define PNM_MAX_DIM (1 << 24)

static bool is_space(int ch) { return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\v' || ch == '\f' || ch == '\r'; }

/* Next whitespace-delimited header token, skipping whitespace and comments. */
static int next_token(const uint8_t *d, size_t len, size_t *pos, char *buf, size_t cap) {
    size_t p = *pos;
    for (;;) {
        while (p < len && is_space(d[p])) p++;
        if (p < len && d[p] == '#') {
            while (p < len && d[p] != '\n' && d[p] != '\r') p++;
            continue;
        }
        break;
    }
    size_t n = 0;
    while (p < len && !is_space(d[p]) && d[p] != '#') {
        if (n + 1 >= cap) return PLX_EFORMAT;
        buf[n++] = (char)d[p++];
    }
    if (n == 0) return PLX_EFORMAT;
    buf[n] = 0;
    *pos = p;
    return PLX_OK;
}

static int next_uint(const uint8_t *d, size_t len, size_t *pos, int lo, int hi, int *out) {
    char buf[16];
    if (next_token(d, len, pos, buf, sizeof buf) < 0) return PLX_EFORMAT;
    int64_t v = 0;
    for (const char *s = buf; *s; s++) {
        if (*s < '0' || *s > '9') return PLX_EFORMAT;
        v = v * 10 + (*s - '0');
        if (v > hi) return PLX_EFORMAT;
    }
    if (v < lo) return PLX_EFORMAT;
    *out = (int)v;
    return PLX_OK;
}

static float load_f32(const uint8_t *p, bool little) {
    uint32_t u = little ? (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24
                        : (uint32_t)p[3] | (uint32_t)p[2] << 8 | (uint32_t)p[1] << 16 | (uint32_t)p[0] << 24;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

int plx_pnm_decode(const uint8_t *data, size_t len, plx_img *out) {
    if (len < 2 || data[0] != 'P') return PLX_EFORMAT;
    int kind = data[1], c;
    if (kind == '5' || kind == 'f') c = 1;
    else if (kind == '6' || kind == 'F') c = 3;
    else return PLX_EUNSUPPORTED;
    size_t pos = 2;
    if (pos < len && !is_space(data[pos]) && data[pos] != '#') return PLX_EFORMAT;
    int w, h, maxval = 0;
    bool little = true;
    if (next_uint(data, len, &pos, 1, PNM_MAX_DIM, &w) < 0 || next_uint(data, len, &pos, 1, PNM_MAX_DIM, &h) < 0)
        return PLX_EFORMAT;
    if (kind == '5' || kind == '6') {
        if (next_uint(data, len, &pos, 1, 65535, &maxval) < 0) return PLX_EFORMAT;
    } else {
        char buf[64], *end;
        if (next_token(data, len, &pos, buf, sizeof buf) < 0) return PLX_EFORMAT;
        double scale = strtod(buf, &end);
        if (*end || scale == 0 || scale != scale) return PLX_EFORMAT;
        little = scale < 0;
    }
    if (pos >= len || !is_space(data[pos])) return PLX_EFORMAT;
    pos++; /* exactly one whitespace byte separates header and raster */

    size_t bytes = maxval > 255 ? 2 : maxval ? 1 : 4, n = (size_t)w * h * c;
    if ((len - pos) / bytes < n) return PLX_EFORMAT;
    const uint8_t *r = data + pos;
    plx_img im = plx_img_new(w, h, c);
    for (int y = 0; y < h; y++) {
        int src_row = maxval ? y : h - 1 - y; /* PFM is bottom row first */
        for (int x = 0; x < w; x++) {
            for (int k = 0; k < c; k++) {
                size_t i = ((size_t)src_row * w + x) * c + k;
                float v;
                if (bytes == 4) v = load_f32(r + 4 * i, little);
                else if (bytes == 2) v = (float)(r[2 * i] << 8 | r[2 * i + 1]) / (float)maxval;
                else v = (float)r[i] / (float)maxval;
                PLX_PX(&im, x, y, k) = v;
            }
        }
    }
    *out = im;
    return PLX_OK;
}

int plx_pnm_read(const char *path, plx_img *out) {
    uint8_t *data;
    size_t len;
    int st = plx_read_file(path, &data, &len);
    if (st < 0) return st;
    st = plx_pnm_decode(data, len, out);
    free(data);
    return st;
}

int plx_pnm_write_u8(const char *path, const uint8_t *pixels, int w, int h, int c) {
    if (w <= 0 || h <= 0 || (c != 1 && c != 3)) return PLX_EINVAL;
    char hdr[64];
    int hn = snprintf(hdr, sizeof hdr, "P%c\n%d %d\n255\n", c == 1 ? '5' : '6', w, h);
    size_t n = (size_t)w * h * c;
    uint8_t *buf = plx_malloc((size_t)hn + n);
    memcpy(buf, hdr, (size_t)hn);
    memcpy(buf + hn, pixels, n);
    int st = plx_write_file(path, buf, (size_t)hn + n);
    free(buf);
    return st;
}

int plx_pfm_write(const char *path, const plx_img *im) {
    if (im->c != 1 && im->c != 3) return PLX_EINVAL;
    char hdr[64];
    int hn = snprintf(hdr, sizeof hdr, "P%c\n%d %d\n-1.0\n", im->c == 1 ? 'f' : 'F', im->w, im->h);
    size_t n = (size_t)im->w * im->h * im->c;
    uint8_t *buf = plx_malloc((size_t)hn + 4 * n), *p = buf + hn;
    memcpy(buf, hdr, (size_t)hn);
    for (int y = im->h - 1; y >= 0; y--) {
        for (int x = 0; x < im->w; x++) {
            for (int k = 0; k < im->c; k++) {
                uint32_t u;
                float v = PLX_PX(im, x, y, k);
                memcpy(&u, &v, sizeof u);
                for (int b = 0; b < 4; b++) *p++ = (uint8_t)(u >> (8 * b));
            }
        }
    }
    int st = plx_write_file(path, buf, (size_t)hn + 4 * n);
    free(buf);
    return st;
}
