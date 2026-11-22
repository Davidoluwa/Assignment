/* Local binary patterns, (8, 1) square ring and (8, 2) interpolated circle, plus uniform labels. */
#include "feat/feat.h"

#include <math.h>

/* Neighbour p as an integer corner (ix, iy) plus fractional offset (fx, fy) in [0, 1). */
typedef struct {
    int   ix, iy;
    float fx, fy;
} lbp_tap;

static void lbp_taps(int radius, lbp_tap t[8]) {
    static const int ring[8][2] = {{1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1}};
    for (int p = 0; p < 8; p++) {
        double dx = ring[p][0], dy = ring[p][1];
        if (radius == 2) {
            double a = 2.0 * 3.14159265358979323846 * p / 8.0;
            dx = 2.0 * cos(a);
            dy = -2.0 * sin(a);
            /* the axis points are exact pixels; do not let cos(pi/2) = 6e-17 create a 1e-17 weight */
            if (fabs(dx - round(dx)) < 1e-9) dx = round(dx);
            if (fabs(dy - round(dy)) < 1e-9) dy = round(dy);
        }
        double fl_x = floor(dx), fl_y = floor(dy);
        t[p] = (lbp_tap){(int)fl_x, (int)fl_y, (float)(dx - fl_x), (float)(dy - fl_y)};
    }
}

void plx_lbp(const plx_img *gray, int k, int radius, uint8_t *codes) {
    PLX_ASSERT(radius == 1 || radius == 2);
    PLX_ASSERT(k >= 0 && k < gray->c);
    lbp_tap t[8];
    lbp_taps(radius, t);
    int w = gray->w, h = gray->h;
    const float *s = plx_img_chan(gray, k);
#define LBP_AT(X, Y) s[(size_t)PLX_CLAMP(Y, 0, h - 1) * w + PLX_CLAMP(X, 0, w - 1)]
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float c = s[(size_t)y * w + x];
            unsigned code = 0;
            for (int p = 0; p < 8; p++) {
                int X = x + t[p].ix, Y = y + t[p].iy;
                /* lerp form: equal corners give exactly that value, so flat areas compare as
                 * ties (bit set) rather than by rounding noise */
                float a = LBP_AT(X, Y), b = LBP_AT(X + 1, Y), cc = LBP_AT(X, Y + 1), d = LBP_AT(X + 1, Y + 1);
                float top = a + t[p].fx * (b - a), bot = cc + t[p].fx * (d - cc);
                float v = top + t[p].fy * (bot - top);
                code |= (unsigned)(v >= c) << p;
            }
            codes[(size_t)y * w + x] = (uint8_t)code;
        }
    }
#undef LBP_AT
}

int plx_lbp_uniform_table(uint8_t table[256]) {
    int next = 0;
    for (int code = 0; code < 256; code++) {
        unsigned rot = ((unsigned)code >> 1) | (((unsigned)code & 1u) << 7); /* circular neighbour */
        int transitions = 0;
        for (unsigned d = (unsigned)code ^ rot; d; d &= d - 1) transitions++;
        table[code] = (uint8_t)(transitions <= 2 ? next++ : PLX_LBP_UNIFORM_BINS - 1);
    }
    PLX_ASSERT(next == PLX_LBP_UNIFORM_BINS - 1);
    return PLX_LBP_UNIFORM_BINS;
}
