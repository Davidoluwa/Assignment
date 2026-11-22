#include "viz/viz.h"

#include <math.h>
#include <string.h>

plx_montage_opts plx_montage_defaults(void) {
    return (plx_montage_opts){.cols = 0, .pad = 4, .bg = {0.12f, 0.12f, 0.12f}, .fg = {1, 1, 1}, .text_scale = 1};
}

plx_img plx_viz_to_rgb(const plx_img *im) {
    if (im->c == 3) return plx_img_clone(im);
    PLX_ASSERT(im->c == 1);
    plx_img out = plx_img_new(im->w, im->h, 3);
    size_t n = (size_t)im->w * im->h;
    for (int k = 0; k < 3; k++) memcpy(out.px + k * n, im->px, sizeof(float) * n);
    return out;
}

plx_img plx_viz_montage(const plx_img *tiles, int n, const char *const *labels, const plx_montage_opts *o) {
    PLX_ASSERT(n > 0 && o->pad >= 0 && o->text_scale >= 1);
    int cw = 1, ch = 1, label_h = 0;
    for (int i = 0; i < n; i++) {
        PLX_ASSERT(tiles[i].c == 1 || tiles[i].c == 3);
        cw = PLX_MAX(cw, tiles[i].w);
        ch = PLX_MAX(ch, tiles[i].h);
        if (labels && labels[i]) {
            int tw, th;
            plx_text_size(labels[i], o->text_scale, &tw, &th);
            label_h = PLX_MAX(label_h, th + 2 * o->text_scale); /* small gap above the tile */
        }
    }
    int cols = o->cols > 0 ? PLX_MIN(o->cols, n) : (int)ceil(sqrt((double)n));
    int rows = (n + cols - 1) / cols, cell_h = ch + label_h;
    int W = cols * cw + (cols + 1) * o->pad, H = rows * cell_h + (rows + 1) * o->pad;
    plx_img out = plx_img_new(W, H, 3);
    size_t plane = (size_t)W * H;
    for (int k = 0; k < 3; k++)
        for (size_t i = 0; i < plane; i++) out.px[k * plane + i] = o->bg[k];
    for (int i = 0; i < n; i++) {
        const plx_img *t = &tiles[i];
        int cx = o->pad + (i % cols) * (cw + o->pad), cy = o->pad + (i / cols) * (cell_h + o->pad);
        int ox = cx + (cw - t->w) / 2, oy = cy + label_h + (ch - t->h) / 2;
        size_t tplane = (size_t)t->w * t->h;
        for (int k = 0; k < 3; k++) {
            const float *src = t->px + (t->c == 3 ? k * tplane : 0);
            for (int y = 0; y < t->h; y++)
                memcpy(out.px + k * plane + (size_t)(oy + y) * W + ox, src + (size_t)y * t->w, sizeof(float) * t->w);
        }
        if (labels && labels[i]) plx_draw_text(&out, cx, cy, labels[i], o->text_scale, o->fg);
    }
    return out;
}

plx_img plx_viz_upscale(const plx_img *im, int f, plx_upscale_mode mode) {
    PLX_ASSERT(f >= 1);
    int W = im->w * f, H = im->h * f;
    plx_img out = plx_img_new(W, H, im->c);
    for (int k = 0; k < im->c; k++) {
        const float *src = plx_img_chan(im, k);
        float *dst = plx_img_chan(&out, k);
        for (int Y = 0; Y < H; Y++)
            for (int X = 0; X < W; X++) {
                float v;
                if (mode == PLX_UP_NEAREST) {
                    v = src[(size_t)(Y / f) * im->w + X / f];
                } else {
                    float sx = PLX_CLAMP((X + 0.5f) / f - 0.5f, 0.0f, (float)(im->w - 1));
                    float sy = PLX_CLAMP((Y + 0.5f) / f - 0.5f, 0.0f, (float)(im->h - 1));
                    int x0 = PLX_MIN((int)sx, im->w - 1), y0 = PLX_MIN((int)sy, im->h - 1);
                    int x1 = PLX_MIN(x0 + 1, im->w - 1), y1 = PLX_MIN(y0 + 1, im->h - 1);
                    float fx = sx - x0, fy = sy - y0;
                    const float *r0 = src + (size_t)y0 * im->w, *r1 = src + (size_t)y1 * im->w;
                    /* Zero-weight taps are skipped so a NaN ("no data") neighbour does not
                     * bleed into pixels that sit exactly on a valid sample. */
                    float wt[4] = {(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy};
                    float tap[4] = {r0[x0], r0[x1], r1[x0], r1[x1]};
                    v = 0;
                    for (int q = 0; q < 4; q++)
                        if (wt[q] > 0) v += wt[q] * tap[q];
                }
                dst[(size_t)Y * W + X] = v;
            }
    }
    return out;
}

int plx_viz_save(const char *path, const plx_img *im) {
    int64_t dim[3] = {im->c, im->h, im->w};
    int rc = im->c == 1 ? plx_tensor_save(path, PLX_F32, 2, dim + 1, im->px)
                        : plx_tensor_save(path, PLX_F32, 3, dim, im->px);
    if (rc != PLX_OK) plx_log("viz: cannot write '%s'", path);
    return rc;
}
