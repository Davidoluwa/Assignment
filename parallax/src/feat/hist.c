/* Weighted cell-histogram descriptors over a rectangular grid. */
#include "feat/feat.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* The in-image part of the region, and the cell column of each of its image columns. */
typedef struct {
    int  xa, xb, ya, yb; /* clipped region [xa, xb) x [ya, yb) */
    int *cell_x;         /* cell column of image column xa + i */
} grid_walk;

static grid_walk walk_begin(const plx_cell_grid *g, int w, int h) {
    PLX_ASSERT(g->w > 0 && g->h > 0 && g->cells_x >= 1 && g->cells_y >= 1);
    PLX_ASSERT(g->cells_x <= g->w && g->cells_y <= g->h && g->eps >= 0.0f);
    grid_walk gw = {PLX_MAX(g->x0, 0), PLX_MIN(g->x0 + g->w, w), PLX_MAX(g->y0, 0), PLX_MIN(g->y0 + g->h, h), NULL};
    gw.cell_x = PLX_NEW(int, PLX_MAX(gw.xb - gw.xa, 1));
    for (int x = gw.xa; x < gw.xb; x++) gw.cell_x[x - gw.xa] = (int)((int64_t)(x - g->x0) * g->cells_x / g->w);
    return gw;
}

static int cell_row(const plx_cell_grid *g, int y) { return (int)((int64_t)(y - g->y0) * g->cells_y / g->h); }

static void normalise_cells(const plx_cell_grid *g, int bins, float *out) {
    if (g->norm == PLX_HNORM_NONE) return;
    for (int c = 0; c < g->cells_x * g->cells_y; c++) {
        float *v = out + (size_t)c * bins;
        double s = 0.0;
        for (int b = 0; b < bins; b++) s += g->norm == PLX_HNORM_L2 ? (double)v[b] * v[b] : fabs((double)v[b]);
        if (g->norm == PLX_HNORM_L2) s = sqrt(s);
        double inv = 1.0 / (s + g->eps);
        for (int b = 0; b < bins; b++) {
            double q = v[b] * inv;
            v[b] = (float)(g->norm == PLX_HNORM_L1_SQRT ? copysign(sqrt(fabs(q)), q) : q);
        }
    }
}

void plx_hist_labels(const uint8_t *labels, int w, int h, int bins, const float *weight,
                     const plx_cell_grid *g, float *out, float *cell_weight) {
    PLX_ASSERT(bins >= 1 && bins <= 256);
    grid_walk gw = walk_begin(g, w, h);
    int ncell = g->cells_x * g->cells_y;
    memset(out, 0, sizeof(float) * (size_t)ncell * bins);
    if (cell_weight) memset(cell_weight, 0, sizeof(float) * (size_t)ncell);
    for (int y = gw.ya; y < gw.yb; y++) {
        int row = cell_row(g, y) * g->cells_x;
        for (int x = gw.xa; x < gw.xb; x++) {
            size_t i = (size_t)y * w + x;
            PLX_ASSERT(labels[i] < bins);
            float wt = weight ? weight[i] : 1.0f;
            int c = row + gw.cell_x[x - gw.xa];
            out[(size_t)c * bins + labels[i]] += wt;
            if (cell_weight) cell_weight[c] += wt;
        }
    }
    free(gw.cell_x);
    normalise_cells(g, bins, out);
}

void plx_hist_channels(const plx_img *chan, const float *weight, const plx_cell_grid *g,
                       float *out, float *cell_weight) {
    int w = chan->w, bins = chan->c;
    size_t plane = (size_t)w * chan->h;
    grid_walk gw = walk_begin(g, w, chan->h);
    int ncell = g->cells_x * g->cells_y;
    memset(out, 0, sizeof(float) * (size_t)ncell * bins);
    if (cell_weight) memset(cell_weight, 0, sizeof(float) * (size_t)ncell);
    for (int y = gw.ya; y < gw.yb; y++) {
        int row = cell_row(g, y) * g->cells_x;
        for (int x = gw.xa; x < gw.xb; x++) {
            size_t i = (size_t)y * w + x;
            float wt = weight ? weight[i] : 1.0f;
            int c = row + gw.cell_x[x - gw.xa];
            float *o = out + (size_t)c * bins;
            for (int b = 0; b < bins; b++) o[b] += wt * chan->px[b * plane + i];
            if (cell_weight) cell_weight[c] += wt;
        }
    }
    free(gw.cell_x);
    normalise_cells(g, bins, out);
}
