#include "render/render.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Sub-pixel precision of the coverage test. 8 bits keeps snapping error (1/512 px) far
 * below anything visible while leaving the int64 edge products enormous headroom. */
#define SUB_BITS 8
#define SUB (1 << SUB_BITS)
/* Vertices further than this from the origin (in pixels) make the triangle be skipped:
 * |coord| < 2^28 sub-pixel units, so edge differences stay < 2^29 and every edge product
 * (and sum of two) fits comfortably in int64. */
#define COORD_LIMIT 1.0e6

static inline int64_t ceil_div_sub(int64_t v) { /* ceil(v / SUB) for any sign */
    return v >= 0 ? (v + SUB - 1) >> SUB_BITS : -((-v) >> SUB_BITS);
}
static inline int64_t floor_div_sub(int64_t v) { return v >= 0 ? v >> SUB_BITS : -((-v + SUB - 1) >> SUB_BITS); }

/* Top-left rule for a triangle with E > 0 (clockwise on screen, y down): walking the edge
 * from i to j, a "top" edge runs exactly horizontally to the right, a "left" edge runs up. */
static inline bool top_left(int64_t dx, int64_t dy) { return dy < 0 || (dy == 0 && dx > 0); }

typedef struct {
    double *zbuf;  /* [h*w], -inf = empty */
    int32_t *tri;  /* [h*w] */
    double *bary;  /* [h*w][3], mesh vertex order */
} zstate;

/* Rasterises one triangle whose (snapped) area is positive. a,b,c are the vertex indices in
 * raster order; perm maps raster order back to mesh order for the stored barycentrics. */
static void raster_tri(zstate *zs, int w, int h, int id, const double (*p)[3], const int64_t (*q)[2], int a, int b,
                       int c, const int perm[3]) {
    const int64_t X0 = q[a][0], Y0 = q[a][1], X1 = q[b][0], Y1 = q[b][1], X2 = q[c][0], Y2 = q[c][1];
    int64_t E = (X1 - X0) * (Y2 - Y0) - (Y1 - Y0) * (X2 - X0);
    int64_t xmin = PLX_MAX(ceil_div_sub(PLX_MIN(X0, PLX_MIN(X1, X2))), 0);
    int64_t xmax = PLX_MIN(floor_div_sub(PLX_MAX(X0, PLX_MAX(X1, X2))), w - 1);
    int64_t ymin = PLX_MAX(ceil_div_sub(PLX_MIN(Y0, PLX_MIN(Y1, Y2))), 0);
    int64_t ymax = PLX_MIN(floor_div_sub(PLX_MAX(Y0, PLX_MAX(Y1, Y2))), h - 1);
    if (xmin > xmax || ymin > ymax) return;

    /* Barycentric planes from the unsnapped positions (exact attribute interpolation); fall
     * back to the snapped ones if snapping changed the sign of a sliver's area. */
    double x0 = p[a][0], y0 = p[a][1], x1 = p[b][0], y1 = p[b][1], x2 = p[c][0], y2 = p[c][1];
    double Ed = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
    if (!(Ed > 0)) {
        x0 = X0 / (double)SUB; y0 = Y0 / (double)SUB; x1 = X1 / (double)SUB;
        y1 = Y1 / (double)SUB; x2 = X2 / (double)SUB; y2 = Y2 / (double)SUB;
        Ed = (double)E / ((double)SUB * SUB);
    }
    /* b_i(x, y) = A_i x + B_i y + C_i, from the edge opposite vertex i. */
    double A[3] = {-(y2 - y1) / Ed, -(y0 - y2) / Ed, -(y1 - y0) / Ed};
    double B[3] = {(x2 - x1) / Ed, (x0 - x2) / Ed, (x1 - x0) / Ed};
    double C[3] = {-(A[0] * x1 + B[0] * y1), -(A[1] * x2 + B[1] * y2), -(A[2] * x0 + B[2] * y0)};
    const double z[3] = {p[a][2], p[b][2], p[c][2]};

    /* Edge functions at the first pixel centre of the box, then stepped incrementally. */
    const int64_t ex[3][4] = {{X1, Y1, X2 - X1, Y2 - Y1}, {X2, Y2, X0 - X2, Y0 - Y2}, {X0, Y0, X1 - X0, Y1 - Y0}};
    int64_t row[3], bias[3], stepx[3], stepy[3];
    int64_t PX = xmin * SUB, PY = ymin * SUB;
    for (int e = 0; e < 3; e++) {
        int64_t xi = ex[e][0], yi = ex[e][1], dx = ex[e][2], dy = ex[e][3];
        row[e] = dx * (PY - yi) - dy * (PX - xi);
        bias[e] = top_left(dx, dy) ? 0 : -1;
        stepx[e] = -dy * SUB;
        stepy[e] = dx * SUB;
    }
    for (int64_t y = ymin; y <= ymax; y++) {
        int64_t e0 = row[0], e1 = row[1], e2 = row[2];
        for (int64_t x = xmin; x <= xmax; x++, e0 += stepx[0], e1 += stepx[1], e2 += stepx[2]) {
            if ((e0 + bias[0]) < 0 || (e1 + bias[1]) < 0 || (e2 + bias[2]) < 0) continue;
            double bl[3];
            for (int k = 0; k < 3; k++) bl[k] = A[k] * (double)x + B[k] * (double)y + C[k];
            double depth = bl[0] * z[0] + bl[1] * z[1] + bl[2] * z[2];
            size_t i = (size_t)y * w + (size_t)x;
            if (!(depth > zs->zbuf[i])) continue;
            zs->zbuf[i] = depth;
            zs->tri[i] = id;
            for (int k = 0; k < 3; k++) zs->bary[3 * i + perm[k]] = bl[k];
        }
        for (int e = 0; e < 3; e++) row[e] += stepy[e];
    }
}

void plx_render(const plx_mesh *m, const plx_camera *cam, const plx_raster_opts *o, plx_raster *out) {
    PLX_ASSERT(o->w > 0 && o->h > 0);
    int w = o->w, h = o->h;
    size_t n = (size_t)w * h;
    double (*p)[3] = plx_malloc(sizeof(double[3]) * (size_t)(m->nv ? m->nv : 1));
    int64_t (*q)[2] = plx_malloc(sizeof(int64_t[2]) * (size_t)(m->nv ? m->nv : 1));
    uint8_t *bad = PLX_NEW(uint8_t, m->nv);
    for (int i = 0; i < m->nv; i++) {
        plx_camera_project(cam, m->v + 3 * i, p[i]);
        bad[i] = !(fabs(p[i][0]) < COORD_LIMIT && fabs(p[i][1]) < COORD_LIMIT && isfinite(p[i][2]));
        if (!bad[i]) {
            q[i][0] = llround(p[i][0] * SUB);
            q[i][1] = llround(p[i][1] * SUB);
        }
    }
    zstate zs = {plx_malloc(sizeof(double) * n), plx_malloc(sizeof(int32_t) * n), PLX_NEW(double, 3 * n)};
    for (size_t i = 0; i < n; i++) { zs.zbuf[i] = -INFINITY; zs.tri[i] = -1; }

    static const int ident[3] = {0, 1, 2}, swapped[3] = {0, 2, 1};
    for (int t = 0; t < m->nt; t++) {
        const int *v = m->tri + 3 * t;
        PLX_ASSERT(v[0] >= 0 && v[0] < m->nv && v[1] >= 0 && v[1] < m->nv && v[2] >= 0 && v[2] < m->nv);
        if (bad[v[0]] || bad[v[1]] || bad[v[2]]) continue;
        int64_t E = (q[v[1]][0] - q[v[0]][0]) * (q[v[2]][1] - q[v[0]][1]) -
                    (q[v[1]][1] - q[v[0]][1]) * (q[v[2]][0] - q[v[0]][0]);
        if (E == 0) continue; /* degenerate on the sub-pixel grid: covers no pixel centre */
        if (E > 0) raster_tri(&zs, w, h, t, (const double(*)[3])p, (const int64_t(*)[2])q, v[0], v[1], v[2], ident);
        else if (o->cull == PLX_CULL_NONE)
            raster_tri(&zs, w, h, t, (const double(*)[3])p, (const int64_t(*)[2])q, v[0], v[2], v[1], swapped);
    }

    /* Resolve: attributes are interpolated once per pixel, after visibility is final. */
    memset(out, 0, sizeof *out);
    out->w = w;
    out->h = h;
    out->depth = plx_img_new(w, h, 1);
    out->mask = PLX_NEW(uint8_t, n);
    if (m->na > 0) out->attr = plx_img_new(w, h, m->na);
    for (size_t i = 0; i < n; i++) {
        int t = zs.tri[i];
        out->mask[i] = t >= 0;
        out->depth.px[i] = t >= 0 ? (float)zs.zbuf[i] : NAN;
        for (int k = 0; k < m->na; k++) {
            float val = o->bg;
            if (t >= 0) {
                const int *v = m->tri + 3 * t;
                const double *bl = zs.bary + 3 * i;
                val = (float)(bl[0] * m->attr[(size_t)v[0] * m->na + k] + bl[1] * m->attr[(size_t)v[1] * m->na + k] +
                              bl[2] * m->attr[(size_t)v[2] * m->na + k]);
            }
            out->attr.px[k * n + i] = val;
        }
    }
    if (o->want_ids) {
        out->tri = zs.tri;
        zs.tri = NULL;
        out->bary = plx_img_new(w, h, 3);
        for (size_t i = 0; i < n; i++)
            for (int k = 0; k < 3; k++) out->bary.px[k * n + i] = (float)zs.bary[3 * i + k];
    }
    free(zs.zbuf);
    free(zs.tri);
    free(zs.bary);
    free(p);
    free(q);
    free(bad);
}

void plx_raster_free(plx_raster *r) {
    plx_img_free(&r->attr);
    plx_img_free(&r->depth);
    plx_img_free(&r->bary);
    free(r->mask);
    free(r->tri);
    memset(r, 0, sizeof *r);
}
