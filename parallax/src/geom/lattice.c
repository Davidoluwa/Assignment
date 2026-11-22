/* Lattice geometry, parity split, the weighted coordinate space and the flow stage's gauges. */
#include "geom/geom.h"

#include "la/la.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

plx_glat plx_glat_flow(void) {
    return (plx_glat){.cols = 17, .rows = 21, .spacing = 8.0, .x0 = -0.5, .y0 = -0.5, .cx = 63.5, .cy = 79.5};
}

static void glat_check(const plx_glat *L) {
    PLX_ASSERT(L->cols >= 3 && (L->cols & 1) && L->rows >= 2 && L->spacing > 0);
    /* mirror symmetry: the centre column must sit on x = cx */
    PLX_ASSERT(fabs(L->x0 + 0.5 * (L->cols - 1) * L->spacing - L->cx) < 1e-9 * L->spacing);
}

double plx_glat_X0(const plx_glat *L, int node) { return L->x0 + L->spacing * (node % L->cols) - L->cx; }
double plx_glat_Y0(const plx_glat *L, int node) { return L->y0 + L->spacing * (node / L->cols) - L->cy; }
int    plx_glat_mirror(const plx_glat *L, int node) {
    int r = node / L->cols, c = node % L->cols;
    return r * L->cols + (L->cols - 1 - c);
}

plx_core_shape plx_core_default(void) { return (plx_core_shape){.ax = 40.0, .ay = 50.0, .yc = 6.0}; }

void plx_core_mask(const plx_glat *L, plx_core_shape c, uint8_t *mask) {
    glat_check(L);
    for (int n = 0; n < plx_glat_n(L); n++) {
        double x = plx_glat_X0(L, n) / c.ax, y = (plx_glat_Y0(L, n) - c.yc) / c.ay;
        mask[n] = x * x + y * y <= 1.0 + 1e-12;
    }
}

bool plx_core_centre(const plx_glat *L, plx_core_shape c, int node) {
    return fabs(plx_glat_X0(L, node)) <= 0.3 * c.ax && fabs(plx_glat_Y0(L, node) - c.yc) <= 0.4 * c.ay;
}

void plx_parity_split(const plx_glat *L, const double *ux, const double *uy, double *E, double *O, double *V,
                      double *Q) {
    for (int n = 0; n < plx_glat_n(L); n++) {
        int m = plx_glat_mirror(L, n);
        if (E) E[n] = 0.5 * (ux[n] + ux[m]);
        if (O) O[n] = 0.5 * (ux[n] - ux[m]);
        if (V) V[n] = 0.5 * (uy[n] + uy[m]);
        if (Q) Q[n] = 0.5 * (uy[n] - uy[m]);
    }
}

/* ---- coordinate space ---------------------------------------------------------------------- */
void plx_gspace_init(plx_gspace *sp, const plx_glat *L, const float *node_weight, const uint8_t *core) {
    glat_check(L);
    memset(sp, 0, sizeof *sp);
    sp->lat = *L;
    int P = sp->P = plx_glat_n(L), axis = L->cols / 2;
    sp->omega = PLX_NEW(double, P);
    for (int n = 0; n < P; n++) {
        int m = plx_glat_mirror(L, n);
        /* symmetrise defensively: ω must be σ-invariant or the parity split is not orthogonal */
        double w = node_weight ? 0.5 * ((double)node_weight[n] + node_weight[m]) : 1.0;
        sp->omega[n] = (core[n] && core[m] && w > 0) ? w : 0.0;
        sp->wsum += sp->omega[n];
    }
    PLX_ASSERT(sp->wsum > 0);
    sp->enode = PLX_NEW(int, P);
    sp->onode = PLX_NEW(int, P);
    sp->esw = PLX_NEW(double, P);
    sp->osw = PLX_NEW(double, P);
    for (int n = 0; n < P; n++) {
        int c = n % L->cols;
        if (c < axis || sp->omega[n] <= 0) continue;
        double mult = c == axis ? 1.0 : 2.0;
        sp->enode[sp->ne] = n;
        sp->esw[sp->ne++] = sqrt(mult * sp->omega[n]);
        if (c > axis) {
            sp->onode[sp->no] = n;
            sp->osw[sp->no++] = sqrt(2.0 * sp->omega[n]);
        }
    }
    sp->e1 = PLX_NEW(double, sp->ne);
    sp->eY = PLX_NEW(double, sp->ne);
    sp->oX = PLX_NEW(double, sp->no);
    for (int k = 0; k < sp->ne; k++) {
        sp->e1[k] = sp->esw[k];
        sp->eY[k] = sp->esw[k] * plx_glat_Y0(L, sp->enode[k]);
    }
    for (int k = 0; k < sp->no; k++) sp->oX[k] = sp->osw[k] * plx_glat_X0(L, sp->onode[k]);
}

void plx_gspace_free(plx_gspace *sp) {
    free(sp->omega);
    free(sp->enode);
    free(sp->onode);
    free(sp->esw);
    free(sp->osw);
    free(sp->e1);
    free(sp->eY);
    free(sp->oX);
    memset(sp, 0, sizeof *sp);
}

void plx_gspace_even(const plx_gspace *sp, const double *f, double *y) {
    for (int k = 0; k < sp->ne; k++) {
        int n = sp->enode[k];
        y[k] = sp->esw[k] * 0.5 * (f[n] + f[plx_glat_mirror(&sp->lat, n)]);
    }
}

void plx_gspace_odd(const plx_gspace *sp, const double *f, double *y) {
    for (int k = 0; k < sp->no; k++) {
        int n = sp->onode[k];
        y[k] = sp->osw[k] * 0.5 * (f[n] - f[plx_glat_mirror(&sp->lat, n)]);
    }
}

void plx_gspace_even_field(const plx_gspace *sp, const double *y, double fill, double *f) {
    for (int n = 0; n < sp->P; n++) f[n] = fill;
    for (int k = 0; k < sp->ne; k++) {
        int n = sp->enode[k];
        f[n] = f[plx_glat_mirror(&sp->lat, n)] = y[k] / sp->esw[k];
    }
}

void plx_gspace_odd_field(const plx_gspace *sp, const double *y, double fill, double *f) {
    for (int n = 0; n < sp->P; n++) f[n] = fill;
    /* the axis column of an odd field is exactly zero wherever the core reaches it */
    for (int k = 0; k < sp->ne; k++)
        if (sp->enode[k] % sp->lat.cols == sp->lat.cols / 2) f[sp->enode[k]] = 0.0;
    for (int k = 0; k < sp->no; k++) {
        int n = sp->onode[k];
        f[n] = y[k] / sp->osw[k];
        f[plx_glat_mirror(&sp->lat, n)] = -f[n];
    }
}

void plx_gspace_remove_plane(const plx_gspace *sp, double *y, double coef[2]) {
    /* 2 x 2 normal equations of y ≈ α e1 + γ eY */
    double a11 = plx_vec_dot(sp->e1, sp->e1, sp->ne), a12 = plx_vec_dot(sp->e1, sp->eY, sp->ne);
    double a22 = plx_vec_dot(sp->eY, sp->eY, sp->ne);
    double b1 = plx_vec_dot(sp->e1, y, sp->ne), b2 = plx_vec_dot(sp->eY, y, sp->ne);
    double det = a11 * a22 - a12 * a12;
    PLX_ASSERT(det > 0);
    double al = (a22 * b1 - a12 * b2) / det, ga = (a11 * b2 - a12 * b1) / det;
    for (int k = 0; k < sp->ne; k++) y[k] -= al * sp->e1[k] + ga * sp->eY[k];
    if (coef) coef[0] = al, coef[1] = ga;
}

/* ---- flow-stage gauges --------------------------------------------------------------------- */
/* Gauge (1), as the flow stage applies it (align/lattice.h, plx_lat_sim_gauge): the similarity
 * S(p) = [a -b; b a] p + t (centred coordinates) minimising Σ g_n |S(p_n) - (p_n + u(p_n))|² over the
 * gauge points (weights g), then u <- S⁻¹(p + u) - p at every node. Linear least squares in
 * (a, b, tx, ty) with the 4 x 4 normal equations over the fields (1,0), (0,1), (X0,Y0), (-Y0,X0). */
void plx_gauge_similarity(const plx_glat *L, const double *g, float *u_i) {
    int P = plx_glat_n(L);
    const float *ux = u_i, *uy = u_i + P;
    double G[16] = {0}, rhs[4] = {0};
    for (int n = 0; n < P; n++) {
        if (g[n] <= 0) continue;
        double X = plx_glat_X0(L, n), Y = plx_glat_Y0(L, n), w = g[n];
        double qx = X + ux[n], qy = Y + uy[n];
        double f[4][2] = {{1, 0}, {0, 1}, {X, Y}, {-Y, X}};
        for (int i = 0; i < 4; i++) {
            rhs[i] += w * (f[i][0] * qx + f[i][1] * qy);
            for (int j = 0; j < 4; j++) G[4 * i + j] += w * (f[i][0] * f[j][0] + f[i][1] * f[j][1]);
        }
    }
    plx_mat Gm = plx_mat_from(4, 4, G), b = plx_mat_from(4, 1, rhs);
    PLX_ASSERT(plx_solve(&Gm, &b) == PLX_OK);
    double tx = PLX_M(&b, 0, 0), ty = PLX_M(&b, 1, 0), a = PLX_M(&b, 2, 0), bb = PLX_M(&b, 3, 0);
    double det = a * a + bb * bb;
    for (int n = 0; n < P; n++) {
        double X = plx_glat_X0(L, n), Y = plx_glat_Y0(L, n);
        double qx = X + ux[n] - tx, qy = Y + uy[n] - ty;
        u_i[n] = (float)((a * qx + bb * qy) / det - X); /* S⁻¹: [a b; -b a] / (a² + b²) */
        u_i[P + n] = (float)((-bb * qx + a * qy) / det - Y);
    }
    plx_mat_free(&Gm);
    plx_mat_free(&b);
}

void plx_gauge_box(const plx_glat *L, double x0, double x1, double y0, double y1, double *g) {
    for (int n = 0; n < plx_glat_n(L); n++) {
        double x = plx_glat_X0(L, n) + L->cx, y = plx_glat_Y0(L, n) + L->cy;
        g[n] = (x >= x0 - 1e-9 && x <= x1 + 1e-9 && y >= y0 - 1e-9 && y <= y1 + 1e-9) ? 1.0 : 0.0;
    }
}

/* Gauge (2). The mirror-symmetrised mean (x odd, y even) is the mean over the population and its
 * mirror twins, so subtracting it leaves the mean of E (and of Q) in place: yaw keeps its absolute
 * zero (the symmetric pose), while the mean of V, which contains -mean(sin φ cos θ)·z, is removed,
 * so pitch is relative to the population mean. */
void plx_gauge_mean(const plx_glat *L, float *u, int n) {
    int P = plx_glat_n(L);
    double *mu = PLX_NEW(double, 2 * P);
    for (int i = 0; i < n; i++)
        for (int k = 0; k < 2 * P; k++) mu[k] += u[(size_t)i * 2 * P + k];
    float *sym = PLX_NEW(float, 2 * P);
    for (int k = 0; k < P; k++) {
        int m = plx_glat_mirror(L, k);
        sym[k] = (float)(0.5 * (mu[k] - mu[m]) / n);
        sym[P + k] = (float)(0.5 * (mu[P + k] + mu[P + m]) / n);
    }
    for (int i = 0; i < n; i++)
        for (int k = 0; k < 2 * P; k++) u[(size_t)i * 2 * P + k] -= sym[k];
    free(mu);
    free(sym);
}
