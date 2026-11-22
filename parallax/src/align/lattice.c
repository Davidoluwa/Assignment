/* Lattice displacement fields: interpolation, refinement, composition, bending energy,
 * similarity gauge and fold checks. See lattice.h. */
#include "align/lattice.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

plx_lattice plx_lattice_make(int step) {
    PLX_ASSERT(step == 4 || step == 8 || step == 16 || step == 32);
    plx_lattice L = {PLX_CANON_W / step + 1, PLX_CANON_H / step + 1, step};
    return L;
}

void plx_lat_taps_at(const plx_lattice *L, double x, double y, plx_lat_taps *t) {
    double gx = (x - PLX_LAT_ORIGIN) / L->step, gy = (y - PLX_LAT_ORIGIN) / L->step;
    gx = PLX_CLAMP(gx, 0.0, (double)(L->nx - 1));
    gy = PLX_CLAMP(gy, 0.0, (double)(L->ny - 1));
    if (!(gx == gx)) gx = 0; /* NaN: any point of the lattice is as good as another */
    if (!(gy == gy)) gy = 0;
    int j = (int)gx, i = (int)gy;
    if (j > L->nx - 2) j = L->nx - 2;
    if (i > L->ny - 2) i = L->ny - 2;
    double fx = gx - j, fy = gy - i;
    int n = i * L->nx + j;
    t->n[0] = n;
    t->n[1] = n + 1;
    t->n[2] = n + L->nx;
    t->n[3] = n + L->nx + 1;
    t->w[0] = (float)((1 - fx) * (1 - fy));
    t->w[1] = (float)(fx * (1 - fy));
    t->w[2] = (float)((1 - fx) * fy);
    t->w[3] = (float)(fx * fy);
}

void plx_lat_eval(const plx_lattice *L, const float *u, double x, double y, double v[2]) {
    plx_lat_taps t;
    plx_lat_taps_at(L, x, y, &t);
    plx_lat_eval_taps(L, u, &t, v);
}

void plx_lat_resample(const plx_lattice *from, const float *u, const plx_lattice *to, float *out) {
    int N = plx_lat_nodes(to);
    for (int i = 0; i < to->ny; i++)
        for (int j = 0; j < to->nx; j++) {
            double v[2];
            plx_lat_eval(from, u, plx_lat_x(to, j), plx_lat_y(to, i), v);
            out[i * to->nx + j] = (float)v[0];
            out[N + i * to->nx + j] = (float)v[1];
        }
}

void plx_lat_compose(const plx_lattice *L, const float *d, const float *u, float *out) {
    PLX_ASSERT(out != d && out != u);
    int N = plx_lat_nodes(L);
    for (int i = 0; i < L->ny; i++)
        for (int j = 0; j < L->nx; j++) {
            int n = i * L->nx + j;
            double v[2];
            plx_lat_eval(L, u, plx_lat_x(L, j) + d[n], plx_lat_y(L, i) + d[N + n], v);
            out[n] = (float)(d[n] + v[0]);
            out[N + n] = (float)(d[N + n] + v[1]);
        }
}

void plx_lat_mirror(const plx_lattice *L, const float *u, float *out) {
    int N = plx_lat_nodes(L);
    for (int c = 0; c < 2; c++) {
        float s = c == 0 ? -1.0f : 1.0f;
        for (int i = 0; i < L->ny; i++)
            for (int j = 0; j <= (L->nx - 1) / 2; j++) {
                int a = c * N + i * L->nx + j, b = c * N + i * L->nx + (L->nx - 1 - j);
                float ua = u[a], ub = u[b];
                out[a] = s * ub;
                out[b] = s * ua;
            }
    }
}

/* ---- bending energy ---------------------------------------------------------------------
 * Every term is (weight / h²) * (Σ_t c_t u[n_t])² over a 3-tap or 4-tap stencil; the energy and
 * the matrix walk the same stencil list so they cannot disagree. */
typedef struct { int n[4], len; double c[4], wgt; } stencil;

static int bending_stencils(const plx_lattice *L, stencil *out) {
    int k = 0, nx = L->nx, ny = L->ny;
    double h2 = L->step * L->step;
    for (int i = 0; i < ny; i++)
        for (int j = 0; j + 2 < nx; j++)
            out[k++] = (stencil){{i * nx + j, i * nx + j + 1, i * nx + j + 2, 0}, 3, {1, -2, 1, 0}, 1 / h2};
    for (int i = 0; i + 2 < ny; i++)
        for (int j = 0; j < nx; j++)
            out[k++] = (stencil){{i * nx + j, (i + 1) * nx + j, (i + 2) * nx + j, 0}, 3, {1, -2, 1, 0}, 1 / h2};
    for (int i = 0; i + 1 < ny; i++)
        for (int j = 0; j + 1 < nx; j++)
            out[k++] = (stencil){{i * nx + j, i * nx + j + 1, (i + 1) * nx + j, (i + 1) * nx + j + 1}, 4,
                                 {1, -1, -1, 1}, 2 / h2};
    return k;
}

static int n_stencils(const plx_lattice *L) {
    return L->ny * (L->nx - 2) + (L->ny - 2) * L->nx + (L->ny - 1) * (L->nx - 1);
}

double plx_lat_bending_energy(const plx_lattice *L, const float *u) {
    stencil *s = PLX_NEW(stencil, n_stencils(L));
    int ns = bending_stencils(L, s), N = plx_lat_nodes(L);
    double e = 0;
    for (int c = 0; c < 2; c++)
        for (int k = 0; k < ns; k++) {
            double d = 0;
            for (int t = 0; t < s[k].len; t++) d += s[k].c[t] * u[c * N + s[k].n[t]];
            e += s[k].wgt * d * d;
        }
    free(s);
    return e;
}

void plx_lat_bending_matrix(const plx_lattice *L, plx_mat *R) {
    int N = plx_lat_nodes(L);
    PLX_ASSERT(R->rows == 2 * N && R->cols == 2 * N);
    memset(R->a, 0, sizeof(double) * (size_t)4 * N * N);
    stencil *s = PLX_NEW(stencil, n_stencils(L));
    int ns = bending_stencils(L, s);
    for (int c = 0; c < 2; c++)
        for (int k = 0; k < ns; k++)
            for (int a = 0; a < s[k].len; a++)
                for (int b = 0; b < s[k].len; b++)
                    PLX_M(R, c * N + s[k].n[a], c * N + s[k].n[b]) += s[k].wgt * s[k].c[a] * s[k].c[b];
    free(s);
}

void plx_lat_bending_apply(const plx_lattice *L, const float *u, double *out) {
    int N = plx_lat_nodes(L);
    memset(out, 0, sizeof(double) * 2 * N);
    stencil *s = PLX_NEW(stencil, n_stencils(L));
    int ns = bending_stencils(L, s);
    for (int c = 0; c < 2; c++)
        for (int k = 0; k < ns; k++) {
            double d = 0;
            for (int t = 0; t < s[k].len; t++) d += s[k].c[t] * u[c * N + s[k].n[t]];
            d *= s[k].wgt;
            for (int t = 0; t < s[k].len; t++) out[c * N + s[k].n[t]] += s[k].c[t] * d;
        }
    free(s);
}

/* ---- similarity gauge ------------------------------------------------------------------- */
plx_sim plx_lat_fit_sim(const plx_lattice *L, const float *u) {
    /* Closed-form weighted Procrustes without reflection, in coordinates about c0:
     * X = p - c0, Y = X + u(p); M = [a -b; b a] fits the centred clouds, then t = Ȳ - M X̄. */
    double sw = 0, mx[2] = {0}, my[2] = {0};
    enum { MAXP = 512 };
    double X[MAXP][2], Y[MAXP][2];
    int np = 0;
    for (double y = PLX_LAT_CORE_Y0; y <= PLX_LAT_CORE_Y1 + 1e-9; y += 8)
        for (double x = PLX_LAT_CORE_X0; x <= PLX_LAT_CORE_X1 + 1e-9; x += 8) {
            double v[2];
            plx_lat_eval(L, u, x, y, v);
            PLX_ASSERT(np < MAXP);
            X[np][0] = x - PLX_CANON_CX;
            X[np][1] = y - PLX_CANON_CY;
            Y[np][0] = X[np][0] + v[0];
            Y[np][1] = X[np][1] + v[1];
            for (int c = 0; c < 2; c++) mx[c] += X[np][c], my[c] += Y[np][c];
            sw += 1;
            np++;
        }
    for (int c = 0; c < 2; c++) mx[c] /= sw, my[c] /= sw;
    double sxx = 0, sa = 0, sb = 0;
    for (int k = 0; k < np; k++) {
        double x0 = X[k][0] - mx[0], x1 = X[k][1] - mx[1], y0 = Y[k][0] - my[0], y1 = Y[k][1] - my[1];
        sxx += x0 * x0 + x1 * x1;
        sa += x0 * y0 + x1 * y1;
        sb += x0 * y1 - x1 * y0;
    }
    double a = sa / sxx, b = sb / sxx;
    /* S(p) = M X + c0 + t, so S maps X̄ to Ȳ when t = Ȳ - M X̄ */
    return (plx_sim){a, b, my[0] - (a * mx[0] - b * mx[1]), my[1] - (b * mx[0] + a * mx[1])};
}

void plx_lat_csim_pullback(const plx_lattice *L, plx_sim S, float *u) {
    plx_sim Si = plx_csim_invert(S);
    int N = plx_lat_nodes(L);
    for (int i = 0; i < L->ny; i++)
        for (int j = 0; j < L->nx; j++) {
            int n = i * L->nx + j;
            double px = plx_lat_x(L, j), py = plx_lat_y(L, i), qx, qy;
            plx_csim_apply(Si, px + u[n], py + u[N + n], &qx, &qy);
            u[n] = (float)(qx - px);
            u[N + n] = (float)(qy - py);
        }
}

plx_sim plx_lat_sim_gauge(const plx_lattice *L, float *u) {
    plx_sim S = plx_lat_fit_sim(L, u);
    plx_lat_csim_pullback(L, S, u);
    return S;
}

/* ---- field statistics ------------------------------------------------------------------- */
double plx_lat_min_jdet(const plx_lattice *L, const float *u) {
    int N = plx_lat_nodes(L), nx = L->nx;
    double h = L->step, m = INFINITY;
    for (int i = 0; i + 1 < L->ny; i++)
        for (int j = 0; j + 1 < nx; j++) {
            double x0 = plx_lat_x(L, j), y0 = plx_lat_y(L, i);
            double s0 = (fmax(x0, PLX_LAT_CORE_X0) - x0) / h, s1 = (fmin(x0 + h, PLX_LAT_CORE_X1) - x0) / h;
            double t0 = (fmax(y0, PLX_LAT_CORE_Y0) - y0) / h, t1 = (fmin(y0 + h, PLX_LAT_CORE_Y1) - y0) / h;
            if (s1 < s0 || t1 < t0) continue; /* cell outside the core */
            int n = i * nx + j;
            double c[2][4];
            for (int k = 0; k < 2; k++) {
                c[k][0] = u[k * N + n];
                c[k][1] = u[k * N + n + 1];
                c[k][2] = u[k * N + n + nx];
                c[k][3] = u[k * N + n + nx + 1];
            }
            double ss[2] = {s0, s1}, tt[2] = {t0, t1};
            for (int a = 0; a < 2; a++)
                for (int b = 0; b < 2; b++) {
                    double s = ss[a], t = tt[b], g[2][2];
                    for (int k = 0; k < 2; k++) {
                        g[k][0] = ((c[k][1] - c[k][0]) * (1 - t) + (c[k][3] - c[k][2]) * t) / h;
                        g[k][1] = ((c[k][2] - c[k][0]) * (1 - s) + (c[k][3] - c[k][1]) * s) / h;
                    }
                    m = fmin(m, (1 + g[0][0]) * (1 + g[1][1]) - g[0][1] * g[1][0]);
                }
        }
    return m;
}

double plx_lat_max_abs(const plx_lattice *L, const float *u) {
    int N = plx_lat_nodes(L);
    double m = 0;
    for (int i = 0; i < L->ny; i++)
        for (int j = 0; j < L->nx; j++) {
            double x = plx_lat_x(L, j), y = plx_lat_y(L, i);
            if (x < PLX_LAT_CORE_X0 || x > PLX_LAT_CORE_X1 || y < PLX_LAT_CORE_Y0 || y > PLX_LAT_CORE_Y1) continue;
            int n = i * L->nx + j;
            m = fmax(m, hypot(u[n], u[N + n]));
        }
    return m;
}
