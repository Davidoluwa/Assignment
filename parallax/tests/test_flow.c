// plx-deps: align feat la ops img core
#include "test.h"

#include "align/canon.h"
#include "align/congeal.h"
#include "align/flow.h"
#include "align/lattice.h"
#include "core/core.h"
#include "img/image.h"
#include "la/la.h"
#include "ops/ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

enum { SIDE = PLX_CANON_IMG_SIDE, DOF8 = 2 * 17 * 21 };

static double urand(plx_rng *r, double a) { return a * (2 * plx_rng_uniform(r) - 1); }

static void random_field(plx_rng *r, const plx_lattice *L, double amp, float *u) {
    for (int k = 0; k < plx_lat_dof(L); k++) u[k] = (float)urand(r, amp);
}

/* A smooth field on the 8-px lattice: a few broad Gaussian bumps (σ 25-40 px) of random
 * displacement plus a random affine part, sampled at the nodes, then its similarity part
 * removed (so it is in the flow's gauge). amp is the bump amplitude in px. */
static void smooth_field(plx_rng *r, double amp, float *u8) {
    plx_lattice L = plx_lattice_make(8);
    int N = plx_lat_nodes(&L);
    double bump[4][5], aff[2][2];
    for (int b = 0; b < 4; b++) {
        bump[b][0] = 63.5 + urand(r, 40);
        bump[b][1] = 79.5 + urand(r, 50);
        bump[b][2] = 25 + 15 * plx_rng_uniform(r);
        bump[b][3] = urand(r, amp);
        bump[b][4] = urand(r, amp);
    }
    for (int c = 0; c < 2; c++) aff[c][0] = urand(r, 0.04), aff[c][1] = urand(r, 0.04);
    for (int n = 0; n < N; n++) {
        double x = plx_lat_x(&L, n % L.nx), y = plx_lat_y(&L, n / L.nx), v[2] = {0, 0};
        for (int c = 0; c < 2; c++) v[c] = aff[c][0] * (x - 63.5) + aff[c][1] * (y - 79.5);
        for (int b = 0; b < 4; b++) {
            double g = exp(-((x - bump[b][0]) * (x - bump[b][0]) + (y - bump[b][1]) * (y - bump[b][1])) /
                           (2 * bump[b][2] * bump[b][2]));
            v[0] += bump[b][3] * g;
            v[1] += bump[b][4] * g;
        }
        u8[n] = (float)v[0];
        u8[N + n] = (float)v[1];
    }
    plx_lat_sim_gauge(&L, u8);
}

/* test_align's blob pattern, spread wider so it textures the whole face core: random Gaussian
 * blobs around the crop centre; mirror=true makes it exactly symmetric about x = 124.5. */
static plx_img blob_pattern(uint64_t seed, int n_blobs, bool mirror) {
    plx_rng r;
    plx_rng_seed(&r, seed);
    plx_img im = plx_img_new(SIDE, SIDE, 1);
    plx_img_fill(&im, 0.5f);
    for (int b = 0; b < n_blobs; b++) {
        double bx = 124.5 + urand(&r, 60), by = 128.5 + urand(&r, 75);
        double sg = 2.5 + 5.0 * plx_rng_uniform(&r), amp = urand(&r, 0.25);
        for (int m = 0; m < (mirror ? 2 : 1); m++) {
            double cx = m ? 249.0 - bx : bx;
            int x0 = (int)(cx - 4 * sg), x1 = (int)(cx + 4 * sg) + 1, y0 = (int)(by - 4 * sg), y1 = (int)(by + 4 * sg) + 1;
            for (int y = PLX_MAX(y0, 0); y <= PLX_MIN(y1, SIDE - 1); y++)
                for (int x = PLX_MAX(x0, 0); x <= PLX_MIN(x1, SIDE - 1); x++) {
                    double d2 = (x - cx) * (x - cx) + (y - by) * (y - by);
                    PLX_PX(&im, x, y, 0) += (float)(amp * exp(-d2 / (2 * sg * sg)));
                }
        }
    }
    return im;
}

/* Solve p + u(p) = q for p by fixed-point iteration (u small and smooth). */
static void invert_field(const plx_lattice *L, const float *u, double qx, double qy, double *px, double *py) {
    double x = qx, y = qy;
    for (int it = 0; it < 50; it++) {
        double v[2];
        plx_lat_eval(L, u, x, y, v);
        x = qx - v[0];
        y = qy - v[1];
    }
    *px = x;
    *py = y;
}

/* The photo in which `base` (laid out at the identity similarity) appears under the full warp
 * W(p) = T(p + u(p)): I(W(p)) = base(T_id(p)). */
static plx_img warp_pattern(const plx_img *base, plx_sim T, const plx_lattice *L, const float *u) {
    plx_img out = plx_img_new(SIDE, SIDE, 1);
    plx_sim id = plx_sim_identity();
    for (int y = 0; y < SIDE; y++)
        for (int x = 0; x < SIDE; x++) {
            double qx, qy, px, py, bx, by;
            plx_sim_unapply(T, x, y, &qx, &qy);
            invert_field(L, u, qx, qy, &px, &py);
            plx_sim_apply(id, px, py, &bx, &by);
            PLX_PX(&out, x, y, 0) = plx_sample1(base, 0, PLX_INTERP_BICUBIC, (float)bx, (float)by);
        }
    return out;
}

static bool in_core(double x, double y) {
    return x >= PLX_LAT_CORE_X0 && x <= PLX_LAT_CORE_X1 && y >= PLX_LAT_CORE_Y0 && y <= PLX_LAT_CORE_Y1;
}

/* Largest canonical-px distance between two full warps at the core nodes of the 8-px lattice,
 * measured through T_ref⁻¹ (so image px are converted to canonical px); *rms (optional) gets the
 * RMS over those nodes. */
static double warp_dist(plx_sim Ta, const float *ua, plx_sim Tb, const float *ub, plx_sim Tref, double *rms) {
    plx_lattice L = plx_lattice_make(8);
    double m = 0, s2 = 0;
    int c = 0;
    for (int i = 0; i < L.ny; i++)
        for (int j = 0; j < L.nx; j++) {
            double x = plx_lat_x(&L, j), y = plx_lat_y(&L, i);
            if (!in_core(x, y)) continue;
            double va[2], vb[2], ax, ay, bx, by, cax, cay, cbx, cby;
            plx_lat_eval(&L, ua, x, y, va);
            plx_lat_eval(&L, ub, x, y, vb);
            plx_sim_apply(Ta, x + va[0], y + va[1], &ax, &ay);
            plx_sim_apply(Tb, x + vb[0], y + vb[1], &bx, &by);
            plx_sim_unapply(Tref, ax, ay, &cax, &cay);
            plx_sim_unapply(Tref, bx, by, &cbx, &cby);
            double d = hypot(cax - cbx, cay - cby);
            m = fmax(m, d);
            s2 += d * d;
            c++;
        }
    if (rms) *rms = sqrt(s2 / c);
    return m;
}

TEST(lattice_interp_refine) {
    plx_lattice L8 = plx_lattice_make(8), L16 = plx_lattice_make(16), L32 = plx_lattice_make(32);
    CHECK(L8.nx == 17 && L8.ny == 21 && L16.nx == 9 && L16.ny == 11 && L32.nx == 5 && L32.ny == 6);
    CHECK(plx_lat_x(&L8, 16) == 127.5 && plx_lat_y(&L8, 20) == 159.5 && plx_lat_y(&L32, 5) == 159.5);
    /* σ-symmetric node set */
    for (int j = 0; j < L8.nx; j++) CHECK(plx_canon_mirror_x(plx_lat_x(&L8, j)) == plx_lat_x(&L8, L8.nx - 1 - j));
    plx_rng r;
    plx_rng_seed(&r, 3);
    /* a bilinear function a + bx + cy + dxy is reproduced exactly inside every cell */
    float u[DOF8];
    double co[2][4] = {{0.3, 0.02, -0.01, 1e-4}, {-0.7, -0.015, 0.03, -2e-4}};
    int N = plx_lat_nodes(&L8);
    for (int i = 0; i < L8.ny; i++)
        for (int j = 0; j < L8.nx; j++)
            for (int c = 0; c < 2; c++) {
                double x = plx_lat_x(&L8, j), y = plx_lat_y(&L8, i);
                u[c * N + i * L8.nx + j] = (float)(co[c][0] + co[c][1] * x + co[c][2] * y);
            }
    double worst = 0;
    for (int t = 0; t < 200; t++) {
        double x = urand(&r, 64) + 63.5, y = urand(&r, 80) + 79.5, v[2];
        plx_lat_eval(&L8, u, x, y, v);
        for (int c = 0; c < 2; c++) worst = fmax(worst, fabs(v[c] - (co[c][0] + co[c][1] * x + co[c][2] * y)));
    }
    CHECK(worst < 1e-5);
    /* clamped outside the lattice */
    double va[2], vb[2];
    plx_lat_eval(&L8, u, -12.0, 170.0, va);
    plx_lat_eval(&L8, u, -0.5, 159.5, vb);
    CHECK(va[0] == vb[0] && va[1] == vb[1]);
    /* nested refinement 32 -> 16 -> 8 is exact */
    float c32[2 * 5 * 6], c16[2 * 9 * 11], c8[DOF8];
    random_field(&r, &L32, 3, c32);
    plx_lat_resample(&L32, c32, &L16, c16);
    plx_lat_resample(&L16, c16, &L8, c8);
    worst = 0;
    for (int t = 0; t < 500; t++) {
        double x = urand(&r, 70) + 63.5, y = urand(&r, 90) + 79.5, a[2], b[2];
        plx_lat_eval(&L32, c32, x, y, a);
        plx_lat_eval(&L8, c8, x, y, b);
        worst = fmax(worst, fmax(fabs(a[0] - b[0]), fabs(a[1] - b[1])));
    }
    CHECK(worst < 1e-5);
    printf("  refinement max error %.2e px\n", worst);
}

TEST(bending_energy) {
    for (int step = 8; step <= 32; step *= 2) {
        plx_lattice L = plx_lattice_make(step);
        int N = plx_lat_nodes(&L), dof = 2 * N;
        double h = step;
        float *u = PLX_NEW(float, dof);
        plx_mat R = plx_mat_new(dof, dof);
        plx_lat_bending_matrix(&L, &R);
        /* every affine field is in the null space (energy relative to its size) */
        for (int i = 0; i < L.ny; i++)
            for (int j = 0; j < L.nx; j++) {
                double x = plx_lat_x(&L, j), y = plx_lat_y(&L, i);
                u[i * L.nx + j] = (float)(1.5 + 0.08 * x - 0.03 * y);
                u[N + i * L.nx + j] = (float)(-2 + 0.02 * x + 0.05 * y);
            }
        CHECK(plx_lat_bending_energy(&L, u) < 1e-9);
        double *Ru = PLX_NEW(double, dof);
        plx_lat_bending_apply(&L, u, Ru);
        double mx = 0;
        for (int k = 0; k < dof; k++) mx = fmax(mx, fabs(Ru[k]));
        CHECK(mx < 1e-6);
        /* u_x = c x²: Δxx = 2 c h² at each of (nx - 2) ny positions, Δyy = Δxy = 0 */
        double c = 0.01;
        memset(u, 0, sizeof(float) * dof);
        for (int i = 0; i < L.ny; i++)
            for (int j = 0; j < L.nx; j++) u[i * L.nx + j] = (float)(c * plx_lat_x(&L, j) * plx_lat_x(&L, j));
        double want = 4 * c * c * h * h * (L.nx - 2) * L.ny;
        CHECK_NEAR(plx_lat_bending_energy(&L, u), want, 1e-4 * want);
        /* u_y = c x y: only Δxy = c h², weight 2, over (nx - 1)(ny - 1) cells: exactly the
         * continuous 2 ∫ u_xy² over the lattice rectangle */
        memset(u, 0, sizeof(float) * dof);
        for (int i = 0; i < L.ny; i++)
            for (int j = 0; j < L.nx; j++) u[N + i * L.nx + j] = (float)(c * plx_lat_x(&L, j) * plx_lat_y(&L, i));
        want = 2 * c * c * 128.0 * 160.0;
        CHECK_NEAR(plx_lat_bending_energy(&L, u), want, 1e-4 * want);
        /* matrix, quadratic form and sparse apply agree on a random field; R is symmetric */
        plx_rng r;
        plx_rng_seed(&r, (uint64_t)step);
        random_field(&r, &L, 1, u);
        plx_lat_bending_apply(&L, u, Ru);
        double q = 0, maxd = 0, asym = 0;
        for (int a = 0; a < dof; a++) {
            double s = 0;
            for (int b = 0; b < dof; b++) {
                s += PLX_M(&R, a, b) * u[b];
                asym = fmax(asym, fabs(PLX_M(&R, a, b) - PLX_M(&R, b, a)));
            }
            q += u[a] * s;
            maxd = fmax(maxd, fabs(s - Ru[a]));
        }
        CHECK(asym == 0);
        CHECK(maxd < 1e-9);
        CHECK_NEAR(q, plx_lat_bending_energy(&L, u), 1e-9 * q);
        /* σ-symmetric: E(mirror u) = E(u) */
        float *m = PLX_NEW(float, dof);
        plx_lat_mirror(&L, u, m);
        CHECK_NEAR(plx_lat_bending_energy(&L, m), plx_lat_bending_energy(&L, u), 1e-9 * q);
        free(m);
        free(Ru);
        free(u);
        plx_mat_free(&R);
    }
}

TEST(compose_affine) {
    /* d(p) = A p + b and u(p) = C p + e are affine, so out(p) = d(p) + u(p + d(p)) is affine and
     * exact wherever p + d(p) stays on the lattice */
    plx_lattice L = plx_lattice_make(8);
    int N = plx_lat_nodes(&L);
    float d[DOF8], u[DOF8], out[DOF8];
    double A[4] = {0.03, -0.01, 0.02, -0.04}, b[2] = {1.2, -0.8}, C[4] = {-0.05, 0.02, 0.01, 0.03}, e[2] = {-2, 1.5};
    for (int i = 0; i < L.ny; i++)
        for (int j = 0; j < L.nx; j++) {
            double x = plx_lat_x(&L, j) - 63.5, y = plx_lat_y(&L, i) - 79.5;
            int n = i * L.nx + j;
            d[n] = (float)(A[0] * x + A[1] * y + b[0]);
            d[N + n] = (float)(A[2] * x + A[3] * y + b[1]);
            u[n] = (float)(C[0] * x + C[1] * y + e[0]);
            u[N + n] = (float)(C[2] * x + C[3] * y + e[1]);
        }
    plx_lat_compose(&L, d, u, out);
    double worst = 0;
    for (int i = 2; i < L.ny - 2; i++)
        for (int j = 2; j < L.nx - 2; j++) {
            double x = plx_lat_x(&L, j) - 63.5, y = plx_lat_y(&L, i) - 79.5;
            int n = i * L.nx + j;
            double dx = A[0] * x + A[1] * y + b[0], dy = A[2] * x + A[3] * y + b[1];
            double wx = dx + C[0] * (x + dx) + C[1] * (y + dy) + e[0];
            double wy = dy + C[2] * (x + dx) + C[3] * (y + dy) + e[1];
            worst = fmax(worst, fmax(fabs(out[n] - wx), fabs(out[N + n] - wy)));
        }
    CHECK(worst < 1e-4);
    /* the composition is what it claims for the warp: sampling C(p) = I(T(p + u(p))) at
     * p + d(p) equals I(T(p + out(p))) at the nodes, for a smooth non-affine u too */
    plx_rng r;
    plx_rng_seed(&r, 4);
    smooth_field(&r, 4, u);
    plx_lat_compose(&L, d, u, out);
    worst = 0;
    for (int n = 0; n < N; n++) {
        double x = plx_lat_x(&L, n % L.nx), y = plx_lat_y(&L, n / L.nx);
        if (!in_core(x, y)) continue;
        double v[2];
        plx_lat_eval(&L, u, x + d[n], y + d[N + n], v);
        worst = fmax(worst, fabs(x + d[n] + v[0] - (x + out[n])) + fabs(y + d[N + n] + v[1] - (y + out[N + n])));
    }
    CHECK(worst < 1e-4);
}

TEST(similarity_gauge) {
    plx_lattice L = plx_lattice_make(8);
    int N = plx_lat_nodes(&L);
    plx_rng r;
    plx_rng_seed(&r, 5);
    for (int t = 0; t < 10; t++) {
        /* a pure similarity field is moved entirely into T */
        plx_sim S0 = plx_sim_make(1 + urand(&r, 0.05), urand(&r, 0.05), urand(&r, 3), urand(&r, 3));
        float u[DOF8], u0[DOF8];
        for (int n = 0; n < N; n++) {
            double x = plx_lat_x(&L, n % L.nx), y = plx_lat_y(&L, n / L.nx), sx, sy;
            plx_csim_apply(S0, x, y, &sx, &sy);
            u[n] = (float)(sx - x);
            u[N + n] = (float)(sy - y);
        }
        plx_sim S = plx_lat_sim_gauge(&L, u);
        CHECK(fabs(S.a - S0.a) + fabs(S.b - S0.b) < 1e-6 && fabs(S.tx - S0.tx) + fabs(S.ty - S0.ty) < 1e-4);
        double mx = 0;
        for (int k = 0; k < 2 * N; k++) mx = fmax(mx, fabs(u[k]));
        CHECK(mx < 1e-4);
        /* similarity + smooth non-similarity: W unchanged at the nodes and anywhere inside the
         * lattice, nothing similar left, and the non-similar part kept */
        smooth_field(&r, 5, u0);
        float ug[DOF8];
        for (int n = 0; n < N; n++) {
            double x = plx_lat_x(&L, n % L.nx), y = plx_lat_y(&L, n / L.nx), sx, sy;
            plx_csim_apply(S0, x + u0[n], y + u0[N + n], &sx, &sy);
            u[n] = (float)(sx - x);
            u[N + n] = (float)(sy - y);
        }
        memcpy(ug, u, sizeof u);
        plx_sim T = plx_sim_make(1.1, 0.1, 3, -4);
        plx_sim Sg = plx_lat_sim_gauge(&L, ug);
        plx_sim T2 = plx_sim_compose(T, Sg);
        double worst = 0;
        for (int k = 0; k < 300; k++) {
            double x = 63.5 + urand(&r, 64), y = 79.5 + urand(&r, 80), va[2], vb[2], ax, ay, bx, by;
            if (k < N) x = plx_lat_x(&L, k % L.nx), y = plx_lat_y(&L, k / L.nx);
            plx_lat_eval(&L, u, x, y, va);
            plx_lat_eval(&L, ug, x, y, vb);
            plx_sim_apply(T, x + va[0], y + va[1], &ax, &ay);
            plx_sim_apply(T2, x + vb[0], y + vb[1], &bx, &by);
            worst = fmax(worst, hypot(ax - bx, ay - by));
        }
        CHECK(worst < 1e-4);
        plx_sim S2 = plx_lat_fit_sim(&L, ug);
        CHECK(fabs(S2.a - 1) + fabs(S2.b) < 1e-6 && fabs(S2.tx) + fabs(S2.ty) < 1e-4);
        double dev = 0;
        for (int k = 0; k < 2 * N; k++) dev = fmax(dev, fabs(ug[k] - u0[k]));
        CHECK(dev < 1e-3);
        /* the gauge commutes with the mirror: gauge(mirror u) = mirror(gauge u), S -> twin(S) */
        float um[DOF8], umg[DOF8];
        plx_lat_mirror(&L, u, um);
        plx_sim Sm = plx_lat_sim_gauge(&L, um);
        plx_lat_mirror(&L, ug, umg);
        plx_sim St = plx_sim_twin(Sg);
        CHECK(fabs(Sm.a - St.a) + fabs(Sm.b - St.b) + fabs(Sm.tx - St.tx) + fabs(Sm.ty - St.ty) < 1e-5);
        dev = 0;
        for (int k = 0; k < 2 * N; k++) dev = fmax(dev, fabs(um[k] - umg[k]));
        CHECK(dev < 1e-4);
    }
    /* fold detection: identity has det 1; a field compressing x by 2 over the core has det 0.5 */
    float u[DOF8] = {0};
    CHECK_NEAR(plx_lat_min_jdet(&L, u), 1.0, 1e-12);
    for (int n = 0; n < N; n++) u[n] = (float)(-0.5 * (plx_lat_x(&L, n % L.nx) - 63.5));
    CHECK_NEAR(plx_lat_min_jdet(&L, u), 0.5, 1e-6);
    u[8 * L.nx + 8] = 20; /* a 20 px kink folds the cell to its left */
    CHECK(plx_lat_min_jdet(&L, u) < 0);
}

/* Single-pattern template. Its population variance is 0, which would make the Huber test
 * (residual relative to σ²) meaningless, so it gets a constant stand-in variance: uniform
 * weights, residuals measured against a plausible feature noise level. */
static plx_cg_template single_template(const plx_img *base, const plx_cg_params *p) {
    plx_cg_accum acc;
    plx_cg_accum_init(&acc, p);
    plx_cg_view v;
    plx_cg_render(base, plx_sim_identity(), p, &v);
    plx_cg_accum_add(&acc, &v);
    plx_cg_view_free(&v);
    plx_cg_template t;
    plx_cg_accum_finish(&acc, p, &t, NULL);
    plx_cg_accum_free(&acc);
    for (int l = 0; l < t.levels; l++) plx_img_fill(&t.var[l], 0.02f);
    return t;
}

static const plx_fl_stage TEST_SCHED[] = {
    {32, 2, 1.0f, 0.1f, 0.3f, 10, 1}, {16, 1, 0.5f, 0.1f, 0.3f, 10, 1}, {16, 1, 0, 0.1f, 0.3f, 10, 1},
    {16, 0, 0, 0.1f, 1, 10, 1},       {16, 0, 0, 0.1f, 1, 10, 1},       {8, 0, 0, 0.3f, 1, 10, 0},
    {8, 0, 0, 0.3f, 1, 10, 0},        {8, 0, 0, 0.3f, 1, 10, 0},
};

/* Same schedule with every round re-rendering (damped by the proximal terms, so with four more
 * 8-px rounds): exactly deformation-equivariant, so it recovers a synthetic warp to the
 * interpolation floor. */
static plx_fl_model test_model_rerender(const plx_img *base);

static plx_fl_model test_model(const plx_img *base) {
    plx_fl_model m = {0};
    m.p = plx_fl_default();
    m.p.cg.levels = 3;
    m.p.cg.eps_frac = 1.0f;
    m.p.cg.interp = PLX_INTERP_BICUBIC; /* the synthetic truth is exact: keep interpolation blur out of it */
    m.n_stages = (int)(sizeof TEST_SCHED / sizeof TEST_SCHED[0]);
    memcpy(m.stages, TEST_SCHED, sizeof TEST_SCHED);
    m.tmpl = single_template(base, &m.p.cg);
    return m;
}

static plx_fl_model test_model_rerender(const plx_img *base) {
    plx_fl_model m = test_model(base);
    for (int s = 0; s < m.n_stages; s++) m.stages[s].rerender = 1;
    for (int k = 0; k < 4; k++) m.stages[m.n_stages + k] = m.stages[m.n_stages - 1]; /* damped: needs rounds */
    m.n_stages += 4;
    m.p.damp = 0.2f; /* the identity damping is for real faces' large first steps; the pattern has none */
    return m;
}

/* Recovery of known smooth warps of a textured pattern. The all-re-render schedule must reach
 * the interpolation floor; the production-style schedule (sampled 8-px rounds) carries the
 * strain bias of its fine increment (these fields are strongly strained: |∇u| up to ~0.15) and
 * gets a looser bound. */
static void recovery_trials(plx_fl_model *m, double *worst_rms, double *worst_max) {
    plx_img base = blob_pattern(11, 160, true);
    plx_cg_template_free(&m->tmpl);
    m->tmpl = single_template(&base, &m->p.cg);
    plx_fl_fitter f;
    plx_fl_fitter_init(&f, m);
    plx_lattice L = plx_lattice_make(8);
    plx_rng r;
    plx_rng_seed(&r, 8);
    *worst_rms = *worst_max = 0;
    for (int trial = 0; trial < 3; trial++) {
        float ut[DOF8], u[DOF8];
        smooth_field(&r, 6, ut);
        double rms = 0;
        for (int k = 0; k < DOF8; k++) rms += ut[k] * ut[k] / (DOF8 / 2);
        plx_sim Tt = plx_sim_make(1 + urand(&r, 0.08), urand(&r, 0.1), urand(&r, 5), urand(&r, 5));
        plx_img I = warp_pattern(&base, Tt, &L, ut);
        plx_sim T;
        plx_fl_fit fit = plx_fl_fit_image(&f, &I, Tt, &T, u);
        double e_rms, e_max = warp_dist(T, u, Tt, ut, Tt, &e_rms);
        printf("  trial %d: true field rms %.2f px; error over core nodes rms %.3f, max %.3f px; cost %.4f\n", trial,
               sqrt(rms), e_rms, e_max, fit.cost);
        *worst_rms = fmax(*worst_rms, e_rms);
        *worst_max = fmax(*worst_max, e_max);
        plx_img_free(&I);
    }
    plx_fl_fitter_free(&f);
    plx_img_free(&base);
}

TEST(recovers_known_warp) {
    plx_img base = blob_pattern(11, 160, true);
    double wr, wm;
    printf("  all rounds re-render:\n");
    plx_fl_model m = test_model_rerender(&base);
    recovery_trials(&m, &wr, &wm);
    CHECK(wr < 0.04);
    CHECK(wm < 0.1);
    plx_fl_model_free(&m);
    printf("  8-px rounds sampled:\n");
    m = test_model(&base);
    recovery_trials(&m, &wr, &wm);
    CHECK(wr < 0.12);
    CHECK(wm < 0.5);
    plx_fl_model_free(&m);
    plx_img_free(&base);
}

TEST(mirror_equivariance) {
    plx_img base = blob_pattern(17, 160, true);
    plx_fl_model m = test_model(&base);
    plx_fl_fitter f;
    plx_fl_fitter_init(&f, &m);
    plx_lattice L = plx_lattice_make(8);
    plx_rng r;
    plx_rng_seed(&r, 21);
    /* an asymmetric photo: the symmetric pattern under an asymmetric warp, plus asymmetric blobs */
    float ut[DOF8], u[DOF8], uf[DOF8], um[DOF8];
    smooth_field(&r, 6, ut);
    plx_sim Tt = plx_sim_make(1.05, 0.08, 4, -3);
    plx_img I = warp_pattern(&base, Tt, &L, ut), extra = blob_pattern(99, 20, false), If = plx_img_new(SIDE, SIDE, 1);
    for (size_t k = 0; k < (size_t)SIDE * SIDE; k++) I.px[k] += 0.5f * (extra.px[k] - 0.5f);
    plx_flip_h(&I, &If);
    plx_sim T0 = plx_sim_make(1.02, 0.05, 2, -1), T, Tf;
    plx_fl_fit_image(&f, &I, T0, &T, u);
    plx_fl_fit_image(&f, &If, plx_sim_twin(T0), &Tf, uf);
    plx_lat_mirror(&L, u, um);
    double dev = 0;
    for (int k = 0; k < DOF8; k++) dev = fmax(dev, fabs(uf[k] - um[k]));
    plx_sim Tw = plx_sim_twin(T);
    double dT = fabs(Tf.a - Tw.a) + fabs(Tf.b - Tw.b) + fabs(Tf.tx - Tw.tx) + fabs(Tf.ty - Tw.ty);
    printf("  flipped fit vs twin of the fit: field %.2e px, similarity %.2e\n", dev, dT);
    CHECK(dev < 1e-3);
    CHECK(dT < 1e-3);
    /* and the render itself is equivariant pixel for pixel */
    plx_cg_view v, vf;
    plx_fl_render(&I, T, &L, u, &m.p.cg, &v);
    plx_fl_render(&If, Tw, &L, um, &m.p.cg, &vf);
    plx_img g = plx_img_new(v.gray.w, v.gray.h, 1), fm = plx_img_new(v.feat.lev[1].w, v.feat.lev[1].h, m.p.cg.feat.k);
    plx_flip_h(&v.gray, &g);
    CHECK(plx_img_max_abs_diff(&g, &vf.gray) < 1e-4f);
    plx_orient_mirror(&v.feat.lev[1], &m.p.cg.feat, &fm);
    CHECK(plx_img_max_abs_diff(&fm, &vf.feat.lev[1]) < 1e-3f);
    plx_img_free(&g);
    plx_img_free(&fm);
    plx_cg_view_free(&v);
    plx_cg_view_free(&vf);
    plx_img_free(&I);
    plx_img_free(&If);
    plx_img_free(&extra);
    plx_fl_fitter_free(&f);
    plx_fl_model_free(&m);
    plx_img_free(&base);
}

TEST(model_save_load) {
    plx_img base = blob_pattern(5, 60, true);
    plx_fl_model m = test_model(&base), m2;
    m.p.weighting = PLX_FL_W_PLAIN;
    m.p.lambda0 = 1e-4f;
    char dir[] = "/tmp/plx_test_flow_modelXXXXXX";
    CHECK(mkdtemp(dir) != NULL);
    CHECK(plx_fl_model_save(&m, dir) == PLX_OK);
    CHECK(plx_fl_model_load(dir, &m2) == PLX_OK);
    CHECK(m2.n_stages == m.n_stages && memcmp(m2.stages, m.stages, sizeof(plx_fl_stage) * m.n_stages) == 0);
    CHECK(m2.p.weighting == PLX_FL_W_PLAIN && m2.p.lambda0 == m.p.lambda0 && m2.p.cg.levels == 3);
    CHECK(m2.p.cg.eps_frac == m.p.cg.eps_frac && m2.p.cg.feat.sigma_n == m.p.cg.feat.sigma_n);
    for (int l = 0; l < m.tmpl.levels; l++) {
        CHECK(plx_img_max_abs_diff(&m.tmpl.mean[l], &m2.tmpl.mean[l]) == 0);
        CHECK(plx_img_max_abs_diff(&m.tmpl.var[l], &m2.tmpl.var[l]) == 0);
    }
    char cmd[256];
    snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
    CHECK(system(cmd) == 0);
    plx_fl_model_free(&m);
    plx_fl_model_free(&m2);
    plx_img_free(&base);
}

/* Miniature population: one symmetric pattern under known similarities and known random smooth
 * fields, plus noise. With T started at the truth and u = 0, the recovered full warps must
 * equal the true ones up to ONE common reparametrisation of the canonical frame (the gauge):
 * W_true⁻¹(W_i(p)) - p is the same field for every i. */
TEST(population_synthetic) {
    enum { N = 24 };
    plx_img base = blob_pattern(23, 160, true);
    plx_fl_params p = plx_fl_default();
    p.cg.levels = 3;
    plx_lattice L = plx_lattice_make(8);
    plx_rng r;
    plx_rng_seed(&r, 9);
    uint8_t *px = PLX_NEW(uint8_t, (size_t)N * SIDE * SIDE);
    plx_sim Tt[N], T[N];
    float *ut = PLX_NEW(float, (size_t)N * DOF8), *u = PLX_NEW(float, (size_t)N * DOF8);
    for (int i = 0; i < N; i++) {
        smooth_field(&r, 6, ut + (size_t)i * DOF8);
        Tt[i] = plx_sim_make(1 + urand(&r, 0.06), urand(&r, 0.08), urand(&r, 4), urand(&r, 4));
        plx_img I = warp_pattern(&base, Tt[i], &L, ut + (size_t)i * DOF8);
        for (size_t k = 0; k < (size_t)SIDE * SIDE; k++) {
            double v = I.px[k] + 0.01 * plx_rng_normal(&r);
            px[(size_t)i * SIDE * SIDE + k] = (uint8_t)lrint(PLX_CLAMP(v, 0.0, 1.0) * 255);
        }
        plx_img_free(&I);
        T[i] = Tt[i];
    }
    plx_cg_source src = {px, N, SIDE, SIDE};
    /* the test schedule, stiffer (a template learnt from 24 noisy images is itself noisy, and a
     * soft field chases that noise: rms deviation 0.26 px at λ = 0.03, 0.17 at 0.3, 0.11 at 1),
     * with its last stage repeated: the template needs rounds to sharpen */
    enum { EXTRA = 6, NS = sizeof TEST_SCHED / sizeof TEST_SCHED[0] + EXTRA };
    plx_fl_stage sched[NS];
    memcpy(sched, TEST_SCHED, sizeof TEST_SCHED);
    for (int k = NS - EXTRA; k < NS; k++) sched[k] = sched[NS - EXTRA - 1];
    for (int k = 0; k < NS; k++) sched[k].lambda = 1.0f;
    int ns = NS;
    plx_fl_round rounds[NS + 2];
    plx_fl_quality q[N];
    plx_cg_template t;
    plx_fl_congeal(&src, &p, sched, ns, NULL, T, u, &t, rounds, q, NULL);

    /* m_i(p) = W_true_i⁻¹(W_i(p)) - p at the core nodes, canonical px */
    int Nn = plx_lat_nodes(&L);
    double *m = PLX_NEW(double, (size_t)N * 2 * Nn), mean[2 * 17 * 21] = {0};
    for (int i = 0; i < N; i++)
        for (int n = 0; n < Nn; n++) {
            double x = plx_lat_x(&L, n % L.nx), y = plx_lat_y(&L, n / L.nx), v[2], ix, iy, cx, cy, qx, qy;
            plx_lat_eval(&L, u + (size_t)i * DOF8, x, y, v);
            plx_sim_apply(T[i], x + v[0], y + v[1], &ix, &iy);
            plx_sim_unapply(Tt[i], ix, iy, &cx, &cy);
            invert_field(&L, ut + (size_t)i * DOF8, cx, cy, &qx, &qy);
            m[(size_t)i * 2 * Nn + n] = qx - x;
            m[(size_t)i * 2 * Nn + Nn + n] = qy - y;
            mean[n] += (qx - x) / N;
            mean[Nn + n] += (qy - y) / N;
        }
    double dev = 0, dev2 = 0, gauge = 0;
    int cnt = 0;
    for (int n = 0; n < Nn; n++) {
        if (!in_core(plx_lat_x(&L, n % L.nx), plx_lat_y(&L, n / L.nx))) continue;
        gauge = fmax(gauge, hypot(mean[n], mean[Nn + n]));
        for (int i = 0; i < N; i++) {
            double d = hypot(m[(size_t)i * 2 * Nn + n] - mean[n], m[(size_t)i * 2 * Nn + Nn + n] - mean[Nn + n]);
            dev = fmax(dev, d);
            dev2 += d * d;
            cnt++;
        }
    }
    printf("  population: deviation from the common gauge rms %.3f, max %.3f px (gauge field up to %.2f px)\n",
           sqrt(dev2 / cnt), dev, gauge);
    CHECK(sqrt(dev2 / cnt) < 0.15);
    CHECK(dev < 0.6);
    CHECK(rounds[ns + 1].core_var < 0.6 * rounds[0].core_var);
    for (int l = 0; l < t.levels; l++) {
        plx_img mir = plx_img_new(t.mean[l].w, t.mean[l].h, t.K);
        plx_orient_mirror(&t.mean[l], &p.cg.feat, &mir);
        CHECK(plx_img_max_abs_diff(&mir, &t.mean[l]) < 1e-6f);
        plx_img_free(&mir);
    }
    for (int i = 0; i < N; i++) CHECK(q[i].step < 0.2 && q[i].min_jdet > 0.5);
    for (int k = 0; k < ns + 2; k++) plx_img_free(&rounds[k].mean_gray);
    plx_cg_template_free(&t);
    free(m);
    free(px);
    free(ut);
    free(u);
    plx_img_free(&base);
}

int main(void) {
    RUN(lattice_interp_refine);
    RUN(bending_energy);
    RUN(compose_affine);
    RUN(similarity_gauge);
    RUN(model_save_load);
    RUN(recovers_known_warp);
    RUN(mirror_equivariance);
    RUN(population_synthetic);
    return TEST_REPORT();
}
