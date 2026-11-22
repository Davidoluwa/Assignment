// plx-deps: geom align render viz feat la ops img core
#include "test.h"

#include "align/canon.h"
#include "align/flow.h"
#include "align/lattice.h"
#include "core/core.h"
#include "geom/relief.h"
#include "img/image.h"
#include "ops/ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

enum { NLAT = PLX_RL_LAT_NODES, DOF = 2 * NLAT };

static plx_img test_photo(uint64_t seed) {
    plx_rng r;
    plx_rng_seed(&r, seed);
    plx_img im = plx_img_new(PLX_CANON_IMG_SIDE, PLX_CANON_IMG_SIDE, 1);
    for (size_t k = 0; k < (size_t)im.w * im.h; k++) im.px[k] = (float)plx_rng_uniform(&r);
    plx_blur(&im, &im, 2.0f);
    return im;
}

/* A constant layer w = c is the lattice field u + (s c, 0): both renders must agree exactly
 * (up to float rounding), and w = NULL must be the plain render. */
TEST(render_relief_matches_lattice_shift) {
    plx_img g = test_photo(3);
    plx_lattice L = plx_lattice_make(PLX_FL_STEP);
    float u[DOF], u2[DOF];
    plx_rng r;
    plx_rng_seed(&r, 5);
    for (int k = 0; k < DOF; k++) u[k] = (float)(2 * plx_rng_uniform(&r) - 1);
    double s = 0.7, c = 1.9;
    memcpy(u2, u, sizeof u);
    for (int k = 0; k < NLAT; k++) u2[k] += (float)(s * c);
    plx_img w = plx_img_new(PLX_CANON_W, PLX_CANON_H, 1);
    plx_img_fill(&w, (float)c);
    plx_sim T = plx_sim_make(1.1, 0.05, 2, -3);
    plx_img a = plx_canon_new(1, 16), b = plx_canon_new(1, 16), p0 = plx_canon_new(1, 16), p1 = plx_canon_new(1, 16);
    plx_fl_render_gray_relief(&g, T, &L, u, s, &w, 16, PLX_INTERP_BILINEAR, &a, NULL);
    plx_fl_render_gray(&g, T, &L, u2, 16, PLX_INTERP_BILINEAR, &b, NULL);
    CHECK(plx_img_max_abs_diff(&a, &b) < 1e-5f);
    plx_fl_render_gray_relief(&g, T, &L, u, s, NULL, 16, PLX_INTERP_BILINEAR, &p0, NULL);
    plx_fl_render_gray(&g, T, &L, u, 16, PLX_INTERP_BILINEAR, &p1, NULL);
    CHECK(plx_img_max_abs_diff(&p0, &p1) == 0.0f);
    plx_img_free(&g), plx_img_free(&w), plx_img_free(&a), plx_img_free(&b), plx_img_free(&p0), plx_img_free(&p1);
}

/* The gauge leaves w orthogonal to 1, Y0 and v (weighted, over the core) and is idempotent. */
TEST(gauge_projects_out_plane_and_v) {
    float vl[NLAT];
    plx_lattice L = plx_lattice_make(PLX_FL_STEP);
    for (int k = 0; k < NLAT; k++) {
        double X = plx_lat_x(&L, k % L.nx) - 63.5, Y = plx_lat_y(&L, k / L.nx) - 79.5;
        vl[k] = (float)(1 - (X * X + Y * Y) / 3000);
    }
    plx_img v = plx_rl_v_dense(vl), w = plx_img_new(PLX_CANON_W, PLX_CANON_H, 1), wt = plx_img_new(PLX_CANON_W, PLX_CANON_H, 1);
    plx_rng r;
    plx_rng_seed(&r, 9);
    for (int y = 0; y < PLX_CANON_H; y++)
        for (int x = 0; x < PLX_CANON_W; x++) {
            PLX_PX(&w, x, y, 0) = (float)(0.3 + 0.01 * (y - 79.5) + 0.5 * PLX_PX(&v, x, y, 0) + 0.1 * plx_rng_normal(&r));
            PLX_PX(&wt, x, y, 0) = (float)(0.5 + plx_rng_uniform(&r));
        }
    plx_core_shape cs = plx_core_default();
    double c1[3], c2[3];
    plx_rl_gauge(&w, &v, &wt, cs, c1);
    CHECK_NEAR(c1[0], 0.3, 0.02);
    CHECK_NEAR(c1[1], 0.01, 0.001);
    CHECK_NEAR(c1[2], 0.5, 0.02);
    double d[3] = {0};
    for (int y = 0; y < PLX_CANON_H; y++)
        for (int x = 0; x < PLX_CANON_W; x++)
            if (plx_rl_in_core(cs, x, y)) {
                double g = PLX_PX(&wt, x, y, 0), f = PLX_PX(&w, x, y, 0);
                d[0] += g * f, d[1] += g * f * (y - 79.5), d[2] += g * f * PLX_PX(&v, x, y, 0);
            }
    for (int k = 0; k < 3; k++) CHECK(fabs(d[k]) < 1e-3 * (1 + fabs(d[k])) || fabs(d[k]) < 0.05);
    plx_rl_gauge(&w, &v, &wt, cs, c2);
    for (int k = 0; k < 3; k++) CHECK(fabs(c2[k]) < 1e-5);
    plx_img_free(&v), plx_img_free(&w), plx_img_free(&wt);
}

static plx_rl_params mini_params(void) {
    plx_cg_params cg = plx_cg_default();
    plx_rl_params p = plx_rl_default(&cg);
    p.n_stages = 2;
    p.stages[0] = (plx_rl_stage){.level = 1, .sigma = 4.0f, .iters = 3};
    p.stages[1] = (plx_rl_stage){.level = 0, .sigma = 2.0f, .iters = 3};
    return p;
}

/* Depth part (step 6 gauge, the estimate's support as weights) of a map. */
static plx_img depth_part(const plx_img *w, const plx_rl_synth *s, const plx_rl_result *res, const plx_rl_params *p) {
    plx_img v = plx_rl_v_dense(s->v_lat), g = plx_img_clone(w);
    plx_rl_gauge(&g, &v, &res->support, p->core, NULL);
    plx_img_free(&v);
    return g;
}

/* Correlation and RMS error of the recovered depth part against the true relief in the same gauge. */
static void score(const plx_rl_synth *s, const plx_rl_result *res, const plx_rl_params *p, double *corr, double *rms,
                  double *rms_true, double *rms_w) {
    plx_img wt = depth_part(&s->w_true, s, res, p), wg = depth_part(&res->w, s, res, p);
    plx_img zero = plx_img_new(PLX_CANON_W, PLX_CANON_H, 1);
    plx_rl_compare(&wg, &wt, p->core, 4.0f, corr, NULL, rms);
    plx_rl_compare(&wt, &zero, p->core, 4.0f, NULL, NULL, rms_true);
    plx_rl_compare(&wg, &zero, p->core, 4.0f, NULL, NULL, rms_w);
    plx_img_free(&wt), plx_img_free(&wg), plx_img_free(&zero);
}

/* The core test: a small rendered population with a known nose, sockets, brow, lips and chin on a
 * dome; given the dome only, w = 0 and the true s, the estimator recovers the relief. With s
 * shuffled across photos (the yaw-misalignment relation destroyed) it must recover nothing. */
TEST(synthetic_relief_recovered) {
    plx_rl_synth_opts so = plx_rl_synth_defaults();
    so.n = 240;
    plx_rl_synth s;
    plx_rl_synth_make(&so, &s);
    plx_rl_params p = mini_params();
    plx_rl_data d = {.src = &s.src, .T = s.T, .u = s.u, .s = s.s, .v_lat = s.v_lat};
    plx_rl_result res;
    plx_rl_estimate(&d, &p, NULL, NULL, false, &res);
    double corr, rms, rms_true, rms_w;
    score(&s, &res, &p, &corr, &rms, &rms_true, &rms_w);
    printf("  synthetic: corr %.3f, rms error %.3f px vs true relief rms %.3f px\n", corr, rms, rms_true);
    CHECK(corr > 0.92);
    CHECK(rms < 0.4 * rms_true);
    /* exactly mirror-even */
    double asym = 0;
    for (int y = 0; y < PLX_CANON_H; y++)
        for (int x = 0; x < PLX_CANON_W; x++) asym = fmax(asym, fabs(PLX_PX(&res.w, x, y, 0) - PLX_PX(&res.w, 127 - x, y, 0)));
    CHECK(asym < 1e-4);

    float *sp = PLX_NEW(float, s.n);
    int *perm = PLX_NEW(int, s.n);
    plx_rng r;
    plx_rng_seed(&r, 77);
    for (int i = 0; i < s.n; i++) perm[i] = i;
    plx_rng_shuffle(&r, perm, s.n);
    for (int i = 0; i < s.n; i++) sp[i] = s.s[perm[i]];
    plx_rl_data dp = d;
    dp.s = sp;
    plx_rl_result rp;
    plx_rl_estimate(&dp, &p, NULL, NULL, false, &rp);
    double c_perm, e_perm, t_perm, rms_perm;
    score(&s, &rp, &p, &c_perm, &e_perm, &t_perm, &rms_perm);
    /* (a permutation keeps ~1/sqrt(n) of the true s by chance, so a faint copy of the relief remains) */
    printf("  permuted s: depth rms %.3f px vs %.3f px with the true s (corr with the truth %.3f)\n", rms_perm, rms_w,
           c_perm);
    CHECK(rms_perm < 0.25 * rms_w);
    plx_rl_result_free(&rp);
    free(sp), free(perm);
    plx_rl_result_free(&res);
    plx_rl_synth_free(&s);
}

int main(void) {
    RUN(render_relief_matches_lattice_shift);
    RUN(gauge_projects_out_plane_and_v);
    RUN(synthetic_relief_recovered);
    return TEST_REPORT();
}
