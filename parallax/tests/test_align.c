// plx-deps: align feat ops img core
#include "test.h"

#include "align/canon.h"
#include "align/congeal.h"
#include "core/core.h"
#include "img/image.h"
#include "ops/ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

enum { SIDE = PLX_CANON_IMG_SIDE };

static plx_sim random_sim(plx_rng *r, double ds, double drot_deg, double dt) {
    double s = 1.0 + ds * (2 * plx_rng_uniform(r) - 1);
    double rho = drot_deg * M_PI / 180.0 * (2 * plx_rng_uniform(r) - 1);
    return plx_sim_make(s, rho, dt * (2 * plx_rng_uniform(r) - 1), dt * (2 * plx_rng_uniform(r) - 1));
}

static double sim_dist(plx_sim A, plx_sim B) {
    return fabs(A.a - B.a) + fabs(A.b - B.b) + fabs(A.tx - B.tx) + fabs(A.ty - B.ty);
}

/* A smooth, face-sized, textured test pattern on a SIDE x SIDE image: random Gaussian blobs of
 * both signs inside an ellipse around the crop centre. mirror=true reflects every blob about
 * x = 124.5, making the pattern exactly symmetric (as a symmetric template requires). */
static plx_img blob_pattern(uint64_t seed, int n_blobs, bool mirror) {
    plx_rng r;
    plx_rng_seed(&r, seed);
    plx_img im = plx_img_new(SIDE, SIDE, 1);
    plx_img_fill(&im, 0.5f);
    for (int b = 0; b < n_blobs; b++) {
        double bx = 124.5 + 45 * (2 * plx_rng_uniform(&r) - 1), by = 128.5 + 60 * (2 * plx_rng_uniform(&r) - 1);
        double sg = 2.5 + 5.0 * plx_rng_uniform(&r), amp = 0.25 * (2 * plx_rng_uniform(&r) - 1);
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

/* The photo in which `base` (laid out at the identity similarity) appears under T: canonical
 * point p sits at T(p) in the result and at T_id(p) in base. */
static plx_img transform_pattern(const plx_img *base, plx_sim T) {
    plx_img out = plx_img_new(SIDE, SIDE, 1);
    plx_sim id = plx_sim_identity();
    for (int y = 0; y < SIDE; y++)
        for (int x = 0; x < SIDE; x++) {
            double cx, cy, bx, by;
            plx_sim_unapply(T, x, y, &cx, &cy);
            plx_sim_apply(id, cx, cy, &bx, &by);
            PLX_PX(&out, x, y, 0) = plx_sample1(base, 0, PLX_INTERP_BICUBIC, (float)bx, (float)by);
        }
    return out;
}

TEST(sim_algebra) {
    plx_rng r;
    plx_rng_seed(&r, 1);
    for (int t = 0; t < 100; t++) {
        plx_sim A = random_sim(&r, 0.3, 30, 20), B = random_sim(&r, 0.3, 30, 20), C = random_sim(&r, 0.3, 30, 20);
        double px = 128 * plx_rng_uniform(&r), py = 160 * plx_rng_uniform(&r);
        /* image apply / unapply round trip */
        double ix, iy, qx, qy;
        plx_sim_apply(A, px, py, &ix, &iy);
        plx_sim_unapply(A, ix, iy, &qx, &qy);
        CHECK_NEAR(qx, px, 1e-9);
        CHECK_NEAR(qy, py, 1e-9);
        /* compose agrees with applying in sequence (A image, B canonical) */
        double bx, by, cx, cy, dx, dy;
        plx_csim_apply(B, px, py, &bx, &by);
        plx_sim_apply(A, bx, by, &cx, &cy);
        plx_sim_apply(plx_sim_compose(A, B), px, py, &dx, &dy);
        CHECK_NEAR(dx, cx, 1e-9);
        CHECK_NEAR(dy, cy, 1e-9);
        /* canonical inverse, associativity */
        CHECK(sim_dist(plx_sim_compose(B, plx_csim_invert(B)), plx_sim_identity()) < 1e-12);
        CHECK(sim_dist(plx_sim_compose(plx_csim_invert(B), B), plx_sim_identity()) < 1e-12);
        CHECK(sim_dist(plx_sim_compose(plx_sim_compose(A, B), C), plx_sim_compose(A, plx_sim_compose(B, C))) < 1e-10);
        /* scale / angle decomposition */
        plx_sim M = plx_sim_make(1.23, 0.3, 4, -5);
        CHECK_NEAR(plx_sim_scale(M), 1.23, 1e-12);
        CHECK_NEAR(plx_sim_angle(M), 0.3, 1e-12);
        /* twin of the image similarity is flip o T o σ, pointwise */
        double tx, ty, sx, sy;
        plx_sim_apply(plx_sim_twin(A), px, py, &tx, &ty);
        plx_sim_apply(A, plx_canon_mirror_x(px), py, &sx, &sy);
        CHECK_NEAR(tx, (SIDE - 1) - sx, 1e-9);
        CHECK_NEAR(ty, sy, 1e-9);
    }
    /* the affine form maps an Ω+ pixel to T(q - margin) */
    plx_sim T = plx_sim_make(1.1, -0.2, 3, 7);
    float Af[6];
    plx_sim_to_affine(T, 16, Af);
    double ex, ey;
    plx_sim_apply(T, 10 - 16, 20 - 16, &ex, &ey);
    CHECK_NEAR(Af[0] * 10 + Af[1] * 20 + Af[2], ex, 1e-4);
    CHECK_NEAR(Af[3] * 10 + Af[4] * 20 + Af[5], ey, 1e-4);
}

/* The twin rule against actual pixels: warping the flipped photo through twin(T) gives the
 * mirror image of warping the photo through T. */
TEST(twin_rule_on_pixels) {
    plx_img I = blob_pattern(7, 80, false), If = plx_img_new(SIDE, SIDE, 1);
    plx_flip_h(&I, &If);
    plx_sim T = plx_sim_make(1.15, 0.12, -6.3, 4.1);
    for (int m = 0; m < 2; m++) {
        plx_interp interp = m ? PLX_INTERP_BICUBIC : PLX_INTERP_BILINEAR;
        plx_img a = plx_canon_new(1, 16), b = plx_canon_new(1, 16), af = plx_canon_new(1, 16);
        plx_canon_warp(&I, T, 16, interp, &a, NULL);
        plx_canon_warp(&If, plx_sim_twin(T), 16, interp, &b, NULL);
        plx_flip_h(&a, &af);
        CHECK(plx_img_max_abs_diff(&af, &b) < 1e-4f);
        plx_img_free(&a);
        plx_img_free(&b);
        plx_img_free(&af);
    }
    plx_img_free(&I);
    plx_img_free(&If);
}

/* A blob placed at image point T(p) lands at canonical p (Ω+ pixel p + margin). */
TEST(canon_warp_places_features) {
    plx_sim T = plx_sim_make(0.9, -0.25, 8, -5);
    double px = 40, py = 60, ix, iy;
    plx_sim_apply(T, px, py, &ix, &iy);
    plx_img I = plx_img_new(SIDE, SIDE, 1);
    for (int y = 0; y < SIDE; y++)
        for (int x = 0; x < SIDE; x++)
            PLX_PX(&I, x, y, 0) = (float)exp(-((x - ix) * (x - ix) + (y - iy) * (y - iy)) / (2 * 3.0 * 3.0));
    int m = 16;
    plx_img C = plx_canon_new(1, m), valid = plx_canon_new(1, m);
    plx_canon_warp(&I, T, m, PLX_INTERP_BICUBIC, &C, &valid);
    double s = 0, sx = 0, sy = 0;
    for (int y = 0; y < C.h; y++)
        for (int x = 0; x < C.w; x++) {
            double v = PLX_PX(&C, x, y, 0);
            s += v;
            sx += v * x;
            sy += v * y;
        }
    CHECK_NEAR(sx / s - m, px, 0.02);
    CHECK_NEAR(sy / s - m, py, 0.02);
    /* the whole of Ω+ is inside the photo at this T; a huge shift leaves none of it */
    double vs = 0;
    for (size_t i = 0; i < (size_t)valid.w * valid.h; i++) vs += valid.px[i];
    CHECK_NEAR(vs, (double)valid.w * valid.h, 0.5);
    plx_canon_warp(&I, plx_sim_make(1, 0, 400, 0), m, PLX_INTERP_BILINEAR, &C, &valid);
    vs = 0;
    for (size_t i = 0; i < (size_t)valid.w * valid.h; i++) vs += valid.px[i];
    CHECK(vs == 0);
    {
        plx_img crop = plx_canon_crop(&C, m);
        CHECK(crop.w == PLX_CANON_W && crop.h == PLX_CANON_H);
        CHECK(PLX_PX(&crop, 5, 7, 0) == PLX_PX(&C, 5 + m, 7 + m, 0));
        plx_img_free(&crop);
    }
    plx_img_free(&I);
    plx_img_free(&C);
    plx_img_free(&valid);
}

/* Template of a single symmetric pattern (accumulated once, so variance is 0 and every weight
 * is the floor's: a plain least-squares fit). */
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
    return t;
}

TEST(ic_recovers_similarity) {
    plx_cg_params p = plx_cg_default();
    p.levels = 3;
    p.eps_frac = 1.0f;
    plx_img base = blob_pattern(11, 120, true);
    plx_cg_template t = single_template(&base, &p);
    /* the template of a symmetric pattern is symmetric (and the mirror is exact) */
    plx_img mir = plx_img_new(t.mean[0].w, t.mean[0].h, t.K);
    plx_orient_mirror(&t.mean[0], &p.feat, &mir);
    CHECK(plx_img_max_abs_diff(&mir, &t.mean[0]) < 1e-5f);
    plx_img_free(&mir);

    plx_cg_stage st[3] = {{2, 0.5f, 15, 0}, {1, 0, 15, 0}, {0, 0, 15, 0}};
    plx_cg_ref refs[3];
    for (int s = 0; s < 3; s++) refs[s] = plx_cg_ref_build(&t, &p, st[s].level, st[s].blur);
    plx_rng r;
    plx_rng_seed(&r, 5);
    double worst_t = 0, worst_s = 0, worst_r = 0;
    for (int trial = 0; trial < 6; trial++) {
        plx_sim Ttrue = random_sim(&r, 0.15, 12, 10);
        plx_img I = transform_pattern(&base, Ttrue);
        /* start from a perturbed guess: +-6% scale, +-6 degrees, +-5 px */
        plx_sim T0 = plx_sim_compose(Ttrue, random_sim(&r, 0.06, 6, 5));
        plx_cg_fit fit;
        plx_sim T = plx_cg_align_image(&I, T0, refs, st, 3, &p, &fit, NULL);
        worst_t = fmax(worst_t, hypot(T.tx - Ttrue.tx, T.ty - Ttrue.ty));
        worst_s = fmax(worst_s, fabs(plx_sim_scale(T) / plx_sim_scale(Ttrue) - 1));
        worst_r = fmax(worst_r, fabs(plx_sim_angle(T) - plx_sim_angle(Ttrue)) * 180 / M_PI);
        CHECK(fit.converged);
        plx_img_free(&I);
    }
    CHECK(worst_t < 0.1);
    CHECK(worst_s < 0.002);
    CHECK(worst_r < 0.1);
    printf("  IC worst errors: %.4f px, %.4f %%, %.4f deg\n", worst_t, 100 * worst_s, worst_r);
    for (int s = 0; s < 3; s++) plx_cg_ref_free(&refs[s]);
    plx_cg_template_free(&t);
    plx_img_free(&base);
}

/* Miniature congealing: one symmetric pattern under known random similarities plus noise. The
 * recovered T_i must equal T_true_i o G for one common G (the gauge), and the template must be
 * exactly symmetric. */
TEST(congeal_synthetic_population) {
    enum { N = 48 };
    plx_cg_params p = plx_cg_default();
    p.levels = 3;
    plx_img base = blob_pattern(23, 120, true);
    plx_rng r;
    plx_rng_seed(&r, 9);
    uint8_t *px = PLX_NEW(uint8_t, (size_t)N * SIDE * SIDE);
    plx_sim Ttrue[N], T[N];
    for (int i = 0; i < N; i++) {
        Ttrue[i] = random_sim(&r, 0.08, 6, 6);
        plx_img I = transform_pattern(&base, Ttrue[i]);
        for (size_t k = 0; k < (size_t)SIDE * SIDE; k++) {
            double v = I.px[k] + 0.02 * plx_rng_normal(&r);
            px[(size_t)i * SIDE * SIDE + k] = (uint8_t)lrint(PLX_CLAMP(v, 0.0, 1.0) * 255);
        }
        plx_img_free(&I);
        T[i] = plx_sim_identity();
    }
    plx_cg_source src = {px, N, SIDE, SIDE};
    plx_cg_stage sched[] = {{2, 1.0f, 10, 0.05f}, {2, 0.5f, 10, 0}, {2, 0, 10, 0}, {1, 0, 10, 0},
                            {1, 0, 10, 0},       {0, 0, 10, 0},    {0, 0, 10, 0}};
    int ns = (int)(sizeof sched / sizeof sched[0]);
    plx_cg_round rounds[sizeof sched / sizeof sched[0] + 2];
    plx_cg_quality q[N];
    plx_cg_template t;
    plx_cg_congeal(&src, &p, sched, ns, T, &t, rounds, q);

    /* G_i = T_true_i^-1 o T_i should be one common similarity */
    double mean[4] = {0}, G[N][4];
    for (int i = 0; i < N; i++) {
        plx_sim g = plx_sim_compose(plx_csim_invert(Ttrue[i]), T[i]);
        double v[4] = {log(plx_sim_scale(g)), plx_sim_angle(g), g.tx, g.ty};
        for (int j = 0; j < 4; j++) {
            G[i][j] = v[j];
            mean[j] += v[j] / N;
        }
    }
    double dev[4] = {0};
    for (int i = 0; i < N; i++)
        for (int j = 0; j < 4; j++) dev[j] = fmax(dev[j], fabs(G[i][j] - mean[j]));
    printf("  congeal max deviation from common gauge: scale %.4f %%, rot %.4f deg, tx %.4f px, ty %.4f px\n",
           100 * dev[0], dev[1] * 180 / M_PI, dev[2], dev[3]);
    CHECK(dev[0] < 0.005);
    CHECK(dev[1] * 180 / M_PI < 0.25);
    CHECK(dev[2] < 0.25 && dev[3] < 0.25);
    /* symmetric base pattern: the template's tx / rotation gauge is pinned, so G is ~ identity
     * in those components */
    CHECK(fabs(mean[1]) * 180 / M_PI < 0.25);
    CHECK(fabs(mean[2]) < 0.3);
    /* gauge fix: mean log scale 0, mean ty 0 */
    double ls = 0, ty = 0;
    for (int i = 0; i < N; i++) {
        ls += log(plx_sim_scale(T[i])) / N;
        ty += T[i].ty / N;
    }
    CHECK(fabs(ls) < 0.01);
    CHECK(fabs(ty) < 0.5);
    /* variance fell, template symmetric at every level */
    CHECK(rounds[ns + 1].core_var < 0.5 * rounds[0].core_var);
    for (int l = 0; l < t.levels; l++) {
        plx_img mir = plx_img_new(t.mean[l].w, t.mean[l].h, t.K);
        plx_orient_mirror(&t.mean[l], &p.feat, &mir);
        CHECK(plx_img_max_abs_diff(&mir, &t.mean[l]) < 1e-6f);
        plx_orient_mirror(&t.var[l], &p.feat, &mir);
        CHECK(plx_img_max_abs_diff(&mir, &t.var[l]) < 1e-6f);
        plx_img_free(&mir);
    }
    for (int i = 0; i < N; i++) CHECK(q[i].disp < 0.2 && !q[i].bound);
    for (int r2 = 0; r2 < ns + 2; r2++) plx_img_free(&rounds[r2].mean_gray);
    plx_cg_template_free(&t);
    free(px);
    plx_img_free(&base);
}

int main(void) {
    RUN(sim_algebra);
    RUN(twin_rule_on_pixels);
    RUN(canon_warp_places_features);
    RUN(ic_recovers_similarity);
    RUN(congeal_synthetic_population);
    return TEST_REPORT();
}
