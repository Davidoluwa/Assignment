// plx-deps: ops img core
#include "test.h"

#include "core/core.h"
#include "img/image.h"
#include "ops/ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static plx_img random_img(int w, int h, int c, uint64_t seed) {
    plx_rng r;
    plx_rng_seed(&r, seed);
    plx_img im = plx_img_new(w, h, c);
    for (size_t i = 0; i < (size_t)w * h * c; i++) im.px[i] = (float)plx_rng_uniform(&r);
    return im;
}

static plx_img smooth_img(int w, int h, int c, uint64_t seed, float sigma) {
    plx_img im = random_img(w, h, c, seed);
    plx_blur(&im, &im, sigma);
    return im;
}

static double img_sum(const plx_img *im, int k) {
    double s = 0;
    const float *p = plx_img_chan(im, k);
    for (size_t i = 0; i < (size_t)im->w * im->h; i++) s += p[i];
    return s;
}

/* intensity-weighted centroid and second central moment along x */
static void moments(const plx_img *im, double *cx, double *cy, double *vx) {
    double s = 0, sx = 0, sy = 0, sxx = 0;
    for (int y = 0; y < im->h; y++)
        for (int x = 0; x < im->w; x++) {
            double v = PLX_PX(im, x, y, 0);
            s += v; sx += v * x; sy += v * y; sxx += v * x * x;
        }
    *cx = sx / s;
    *cy = sy / s;
    if (vx) *vx = sxx / s - *cx * *cx;
}

TEST(blur_constant_and_mass) {
    plx_img a = plx_img_new(40, 30, 2), b = plx_img_new(40, 30, 2);
    plx_img_fill(&a, 0.37f);
    plx_blur(&a, &b, 2.3f);
    CHECK(plx_img_max_abs_diff(&a, &b) < 1e-6f);

    /* mass inside the image is preserved when the border is far enough to never be read */
    plx_img c = plx_img_new(64, 64, 1), d = plx_img_new(64, 64, 1);
    plx_rng r;
    plx_rng_seed(&r, 3);
    for (int y = 20; y < 44; y++)
        for (int x = 20; x < 44; x++) PLX_PX(&c, x, y, 0) = (float)plx_rng_uniform(&r);
    plx_blur(&c, &d, 2.5f);
    CHECK_NEAR(img_sum(&d, 0), img_sum(&c, 0), 1e-3);

    plx_blur(&c, &c, 0.0f); /* sigma <= 0 is a copy */
    plx_img_copy(&d, &c);
    plx_blur(&c, &d, -1.0f);
    CHECK(plx_img_max_abs_diff(&c, &d) == 0.0f);
    plx_img_free(&a); plx_img_free(&b); plx_img_free(&c); plx_img_free(&d);
}

TEST(blur_impulse_is_the_kernel_and_composes) {
    float sigma = 1.7f;
    int n = 61, cx = 30, cy = 28;
    plx_img im = plx_img_new(n, n, 1), out = plx_img_new(n, n, 1);
    PLX_PX(&im, cx, cy, 0) = 1.0f;
    plx_blur(&im, &out, sigma);
    int r = plx_gauss_radius(sigma);
    double z = 0;
    for (int i = -r; i <= r; i++) z += exp(-i * i / (2.0 * sigma * sigma));
    float worst = 0;
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++) {
            int dx = x - cx, dy = y - cy;
            double want = (abs(dx) <= r && abs(dy) <= r) ? exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma)) / (z * z) : 0.0;
            worst = fmaxf(worst, fabsf(PLX_PX(&out, x, y, 0) - (float)want));
        }
    CHECK(worst < 1e-7f);
    double mx, my, vx;
    moments(&out, &mx, &my, &vx);
    CHECK_NEAR(mx, cx, 1e-5);
    CHECK_NEAR(my, cy, 1e-5);
    CHECK_NEAR(vx, sigma * sigma, 0.02 * sigma * sigma); /* truncation at 3 sigma trims ~1.5% of variance */

    /* sigma_a then sigma_b == sqrt(a^2 + b^2) */
    plx_img two = plx_img_new(n, n, 1), one = plx_img_new(n, n, 1);
    plx_blur(&im, &two, 2.0f);
    plx_blur(&two, &two, 1.5f); /* in place */
    plx_blur(&im, &one, 2.5f);
    double v2, v1;
    moments(&two, &mx, &my, &v2);
    moments(&one, &mx, &my, &v1);
    CHECK_NEAR(v2, 6.25, 0.1);
    CHECK_NEAR(v2, v1, 0.05);
    CHECK(plx_img_max_abs_diff(&two, &one) < 0.01f * PLX_PX(&one, cx, cy, 0));
    plx_img_free(&im); plx_img_free(&out); plx_img_free(&two); plx_img_free(&one);
}

TEST(blur_in_place_matches_out_of_place) {
    plx_img a = random_img(37, 23, 3, 5), b = plx_img_new(37, 23, 3), c = plx_img_clone(&a);
    plx_blur(&a, &b, 1.3f);
    plx_blur(&c, &c, 1.3f);
    CHECK(plx_img_max_abs_diff(&b, &c) == 0.0f);
    plx_img_free(&a); plx_img_free(&b); plx_img_free(&c);
}

typedef struct { const plx_img *src; plx_img *out; } par_blur_ctx;
static void par_blur(int64_t i, int tid, void *p) {
    par_blur_ctx *c = p;
    plx_blur(&c->src[i], &c->out[i], 1.0f + 0.25f * (float)i);
}

TEST(blur_is_thread_safe) {
    enum { N = 16 };
    plx_img src[N], par[N], ser[N];
    for (int i = 0; i < N; i++) {
        src[i] = random_img(50, 40, 2, 100 + i);
        par[i] = plx_img_new(50, 40, 2);
        ser[i] = plx_img_new(50, 40, 2);
        plx_blur(&src[i], &ser[i], 1.0f + 0.25f * (float)i);
    }
    par_blur_ctx c = {src, par};
    plx_parallel_for(N, 1, par_blur, &c);
    for (int i = 0; i < N; i++) {
        CHECK(plx_img_max_abs_diff(&par[i], &ser[i]) == 0.0f);
        plx_img_free(&src[i]); plx_img_free(&par[i]); plx_img_free(&ser[i]);
    }
}

TEST(gradients_exact_on_quadratic) {
    int w = 48, h = 40;
    plx_img f = plx_img_new(w, h, 1), gx = plx_img_new(w, h, 1), gy = plx_img_new(w, h, 1);
    /* f = a x^2 + b xy + c y^2 + d x + e y + g, coordinates centred to keep floats exact-ish */
    const double a = 0.0004, b = -0.0006, c = 0.0003, d = 0.01, e = -0.007, g = 0.5;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            double X = x - 24, Y = y - 20;
            PLX_PX(&f, x, y, 0) = (float)(a * X * X + b * X * Y + c * Y * Y + d * X + e * Y + g);
        }
    for (int pass = 0; pass < 2; pass++) {
        int r = pass == 0 ? plx_gauss_radius(1.5f) : 1;
        if (pass == 0) plx_gradient_gauss(&f, 1.5f, &gx, &gy);
        else plx_gradient_scharr(&f, &gx, &gy);
        float worst = 0;
        for (int y = r; y < h - r; y++)
            for (int x = r; x < w - r; x++) {
                double X = x - 24, Y = y - 20;
                worst = fmaxf(worst, fabsf(PLX_PX(&gx, x, y, 0) - (float)(2 * a * X + b * Y + d)));
                worst = fmaxf(worst, fabsf(PLX_PX(&gy, x, y, 0) - (float)(b * X + 2 * c * Y + e)));
            }
        CHECK(worst < 2e-6f);
    }
    plx_img_free(&f); plx_img_free(&gx); plx_img_free(&gy);
}

TEST(sampling_reproduces_ramps_and_quadratics) {
    int w = 20, h = 15;
    plx_img lin = plx_img_new(w, h, 1), quad = plx_img_new(w, h, 1);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            PLX_PX(&lin, x, y, 0) = 0.03f * x - 0.02f * y + 0.4f;
            PLX_PX(&quad, x, y, 0) = 0.002f * (x - 9) * (x - 9) + 0.003f * (x - 9) * (y - 7) - 0.001f * (y - 7) * (y - 7);
        }
    plx_rng r;
    plx_rng_seed(&r, 11);
    float wl = 0, wc = 0, wq = 0, wg = 0;
    for (int i = 0; i < 4000; i++) {
        float x = (float)(plx_rng_uniform(&r) * (w - 1)), y = (float)(plx_rng_uniform(&r) * (h - 1));
        if (i < 300) { x = floorf(x); y = floorf(y); } /* exact grid points too */
        float dx, dy, v = plx_sample1_grad(&lin, 0, PLX_INTERP_BILINEAR, x, y, &dx, &dy);
        float want = 0.03f * x - 0.02f * y + 0.4f;
        wl = fmaxf(wl, fabsf(v - want));
        wg = fmaxf(wg, fmaxf(fabsf(dx - 0.03f), fabsf(dy + 0.02f)));
        if (x >= 1 && x <= w - 2 && y >= 1 && y <= h - 2) {
            v = plx_sample1_grad(&lin, 0, PLX_INTERP_BICUBIC, x, y, &dx, &dy);
            wc = fmaxf(wc, fabsf(v - want));
            wg = fmaxf(wg, fmaxf(fabsf(dx - 0.03f), fabsf(dy + 0.02f)));
            float X = x - 9, Y = y - 7;
            v = plx_sample1_grad(&quad, 0, PLX_INTERP_BICUBIC, x, y, &dx, &dy);
            wq = fmaxf(wq, fabsf(v - (0.002f * X * X + 0.003f * X * Y - 0.001f * Y * Y)));
            wq = fmaxf(wq, fabsf(dx - (0.004f * X + 0.003f * Y)));
            wq = fmaxf(wq, fabsf(dy - (0.003f * X - 0.002f * Y)));
        }
    }
    CHECK(wl < 2e-6f);
    CHECK(wc < 2e-6f);
    CHECK(wg < 2e-5f);
    CHECK(wq < 2e-5f);
    plx_img_free(&lin); plx_img_free(&quad);
}

/* Difference quotient of channel 0 along `axis` between offsets lo and hi from (x, y), divided by
 * the step actually representable in float, so coordinate rounding does not masquerade as error. */
static double diffq(const plx_img *im, plx_interp m, float x, float y, int axis, float lo, float hi) {
    float xa = axis == 0 ? x + lo : x, ya = axis == 1 ? y + lo : y;
    float xb = axis == 0 ? x + hi : x, yb = axis == 1 ? y + hi : y;
    double step = axis == 0 ? (double)xb - xa : (double)yb - ya;
    return ((double)plx_sample1(im, 0, m, xb, yb) - plx_sample1(im, 0, m, xa, ya)) / step;
}

/* Finite-difference check of the analytic gradient. Bilinear is linear inside each cell, so a
 * forward difference that stays inside the cell is exact. Bicubic is C1 with a jump in the
 * second derivative at knots, where a central difference errs by O(h), so h is kept small. */
TEST(sampling_gradients_match_finite_differences) {
    int w = 17, h = 13;
    plx_img im = random_img(w, h, 1, 21);
    plx_rng r;
    plx_rng_seed(&r, 22);
    const float hl = 1e-2f, hc = 1e-3f;
    float worst_lin = 0, worst_cub = 0;
    for (int i = 0; i < 6000; i++) {
        /* cover the whole extended domain [-3, w+2], with two thirds of the points on cell edges */
        float x = (float)(-3 + plx_rng_uniform(&r) * (w + 5)), y = (float)(-3 + plx_rng_uniform(&r) * (h + 5));
        if (i % 3 == 0) x = floorf(x);
        if (i % 3 == 1) y = floorf(y);
        float dx, dy;
        if (x - floorf(x) < 1 - 2 * hl && y - floorf(y) < 1 - 2 * hl) { /* forward step stays in the cell */
            plx_sample1_grad(&im, 0, PLX_INTERP_BILINEAR, x, y, &dx, &dy);
            double ex = fabs(diffq(&im, PLX_INTERP_BILINEAR, x, y, 0, 0, hl) - dx);
            double ey = fabs(diffq(&im, PLX_INTERP_BILINEAR, x, y, 1, 0, hl) - dy);
            worst_lin = fmaxf(worst_lin, (float)fmax(ex, ey));
        }
        plx_sample1_grad(&im, 0, PLX_INTERP_BICUBIC, x, y, &dx, &dy);
        double ex = fabs(diffq(&im, PLX_INTERP_BICUBIC, x, y, 0, -hc, hc) - dx);
        double ey = fabs(diffq(&im, PLX_INTERP_BICUBIC, x, y, 1, -hc, hc) - dy);
        worst_cub = fmaxf(worst_cub, (float)fmax(ex, ey));
    }
    CHECK(worst_lin < 2e-4f);
    CHECK(worst_cub < 1e-3f);

    /* far outside: edge value, zero gradient */
    float dx = 1, dy = 1, v = plx_sample1_grad(&im, 0, PLX_INTERP_BICUBIC, -50.0f, 1e7f, &dx, &dy);
    CHECK(v == PLX_PX(&im, 0, h - 1, 0) && dx == 0.0f && dy == 0.0f);
    plx_img_free(&im);
}

TEST(multichannel_sampling_matches_single_channel) {
    plx_img im = random_img(30, 25, 5, 31);
    float val[5], grad[10];
    plx_rng r;
    plx_rng_seed(&r, 32);
    for (int i = 0; i < 200; i++) {
        float x = (float)(-2 + plx_rng_uniform(&r) * 33), y = (float)(-2 + plx_rng_uniform(&r) * 28);
        for (int m = 0; m < 2; m++) {
            bool in = plx_sample(&im, (plx_interp)m, x, y, val, grad);
            CHECK(in == (x >= 0 && y >= 0 && x <= 29 && y <= 24));
            for (int k = 0; k < 5; k++) {
                float dx, dy, v = plx_sample1_grad(&im, k, (plx_interp)m, x, y, &dx, &dy);
                CHECK(v == val[k] && dx == grad[2 * k] && dy == grad[2 * k + 1]);
            }
        }
    }
    /* batch call == loop of single calls, grad/inside optional */
    float xs[3] = {1.5f, -1.0f, 28.9f}, ys[3] = {2.25f, 3.0f, 24.0f}, bv[15], bg[30], v2[15];
    uint8_t ins[3];
    plx_sample_points(&im, PLX_INTERP_BICUBIC, 3, xs, ys, bv, bg, ins);
    plx_sample_points(&im, PLX_INTERP_BICUBIC, 3, xs, ys, v2, NULL, NULL);
    CHECK(ins[0] == 1 && ins[1] == 0 && ins[2] == 1);
    for (int i = 0; i < 3; i++) {
        plx_sample(&im, PLX_INTERP_BICUBIC, xs[i], ys[i], val, grad);
        for (int k = 0; k < 5; k++) CHECK(bv[i * 5 + k] == val[k] && v2[i * 5 + k] == val[k] && bg[i * 10 + 2 * k + 1] == grad[2 * k + 1]);
    }
    /* NaN coordinates are outside and still produce finite values */
    CHECK(!plx_sample(&im, PLX_INTERP_BICUBIC, NAN, 3.0f, val, grad));
    CHECK(isfinite(val[0]) && isfinite(grad[0]));
    plx_img_free(&im);
}

TEST(sampling_speed_8ch_25600_points) {
    plx_img im = smooth_img(160, 160, 8, 41, 1.0f);
    int n = 25600;
    float *x = PLX_NEW(float, n), *y = PLX_NEW(float, n), *v = PLX_NEW(float, n * 8), *g = PLX_NEW(float, n * 16);
    plx_rng r;
    plx_rng_seed(&r, 42);
    for (int i = 0; i < n; i++) { x[i] = (float)(plx_rng_uniform(&r) * 159); y[i] = (float)(plx_rng_uniform(&r) * 159); }
    for (int m = 0; m < 2; m++) {
        int reps = 20;
        double t0 = plx_now();
        for (int rep = 0; rep < reps; rep++) plx_sample_points(&im, (plx_interp)m, n, x, y, v, g, NULL);
        double dt = (plx_now() - t0) / reps;
        printf("  timing: %s, 8 ch x %d points with gradients: %.3f ms\n", m ? "bicubic " : "bilinear", n, dt * 1e3);
        CHECK(dt < 0.25); /* sanity bound only; generous for sanitizer builds */
    }
    free(x); free(y); free(v); free(g);
    plx_img_free(&im);
}

TEST(affine_warp_identity_and_inverse) {
    int w = 64, h = 56;
    plx_img src = smooth_img(w, h, 2, 51, 2.5f);
    plx_img dst = plx_img_new(w, h, 2), back = plx_img_new(w, h, 2), valid = plx_img_new(w, h, 1);
    const float id[6] = {1, 0, 0, 0, 1, 0};
    for (int m = 0; m < 2; m++) {
        plx_warp_affine(&src, id, (plx_interp)m, &dst, &valid);
        CHECK(plx_img_max_abs_diff(&src, &dst) == 0.0f);
        CHECK(img_sum(&valid, 0) == (double)w * h);
    }
    /* T: rotate 12 degrees and scale 1.08 about the centre, plus a sub-pixel shift */
    float c = 1.08f * cosf(0.21f), s = 1.08f * sinf(0.21f), cx = 31.5f, cy = 27.5f;
    float T[6] = {c, -s, 0, s, c, 0}, Ti[6], I[6];
    T[2] = cx - (c * cx - s * cy) + 0.37f;
    T[5] = cy - (s * cx + c * cy) - 0.21f;
    CHECK(plx_affine_invert(T, Ti));
    plx_affine_compose(T, Ti, I);
    for (int i = 0; i < 6; i++) CHECK_NEAR(I[i], id[i], 1e-5);
    plx_warp_affine(&src, T, PLX_INTERP_BICUBIC, &dst, &valid);
    CHECK(img_sum(&valid, 0) < (double)w * h); /* corners map outside */
    plx_warp_affine(&dst, Ti, PLX_INTERP_BICUBIC, &back, NULL);
    float worst = 0;
    for (int k = 0; k < 2; k++)
        for (int y = 14; y < h - 14; y++)
            for (int x = 14; x < w - 14; x++) worst = fmaxf(worst, fabsf(PLX_PX(&back, x, y, k) - PLX_PX(&src, x, y, k)));
    CHECK(worst < 2e-3f); /* two bicubic resamplings of a sigma 2.5 image */

    /* a dense map holding the affine coordinates gives the same warp, bit for bit */
    plx_img map = plx_img_new(w, h, 2), dst2 = plx_img_new(w, h, 2), valid2 = plx_img_new(w, h, 1);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            PLX_PX(&map, x, y, 0) = T[0] * (float)x + T[1] * (float)y + T[2];
            PLX_PX(&map, x, y, 1) = T[3] * (float)x + T[4] * (float)y + T[5];
        }
    plx_warp_map(&src, &map, PLX_INTERP_BICUBIC, &dst2, &valid2);
    CHECK(plx_img_max_abs_diff(&dst, &dst2) == 0.0f && plx_img_max_abs_diff(&valid, &valid2) == 0.0f);
    float sing[6] = {1, 2, 0, 2, 4, 0};
    CHECK(!plx_affine_invert(sing, Ti));
    plx_img_free(&src); plx_img_free(&dst); plx_img_free(&back); plx_img_free(&valid);
    plx_img_free(&map); plx_img_free(&dst2); plx_img_free(&valid2);
}

/* A blob placed at a known sub-pixel point must land at (x + 0.5) / 2^L - 0.5 on level L. */
TEST(pyramid_registration_by_centroid) {
    const int sizes[2][2] = {{256, 240}, {251, 233}}; /* even and odd */
    for (int t = 0; t < 2; t++) {
        int w = sizes[t][0], h = sizes[t][1];
        double x0 = 117.3, y0 = 101.8, s = 9.0;
        plx_img im = plx_img_new(w, h, 1);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) PLX_PX(&im, x, y, 0) = (float)exp(-((x - x0) * (x - x0) + (y - y0) * (y - y0)) / (2 * s * s));
        plx_pyramid p = plx_pyramid_build(&im, 5, 8);
        CHECK(p.n == 5);
        for (int L = 0; L < p.n; L++) {
            if (L > 0) CHECK(p.lev[L].w == (p.lev[L - 1].w + 1) / 2 && p.lev[L].h == (p.lev[L - 1].h + 1) / 2);
            double cx, cy;
            moments(&p.lev[L], &cx, &cy, NULL);
            CHECK_NEAR(cx, plx_pyr_coord((float)x0, 0, L), 2e-3);
            CHECK_NEAR(cy, plx_pyr_coord((float)y0, 0, L), 2e-3);
            CHECK_NEAR(plx_pyr_coord(plx_pyr_coord((float)x0, 0, L), L, 0), x0, 1e-4);
        }
        plx_pyramid_free(&p);
        plx_img_free(&im);
    }
    plx_img tiny = plx_img_new(20, 9, 1);
    plx_pyramid p = plx_pyramid_build(&tiny, 6, 3); /* 20x9 -> 10x5 -> 5x3; next would be 3x2 */
    CHECK(p.n == 3 && p.lev[2].w == 5 && p.lev[2].h == 3);
    plx_pyramid_free(&p);
    plx_img_free(&tiny);
}

TEST(downsample_is_flip_symmetric_for_even_sizes) {
    plx_img a = random_img(40, 30, 1, 61), fa = plx_img_new(40, 30, 1);
    plx_flip_h(&a, &fa);
    plx_img da = plx_downsample2(&a), dfa = plx_downsample2(&fa), fda = plx_img_new(da.w, da.h, 1);
    plx_flip_h(&da, &fda);
    CHECK(plx_img_max_abs_diff(&dfa, &fda) < 1e-6f);
    plx_img_free(&a); plx_img_free(&fa); plx_img_free(&da); plx_img_free(&dfa); plx_img_free(&fda);
}

TEST(resize_area_and_bilinear) {
    plx_img a = random_img(30, 24, 2, 71);
    /* integer shrink = block average */
    plx_img half = plx_resize(&a, 15, 12);
    float worst = 0;
    for (int k = 0; k < 2; k++)
        for (int y = 0; y < 12; y++)
            for (int x = 0; x < 15; x++) {
                float m = 0.25f * (PLX_PX(&a, 2 * x, 2 * y, k) + PLX_PX(&a, 2 * x + 1, 2 * y, k) +
                                   PLX_PX(&a, 2 * x, 2 * y + 1, k) + PLX_PX(&a, 2 * x + 1, 2 * y + 1, k));
                worst = fmaxf(worst, fabsf(PLX_PX(&half, x, y, k) - m));
            }
    CHECK(worst < 1e-6f);
    /* non-integer area shrink preserves the mean */
    plx_img odd = plx_resize(&a, 11, 7);
    for (int k = 0; k < 2; k++) CHECK_NEAR(img_sum(&odd, k) / 77.0, img_sum(&a, k) / 720.0, 1e-5);
    /* same size is the identity */
    plx_img same = plx_resize(&a, 30, 24);
    CHECK(plx_img_max_abs_diff(&same, &a) == 0.0f);
    /* enlarging a ramp: bilinear reproduces it at the area-aligned centres (interior) */
    plx_img ramp = plx_img_new(10, 8, 1);
    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 10; x++) PLX_PX(&ramp, x, y, 0) = 0.1f * x + 0.05f * y;
    plx_img big = plx_resize(&ramp, 25, 19);
    worst = 0;
    for (int y = 0; y < 19; y++)
        for (int x = 0; x < 25; x++) {
            double sx = (x + 0.5) * 10 / 25 - 0.5, sy = (y + 0.5) * 8 / 19.0 - 0.5;
            if (sx < 0 || sy < 0 || sx > 9 || sy > 7) continue;
            worst = fmaxf(worst, fabsf(PLX_PX(&big, x, y, 0) - (float)(0.1 * sx + 0.05 * sy)));
        }
    CHECK(worst < 1e-5f);
    plx_img_free(&a); plx_img_free(&half); plx_img_free(&odd); plx_img_free(&same);
    plx_img_free(&ramp); plx_img_free(&big);
}

TEST(flip_and_normalize) {
    plx_img a = random_img(9, 4, 3, 81), b = plx_img_new(9, 4, 3), c = plx_img_clone(&a);
    plx_flip_h(&a, &b);
    CHECK(PLX_PX(&b, 0, 2, 1) == PLX_PX(&a, 8, 2, 1) && PLX_PX(&b, 4, 3, 2) == PLX_PX(&a, 4, 3, 2));
    plx_flip_h(&b, &b);
    CHECK(plx_img_max_abs_diff(&a, &b) == 0.0f);
    int perm[3] = {2, 0, 1};
    plx_flip_h_perm(&a, perm, &b);
    plx_flip_h_perm(&c, perm, &c); /* in place */
    CHECK(plx_img_max_abs_diff(&b, &c) == 0.0f);
    CHECK(PLX_PX(&b, 1, 0, 0) == PLX_PX(&a, 7, 0, 2) && PLX_PX(&b, 1, 0, 1) == PLX_PX(&a, 7, 0, 0));

    plx_img_scale_offset(&a, 3.0f, -1.0f);
    plx_normalize_meanvar(&a, 1e-8f);
    for (int k = 0; k < 3; k++) {
        double m, v;
        plx_chan_stats(&a, k, &m, &v);
        CHECK_NEAR(m, 0.0, 1e-6);
        CHECK_NEAR(v, 1.0, 1e-5);
    }
    plx_img e = plx_img_extract(&a, 1);
    CHECK(e.c == 1 && e.px[5] == PLX_PX(&a, 5, 0, 1));
    plx_img_free(&a); plx_img_free(&b); plx_img_free(&c); plx_img_free(&e);
}

TEST(integral_box_sums_match_brute_force) {
    plx_img a = random_img(31, 22, 2, 91);
    plx_integral ii = plx_integral_build(&a);
    plx_rng r;
    plx_rng_seed(&r, 92);
    for (int t = 0; t < 300; t++) {
        int k = (int)plx_rng_below(&r, 2);
        int x0 = (int)plx_rng_below(&r, 40) - 4, x1 = (int)plx_rng_below(&r, 40) - 4;
        int y0 = (int)plx_rng_below(&r, 30) - 4, y1 = (int)plx_rng_below(&r, 30) - 4;
        double want = 0;
        for (int y = PLX_MAX(y0, 0); y < PLX_MIN(y1, 22); y++)
            for (int x = PLX_MAX(x0, 0); x < PLX_MIN(x1, 31); x++) want += PLX_PX(&a, x, y, k);
        CHECK_NEAR(plx_integral_box(&ii, k, x0, y0, x1, y1), want, 1e-9);
    }
    plx_integral_free(&ii);
    plx_img_free(&a);
}

int main(void) {
    RUN(blur_constant_and_mass);
    RUN(blur_impulse_is_the_kernel_and_composes);
    RUN(blur_in_place_matches_out_of_place);
    RUN(blur_is_thread_safe);
    RUN(gradients_exact_on_quadratic);
    RUN(sampling_reproduces_ramps_and_quadratics);
    RUN(sampling_gradients_match_finite_differences);
    RUN(multichannel_sampling_matches_single_channel);
    RUN(sampling_speed_8ch_25600_points);
    RUN(affine_warp_identity_and_inverse);
    RUN(pyramid_registration_by_centroid);
    RUN(downsample_is_flip_symmetric_for_even_sizes);
    RUN(resize_area_and_bilinear);
    RUN(flip_and_normalize);
    RUN(integral_box_sums_match_brute_force);
    return TEST_REPORT();
}
