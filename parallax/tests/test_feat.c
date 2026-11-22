// plx-deps: feat ops img core
#include "test.h"

#include "core/core.h"
#include "feat/feat.h"
#include "img/image.h"
#include "ops/ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Real-data checks need the JPEG decoder (being written alongside this module) and the LFW
 * images; without either they are skipped, never failed. */
#if defined(__has_include)
#if __has_include("img/jpeg.h")
#include "img/jpeg.h"
#define HAVE_JPEG 1
#endif
#endif

static const double PI = 3.14159265358979323846;

/* blurred noise plus a few strong structures: texture at all orientations */
static plx_img textured(int w, int h, uint64_t seed) {
    plx_rng r;
    plx_rng_seed(&r, seed);
    plx_img im = plx_img_new(w, h, 1);
    for (size_t i = 0; i < (size_t)w * h; i++) im.px[i] = (float)plx_rng_uniform(&r);
    plx_blur(&im, &im, 1.2f);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            double dx = x - 0.37 * w, dy = y - 0.55 * h;
            PLX_PX(&im, x, y, 0) = 0.5f * PLX_PX(&im, x, y, 0) + (dx * dx + 1.7 * dy * dy < 150 ? 0.4f : 0.1f) +
                                   (x > 0.7 * w + 0.3 * y ? 0.2f : 0.0f);
        }
    return im;
}

/* straight smooth edge through the image centre, intensity increasing along angle theta */
static plx_img edge_img(int n, double theta, double lo, double hi) {
    plx_img im = plx_img_new(n, n, 1);
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++) {
            double s = (x - n / 2.0) * cos(theta) + (y - n / 2.0) * sin(theta);
            PLX_PX(&im, x, y, 0) = (float)(lo + (hi - lo) * 0.5 * (1 + tanh(s / 1.5)));
        }
    return im;
}

static float max_abs(const plx_img *im) {
    float m = 0;
    for (size_t i = 0; i < (size_t)im->w * im->h * im->c; i++) m = fmaxf(m, fabsf(im->px[i]));
    return m;
}

TEST(orient_bin_layout_and_perm) {
    plx_orient_params p = plx_orient_default();
    CHECK(p.k == 8 && p.signed_);
    int perm[8];
    plx_orient_perm(&p, perm);
    const int want[8] = {0, 7, 6, 5, 4, 3, 2, 1};
    for (int j = 0; j < 8; j++) {
        CHECK(perm[j] == want[j] && perm[perm[j]] == j);
        /* c_perm[j] == pi - c_j (mod period) */
        double d = fmod(plx_orient_bin_center(&p, perm[j]) - (PI - plx_orient_bin_center(&p, j)) + 8 * PI, 2 * PI);
        CHECK(fmin(d, 2 * PI - d) < 1e-5);
    }
    CHECK_NEAR(plx_orient_bin_center(&p, 0), PI / 2, 1e-6);
    CHECK_NEAR(plx_orient_bin_center(&p, 2), PI, 1e-6);
}

/* An edge at angle theta must put its energy in the two bins around theta, split linearly. */
TEST(orient_edge_energy_in_right_bins) {
    int n = 96, lo = 30, hi = 66;
    for (int mode = 0; mode < 2; mode++) {
        plx_orient_params p = plx_orient_default();
        p.signed_ = mode == 0;
        double period = p.signed_ ? 2 * PI : PI;
        for (int t = 0; t < 24; t++) {
            double theta = -PI + t * (2 * PI / 24) + 0.13;
            plx_img im = edge_img(n, theta, 0.2, 0.7);
            plx_img f = plx_orient_channels(&im, &p);
            double e[8] = {0}, tot = 0;
            for (int k = 0; k < 8; k++)
                for (int y = lo; y < hi; y++)
                    for (int x = lo; x < hi; x++) { e[k] += PLX_PX(&f, x, y, k); tot += PLX_PX(&f, x, y, k); }
            double u = fmod((theta - PI / 2) * 8 / period + 80, 8.0);
            int j0 = (int)u, j1 = (j0 + 1) % 8;
            double frac = u - j0;
            CHECK((e[j0] + e[j1]) / tot > 0.999);
            CHECK_NEAR(e[j1] / (e[j0] + e[j1]), frac, 0.01); /* 0.01 bin = 0.45 deg (signed) */
            plx_img_free(&im);
            plx_img_free(&f);
        }
    }
}

TEST(orient_invariant_to_brightness_and_contrast) {
    plx_orient_params p = plx_orient_default();
    plx_img im = textured(90, 80, 5);
    plx_img f = plx_orient_channels(&im, &p);
    float scale = max_abs(&f);
    CHECK(scale > 1.0f);

    plx_img b = plx_img_clone(&im);
    plx_img_scale_offset(&b, 1.0f, 0.15f); /* brightness: exact */
    plx_img fb = plx_orient_channels(&b, &p);
    CHECK(plx_img_max_abs_diff(&f, &fb) < 1e-5f * scale);

    /* contrast: exact up to the eps floor, so check with a negligible eps */
    plx_orient_params q = p;
    q.eps = 1e-7f;
    plx_img fq = plx_orient_channels(&im, &q);
    plx_img_scale_offset(&b, 0.4f, 0.1f);
    plx_img fc = plx_orient_channels(&b, &q);
    CHECK(plx_img_max_abs_diff(&fq, &fc) < 1e-4f * max_abs(&fq));
    /* with the default floor, halving the contrast changes the features only mildly */
    plx_img h = plx_img_clone(&im);
    plx_img_scale_offset(&h, 0.5f, 0.25f);
    plx_img fh = plx_orient_channels(&h, &p);
    CHECK(plx_img_max_abs_diff(&f, &fh) < 0.25f * scale);

    /* polarity: inverting the image rotates signed bins by K/2 and leaves unsigned ones */
    plx_img inv = plx_img_clone(&im);
    plx_img_scale_offset(&inv, -1.0f, 1.0f);
    plx_img fi = plx_orient_channels(&inv, &p);
    float worst = 0;
    for (int k = 0; k < 8; k++)
        for (size_t i = 0; i < (size_t)90 * 80; i++) worst = fmaxf(worst, fabsf(fi.px[k * 7200 + i] - f.px[((k + 4) % 8) * 7200 + i]));
    CHECK(worst < 1e-5f * scale);
    plx_orient_params u = p;
    u.signed_ = false;
    plx_img fu = plx_orient_channels(&im, &u), fui = plx_orient_channels(&inv, &u);
    CHECK(plx_img_max_abs_diff(&fu, &fui) < 1e-5f * scale);

    plx_img_free(&im); plx_img_free(&f); plx_img_free(&b); plx_img_free(&fb); plx_img_free(&fq);
    plx_img_free(&fc); plx_img_free(&h); plx_img_free(&fh); plx_img_free(&inv); plx_img_free(&fi);
    plx_img_free(&fu); plx_img_free(&fui);
}

/* features(flip(I)) == flip_perm(features(I)) - for even and odd K, both modes. Every stage
 * is flip-equivariant (replicate borders included), so it holds over the whole image. */
TEST(orient_mirror_permutation_is_exact) {
    plx_img im = textured(101, 84, 7), fl = plx_img_new(101, 84, 1);
    plx_flip_h(&im, &fl);
    const int ks[3] = {8, 9, 4};
    for (int t = 0; t < 3; t++)
        for (int mode = 0; mode < 2; mode++) {
            plx_orient_params p = plx_orient_default();
            p.k = ks[t];
            p.signed_ = mode == 0;
            plx_img f = plx_orient_channels(&im, &p), ff = plx_orient_channels(&fl, &p);
            plx_img m = plx_img_new(f.w, f.h, f.c);
            plx_orient_mirror(&f, &p, &m);
            float scale = max_abs(&f);
            CHECK(plx_img_max_abs_diff(&m, &ff) < 1e-5f * scale);
            /* and it is not trivially true: without the permutation it fails */
            plx_img plain = plx_img_new(f.w, f.h, f.c);
            plx_flip_h(&f, &plain);
            CHECK(plx_img_max_abs_diff(&plain, &ff) > 0.1f * scale);
            plx_orient_mirror(&m, &p, &m); /* involution, in place */
            CHECK(plx_img_max_abs_diff(&m, &f) == 0.0f);
            plx_img_free(&f); plx_img_free(&ff); plx_img_free(&m); plx_img_free(&plain);
        }
    plx_img_free(&im); plx_img_free(&fl);
}

TEST(orient_speed_250) {
    plx_img im = textured(250, 250, 9);
    plx_orient_params p = plx_orient_default();
    int reps = 10;
    double t0 = plx_now();
    for (int r = 0; r < reps; r++) {
        plx_img f = plx_orient_channels(&im, &p);
        plx_img_free(&f);
    }
    double dt = (plx_now() - t0) / reps;
    printf("  timing: orientation channels (K=8) on 250x250: %.2f ms\n", dt * 1e3);
    CHECK(dt < 0.5); /* sanity bound only; generous for sanitizer builds */
    plx_img_free(&im);
}

TEST(lbp_hand_computed) {
    /* centre 5; E 6, NE 2, N 7, NW 5 (tie -> 1), W 1, SW 9, S 3, SE 8 */
    const float patch[9] = {5, 7, 2,
                            1, 5, 6,
                            9, 3, 8};
    plx_img im = plx_img_new(3, 3, 1);
    memcpy(im.px, patch, sizeof patch);
    uint8_t codes[25];
    plx_lbp(&im, 0, 1, codes);
    CHECK(codes[4] == (1 | 4 | 8 | 32 | 128));

    /* a linear ramp f = x + 10 y: bilinear is exact on it, so the radius-2 code is computable
     * by hand: E 24 (1), NE 22+1.41-14.1 (0), N 2 (0), NW (0), W 20 (0), SW 22-1.41+14.1 (1),
     * S 42 (1), SE (1). Radius 1 gives the same bits. */
    plx_img ramp = plx_img_new(5, 5, 1);
    for (int y = 0; y < 5; y++)
        for (int x = 0; x < 5; x++) PLX_PX(&ramp, x, y, 0) = (float)(x + 10 * y);
    plx_lbp(&ramp, 0, 2, codes);
    CHECK(codes[12] == (1 | 32 | 64 | 128));
    plx_lbp(&ramp, 0, 1, codes);
    CHECK(codes[12] == (1 | 32 | 64 | 128));

    /* radius-2 diagonal is interpolated: a single bright pixel at (3, 1) is seen by NE with
     * bilinear weight (1 - 0.414)^2 = 0.343 */
    plx_img spot = plx_img_new(5, 5, 1);
    plx_img_fill(&spot, 0.5f);
    PLX_PX(&spot, 2, 2, 0) = 0.6f;
    PLX_PX(&spot, 3, 1, 0) = 0.8f; /* NE sample = 0.5 + 0.343 * 0.3 = 0.603 >= 0.6 */
    plx_lbp(&spot, 0, 2, codes);
    CHECK(codes[12] == 2);
    PLX_PX(&spot, 3, 1, 0) = 0.78f; /* 0.5 + 0.343 * 0.28 = 0.596 < 0.6 */
    plx_lbp(&spot, 0, 2, codes);
    CHECK(codes[12] == 0);

    plx_img flat = plx_img_new(6, 4, 1); /* ties everywhere -> all bits set */
    plx_img_fill(&flat, 0.3f);
    plx_lbp(&flat, 0, 2, codes);
    for (int i = 0; i < 24; i++) CHECK(codes[i] == 255);
    plx_img_free(&im); plx_img_free(&ramp); plx_img_free(&spot); plx_img_free(&flat);
}

TEST(lbp_uniform_table) {
    uint8_t t[256];
    CHECK(plx_lbp_uniform_table(t) == 59);
    int count[59] = {0};
    for (int c = 0; c < 256; c++) count[t[c]]++;
    for (int l = 0; l < 58; l++) CHECK(count[l] == 1);
    CHECK(count[58] == 256 - 58);
    CHECK(t[0] == 0 && t[255] == 57);
    CHECK(t[0x0F] != 58 && t[0x81] != 58 && t[0xF0] != 58); /* runs, including wrap-around */
    CHECK(t[0x05] == 58 && t[0x55] == 58);
    for (int c = 1; c < 256; c++) /* labels increase with code among uniform patterns */
        for (int d = 0; d < c; d++) if (t[c] != 58 && t[d] != 58) CHECK(t[d] < t[c]);
}

/* brute force: loop over every cell, then every pixel of the image, test membership */
static void ref_hist(const uint8_t *lab, const plx_img *ch, int w, int h, int bins, const float *wt,
                     const plx_cell_grid *g, double *out, double *cw) {
    for (int cy = 0; cy < g->cells_y; cy++)
        for (int cx = 0; cx < g->cells_x; cx++) {
            double *o = out + (cy * g->cells_x + cx) * bins, s = 0, tw = 0;
            for (int b = 0; b < bins; b++) o[b] = 0;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++) {
                    if (x < g->x0 || x >= g->x0 + g->w || y < g->y0 || y >= g->y0 + g->h) continue;
                    if ((x - g->x0) * g->cells_x / g->w != cx || (y - g->y0) * g->cells_y / g->h != cy) continue;
                    double ww = wt ? wt[y * w + x] : 1.0;
                    tw += ww;
                    if (lab) o[lab[y * w + x]] += ww;
                    else for (int b = 0; b < bins; b++) o[b] += ww * PLX_PX(ch, x, y, b);
                }
            cw[cy * g->cells_x + cx] = tw;
            for (int b = 0; b < bins; b++) s += g->norm == PLX_HNORM_L2 ? o[b] * o[b] : fabs(o[b]);
            if (g->norm == PLX_HNORM_L2) s = sqrt(s);
            for (int b = 0; b < bins && g->norm != PLX_HNORM_NONE; b++) {
                o[b] /= s + g->eps;
                if (g->norm == PLX_HNORM_L1_SQRT) o[b] = copysign(sqrt(fabs(o[b])), o[b]);
            }
        }
}

TEST(cell_histograms_match_brute_force) {
    int w = 41, h = 37;
    plx_rng r;
    plx_rng_seed(&r, 3);
    uint8_t *lab = PLX_NEW(uint8_t, w * h);
    float *wt = PLX_NEW(float, w * h);
    plx_img ch = plx_img_new(w, h, 5);
    for (int i = 0; i < w * h; i++) { lab[i] = (uint8_t)plx_rng_below(&r, 59); wt[i] = (float)plx_rng_uniform(&r); }
    for (int i = 0; i < w * h * 5; i++) ch.px[i] = (float)plx_rng_uniform(&r);
    /* a region inside the image and one hanging off two edges, with uneven cell sizes */
    const plx_cell_grid grids[2] = {{3, 2, 35, 33, 5, 4, PLX_HNORM_NONE, 1e-6f}, {-16, 20, 40, 25, 5, 3, PLX_HNORM_NONE, 1e-6f}};
    float out[20 * 59], cw[20];
    double ref[20 * 59], rcw[20];
    for (int gi = 0; gi < 2; gi++)
        for (int norm = 0; norm < 4; norm++)
            for (int use_w = 0; use_w < 2; use_w++) {
                plx_cell_grid g = grids[gi];
                g.norm = (plx_hist_norm)norm;
                int nc = g.cells_x * g.cells_y;
                const float *W = use_w ? wt : NULL;
                plx_hist_labels(lab, w, h, 59, W, &g, out, cw);
                ref_hist(lab, NULL, w, h, 59, W, &g, ref, rcw);
                float worst = 0, worst_w = 0;
                for (int i = 0; i < nc * 59; i++) worst = fmaxf(worst, (float)fabs(out[i] - ref[i]));
                for (int i = 0; i < nc; i++) worst_w = fmaxf(worst_w, (float)fabs(cw[i] - rcw[i]));
                CHECK(worst < 1e-5f && worst_w < 1e-4f);

                plx_hist_channels(&ch, W, &g, out, cw);
                ref_hist(NULL, &ch, w, h, 5, W, &g, ref, rcw);
                worst = 0;
                for (int i = 0; i < nc * 5; i++) worst = fmaxf(worst, (float)(fabs(out[i] - ref[i]) / (1 + fabs(ref[i]))));
                CHECK(worst < 1e-5f);
            }
    /* normalised cells have the promised norms; the cells left of the image are empty */
    plx_cell_grid g = grids[1];
    g.norm = PLX_HNORM_L1_SQRT;
    plx_hist_labels(lab, w, h, 59, wt, &g, out, cw);
    CHECK(cw[0] == 0.0f && cw[1] == 0.0f && cw[2] > 0.0f);
    double l2 = 0;
    for (int b = 0; b < 59; b++) { CHECK(out[b] == 0.0f && out[59 + b] == 0.0f); l2 += (double)out[118 + b] * out[118 + b]; }
    CHECK_NEAR(l2, 1.0, 1e-4);
    free(lab); free(wt); plx_img_free(&ch);
}

TEST(orient_histogram_descriptor_on_edge) {
    /* end to end: orientation channels -> 2x2 cell histogram; a vertical edge (gradient +x,
     * theta = 0) sits in bin 6 (c_6 = pi/2 + 6 pi/4 = 2 pi) */
    plx_orient_params p = plx_orient_default();
    plx_img im = edge_img(64, 0.0, 0.2, 0.8);
    plx_img f = plx_orient_channels(&im, &p);
    plx_cell_grid g = {16, 16, 32, 32, 2, 2, PLX_HNORM_L1, 1e-6f};
    float d[4 * 8];
    plx_hist_channels(&f, NULL, &g, d, NULL);
    for (int c = 0; c < 4; c++) CHECK(d[c * 8 + 6] > 0.99f);
    plx_img_free(&im); plx_img_free(&f);
}

#ifdef HAVE_JPEG
static bool load_gray(const char *path, plx_img *out) {
    uint8_t *data;
    size_t len;
    if (plx_read_file(path, &data, &len) != PLX_OK) return false;
    plx_img rgb = {0};
    int st = plx_jpeg_decode(data, len, &rgb);
    free(data);
    if (st != PLX_OK) return false;
    *out = plx_img_luma(&rgb);
    plx_img_free(&rgb);
    return true;
}
#endif

/* The mirror property on real faces, both modes, plus a sanity look at the value range. */
TEST(orient_mirror_on_lfw) {
#ifdef HAVE_JPEG
    const char *paths[3] = {"data/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg", "data/lfw/AJ_Cook/AJ_Cook_0001.jpg",
                            "data/lfw/George_W_Bush/George_W_Bush_0001.jpg"};
    int used = 0;
    for (int i = 0; i < 3; i++) {
        plx_img im;
        if (!load_gray(paths[i], &im)) continue;
        used++;
        plx_img fl = plx_img_new(im.w, im.h, 1);
        plx_flip_h(&im, &fl);
        for (int mode = 0; mode < 2; mode++) {
            plx_orient_params p = plx_orient_default();
            p.signed_ = mode == 0;
            plx_img f = plx_orient_channels(&im, &p), ff = plx_orient_channels(&fl, &p);
            plx_img m = plx_img_new(f.w, f.h, f.c);
            plx_orient_mirror(&f, &p, &m);
            float scale = max_abs(&f);
            CHECK(scale > 0.5f && scale < 50.0f);
            CHECK(plx_img_max_abs_diff(&m, &ff) < 1e-5f * scale);
            plx_img_free(&f); plx_img_free(&ff); plx_img_free(&m);
        }
        plx_img_free(&im); plx_img_free(&fl);
    }
    if (!used) printf("  skip: no LFW images decodable\n");
#else
    printf("  skip: img/jpeg.h not available\n");
#endif
}

int main(void) {
    RUN(orient_bin_layout_and_perm);
    RUN(orient_edge_energy_in_right_bins);
    RUN(orient_invariant_to_brightness_and_contrast);
    RUN(orient_mirror_permutation_is_exact);
    RUN(orient_mirror_on_lfw);
    RUN(orient_speed_250);
    RUN(lbp_hand_computed);
    RUN(lbp_uniform_table);
    RUN(cell_histograms_match_brute_force);
    RUN(orient_histogram_descriptor_on_edge);
    return TEST_REPORT();
}
