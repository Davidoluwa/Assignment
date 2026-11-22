// plx-deps: rec eval data la feat ops img core
#include "test.h"

#include "core/core.h"
#include "data/lfw.h"
#include "ops/ops.h"
#include "rec/match.h"
#include "rec/proto.h"
#include "rec/rec.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* blurred noise plus a few blobs and edges: something LBP and orientation channels can bite on */
static plx_img textured(int w, int h, uint64_t seed) {
    plx_rng r;
    plx_rng_seed(&r, seed);
    plx_img im = plx_img_new(w, h, 1);
    for (size_t i = 0; i < (size_t)w * h; i++) im.px[i] = (float)plx_rng_uniform(&r);
    plx_blur(&im, &im, 1.0f);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            double dx = x - 0.4 * w, dy = y - 0.45 * h;
            PLX_PX(&im, x, y, 0) = 0.15f + 0.5f * PLX_PX(&im, x, y, 0) + (dx * dx + 2 * dy * dy < 300 ? 0.25f : 0.0f) +
                                   (x > 0.6 * w + 0.2 * y ? 0.1f : 0.0f);
        }
    return im;
}

static double vec_cos(const float *a, const float *b, int n) {
    double ab = 0, aa = 0, bb = 0;
    for (int i = 0; i < n; i++) {
        ab += (double)a[i] * b[i];
        aa += (double)a[i] * a[i];
        bb += (double)b[i] * b[i];
    }
    return ab / sqrt(aa * bb);
}

static float max_diff(const float *a, const float *b, int n) {
    float m = 0;
    for (int i = 0; i < n; i++) m = fmaxf(m, fabsf(a[i] - b[i]));
    return m;
}

static plx_desc_params params_with(const char *blocks) {
    plx_desc_params p = {.x0 = 16, .y0 = 40, .w = 96, .h = 112, .sqrt_hist = true, .photo = plx_photo_default(),
                         .orient = plx_orient_default()};
    PLX_ASSERT(plx_desc_parse_blocks(blocks, &p) == PLX_OK);
    return p;
}

/* ---- texture ------------------------------------------------------------------------- */

static void apply(const float A[6], float x, float y, float *u, float *v) {
    *u = A[0] * x + A[1] * y + A[2];
    *v = A[3] * x + A[4] * y + A[5];
}

TEST(sim_affine_hand_points) {
    float A[6], u, v;
    plx_rec_sim_affine((const float[4]){1.1f, 0.2f, 3.0f, -2.0f}, A);
    apply(A, 63.5f, 79.5f, &u, &v); /* c0 -> (124.5 + tx, 128.5 + ty) */
    CHECK_NEAR(u, 127.5, 1e-4);
    CHECK_NEAR(v, 126.5, 1e-4);
    apply(A, 73.5f, 79.5f, &u, &v); /* +10 x: + 10 (a, b) */
    CHECK_NEAR(u, 138.5, 1e-4);
    CHECK_NEAR(v, 128.5, 1e-4);
    apply(A, 63.5f, 89.5f, &u, &v); /* +10 y: + 10 (-b, a) */
    CHECK_NEAR(u, 125.5, 1e-4);
    CHECK_NEAR(v, 137.5, 1e-4);
    /* identity: canonical (x, y) reads image (x + 61, y + 49) */
    plx_rec_sim_affine((const float[4]){1, 0, 0, 0}, A);
    apply(A, 0, 0, &u, &v);
    CHECK_NEAR(u, 61, 1e-5);
    CHECK_NEAR(v, 49, 1e-5);
}

TEST(identity_texture_is_centre_crop) {
    plx_img g = textured(250, 250, 1);
    plx_img tex = plx_img_new(PLX_CANON_W, PLX_CANON_H, 1), conf = plx_img_new(PLX_CANON_W, PLX_CANON_H, 1);
    plx_rec_texture_sim(&g, (const float[4]){1, 0, 0, 0}, &tex, &conf);
    float md = 0, cmin = 1;
    for (int y = 0; y < PLX_CANON_H; y++)
        for (int x = 0; x < PLX_CANON_W; x++) {
            md = fmaxf(md, fabsf(PLX_PX(&tex, x, y, 0) - PLX_PX(&g, x + 61, y + 49, 0)));
            cmin = fminf(cmin, PLX_PX(&conf, x, y, 0));
        }
    CHECK(md < 1e-5f);
    CHECK(cmin == 1.0f);
    /* a translation that pushes the frame off the photo leaves zero confidence there */
    plx_rec_texture_sim(&g, (const float[4]){1, 0, 100, 0}, &tex, &conf);
    CHECK(PLX_PX(&conf, PLX_CANON_W - 1, 80, 0) == 0.0f && PLX_PX(&conf, 0, 80, 0) == 1.0f);
    plx_img_free(&g);
    plx_img_free(&tex);
    plx_img_free(&conf);
}

/* ---- photometric / descriptor invariances -------------------------------------------- */

static plx_img affine_copy(const plx_img *im, float gain, float offset) {
    plx_img c = plx_img_clone(im);
    plx_img_scale_offset(&c, gain, offset);
    return c;
}

TEST(photometric_invariances) {
    plx_img t = textured(PLX_CANON_W, PLX_CANON_H, 2);
    plx_photo_params p = plx_photo_default();
    plx_img a = plx_img_clone(&t), b = affine_copy(&t, 0.6f, 0.0f);
    plx_photo_normalize(&a, NULL, &p);
    plx_photo_normalize(&b, NULL, &p);
    CHECK(plx_img_max_abs_diff(&a, &b) < 2e-4f); /* gamma, DoG and CE together: gain-invariant */
    plx_img_free(&a);
    plx_img_free(&b);
    p.gamma = 1.0f; /* without gamma: offset too */
    a = plx_img_clone(&t);
    b = affine_copy(&t, 0.45f, 0.3f);
    plx_photo_normalize(&a, NULL, &p);
    plx_photo_normalize(&b, NULL, &p);
    CHECK(plx_img_max_abs_diff(&a, &b) < 2e-3f);
    plx_img_free(&a);
    plx_img_free(&b);
    plx_img_free(&t);
}

TEST(descriptor_photometric_invariance) {
    plx_img t = textured(PLX_CANON_W, PLX_CANON_H, 3);
    plx_img g = affine_copy(&t, 0.6f, 0.0f), go = affine_copy(&t, 0.6f, 0.1f);
    plx_desc_params p = params_with("lbp1:8,lbp2:8,lbp2@1:8,ori:8");
    plx_desc_layout L = plx_desc_layout_of(&p);
    float *d0 = PLX_NEW(float, L.d), *d1 = PLX_NEW(float, L.d);
    plx_desc_compute(&p, &t, NULL, d0, NULL);
    plx_desc_compute(&p, &g, NULL, d1, NULL);
    double cg = vec_cos(d0, d1, L.d);
    CHECK(cg > 0.999);
    /* with gamma off, the LBP blocks also ignore an offset; the ORI block is offset-invariant
     * exactly and gain-invariant up to its eps floor */
    p.photo.gamma = 1.0f;
    plx_desc_compute(&p, &t, NULL, d0, NULL);
    plx_desc_compute(&p, &go, NULL, d1, NULL);
    double co = vec_cos(d0, d1, L.d);
    CHECK(co > 0.995);
    printf("  descriptor cosine under gain: %.6f, under gain + offset (gamma off): %.6f\n", cg, co);
    free(d0);
    free(d1);
    plx_img_free(&t);
    plx_img_free(&g);
    plx_img_free(&go);
}

TEST(descriptor_weights) {
    plx_img t = textured(PLX_CANON_W, PLX_CANON_H, 4);
    plx_desc_params p = params_with("lbp1:8,lbp2:8,ori:8");
    plx_desc_layout L = plx_desc_layout_of(&p);
    float *d0 = PLX_NEW(float, L.d), *d1 = PLX_NEW(float, L.d), *w0 = PLX_NEW(float, L.n_cells),
          *w1 = PLX_NEW(float, L.n_cells);
    plx_desc_compute(&p, &t, NULL, d0, w0);
    /* uniform weights, whatever their level, give the unweighted descriptor */
    plx_img conf = plx_img_new(PLX_CANON_W, PLX_CANON_H, 1);
    plx_img_fill(&conf, 0.7f);
    plx_desc_compute(&p, &t, &conf, d1, w1);
    CHECK(max_diff(d0, d1, L.d) < 1e-5f);
    float wmin = 1, wmax = 0;
    for (int c = 0; c < L.n_cells; c++) {
        CHECK(w0[c] == 1.0f);
        wmin = fminf(wmin, w1[c]);
        wmax = fmaxf(wmax, w1[c]);
    }
    CHECK_NEAR(wmin, 0.7, 1e-5);
    CHECK_NEAR(wmax, 0.7, 1e-5);

    /* zero weight on x < 56: whatever the texture is far inside that strip, the LBP blocks
     * cannot see it (LBP reach 2 px + DoG reach 6 px, contrast means are confidence-weighted) */
    plx_desc_params pl = params_with("lbp1:8,lbp2:8");
    plx_desc_layout LL = plx_desc_layout_of(&pl);
    for (int y = 0; y < PLX_CANON_H; y++)
        for (int x = 0; x < PLX_CANON_W; x++) PLX_PX(&conf, x, y, 0) = x < 56 ? 0.0f : 1.0f;
    plx_img t2 = plx_img_clone(&t);
    for (int y = 0; y < PLX_CANON_H; y++)
        for (int x = 0; x < 40; x++) PLX_PX(&t2, x, y, 0) = 0.9f - 0.8f * PLX_PX(&t2, x, y, 0);
    plx_desc_compute(&pl, &t, &conf, d0, w0);
    plx_desc_compute(&pl, &t2, &conf, d1, w1);
    CHECK(max_diff(d0, d1, LL.d) == 0.0f);
    /* cells wholly inside the strip: zero weight, empty histogram */
    int zero_cells = 0;
    for (int i = 0; i < LL.n_blocks; i++)
        for (int c = 0; c < LL.cells_x[i] * LL.cells_y[i]; c++) {
            int cx = c % LL.cells_x[i];
            int x1 = pl.x0 + (cx + 1) * pl.w / LL.cells_x[i]; /* exclusive right edge of the cell */
            if (x1 > 56) continue;
            zero_cells++;
            CHECK(w0[LL.cell_off[i] + c] == 0.0f);
            float s = 0;
            for (int b = 0; b < LL.bins[i]; b++) s += fabsf(d0[LL.off[i] + c * LL.bins[i] + b]);
            CHECK(s == 0.0f);
        }
    CHECK(zero_cells > 0);
    free(d0);
    free(d1);
    free(w0);
    free(w1);
    plx_img_free(&t);
    plx_img_free(&t2);
    plx_img_free(&conf);
}

TEST(block_spec_parsing) {
    plx_desc_params p = params_with("lbp1:8");
    CHECK(plx_desc_parse_blocks("lbp2@1:6,ori:10", &p) == PLX_OK);
    CHECK(p.n_blocks == 2 && p.block[0].kind == PLX_BLOCK_LBP2 && p.block[0].level == 1 && p.block[0].cell == 6);
    CHECK(p.block[1].kind == PLX_BLOCK_ORI && p.block[1].level == 0 && p.block[1].cell == 10);
    CHECK(plx_desc_parse_blocks("lbp3:8", &p) != PLX_OK);
    CHECK(plx_desc_parse_blocks("lbp1@2:8", &p) != PLX_OK);
    CHECK(plx_desc_parse_blocks("lbp1:", &p) != PLX_OK);
    plx_desc_params q = params_with("lbp1:8,ori:12");
    plx_desc_layout L = plx_desc_layout_of(&q);
    CHECK(L.cells_x[0] == 12 && L.cells_y[0] == 14 && L.bins[0] == 59 && L.bins[1] == 8);
    CHECK(L.cells_x[1] == 8 && L.cells_y[1] == 9);
    CHECK(L.d == 12 * 14 * 59 + 8 * 9 * 8 && L.off[1] == 12 * 14 * 59 && L.cell_off[1] == 12 * 14);
}

/* ---- matching algebra on synthetic data ---------------------------------------------- */

/* a one-block, one-cell layout so arbitrary d-vectors can be fed to the matcher */
static plx_desc_layout flat_layout(int d) {
    plx_desc_layout L = {.n_blocks = 1, .d = d, .n_cells = 1};
    L.dim[0] = d;
    L.cells_x[0] = L.cells_y[0] = 1;
    L.bins[0] = d;
    return L;
}

TEST(whitening_cosine_algebra) {
    enum { D = 12, N = 40000 };
    plx_rng r;
    plx_rng_seed(&r, 7);
    /* x = mu + A z with a known, badly conditioned A */
    double A[D][D], mu[D];
    for (int i = 0; i < D; i++) {
        mu[i] = 2.0 + i;
        for (int j = 0; j < D; j++) A[i][j] = (i == j ? 3.0 / (1 + i) : 0.0) + 0.3 * plx_rng_normal(&r) / (1 + j);
    }
    float *X = PLX_NEW(float, (size_t)N * D);
    for (int n = 0; n < N; n++) {
        double z[D];
        for (int j = 0; j < D; j++) z[j] = plx_rng_normal(&r);
        for (int i = 0; i < D; i++) {
            double s = mu[i];
            for (int j = 0; j < D; j++) s += A[i][j] * z[j];
            X[(size_t)n * D + i] = (float)s;
        }
    }
    plx_desc_layout L = flat_layout(D);
    plx_match_cfg c = plx_match_default();
    c.pca_dim = D;
    c.pca_max_train = N;
    plx_feats f = {.n = N, .d = D, .n_cells = 1, .x = X, .cw = NULL};
    plx_match_model m;
    CHECK(plx_match_fit(&c, &L, &f, &m) == PLX_OK);
    CHECK(plx_match_embed_dim(&m, false) == D);
    /* whitened cosine == cosine of A^-1 (x - mu): the true whitening up to a rotation */
    plx_mat Am = plx_mat_new(D, D), Ai = plx_mat_new(D, D);
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++) PLX_M(&Am, i, j) = A[i][j];
    CHECK(plx_inverse(&Am, &Ai) == PLX_OK);
    int a[64], b[64];
    double s[64], worst = 0;
    for (int t = 0; t < 64; t++) {
        a[t] = (int)plx_rng_below(&r, N);
        b[t] = (int)plx_rng_below(&r, N);
    }
    plx_match_score(&m, &f, a, b, 64, false, s);
    for (int t = 0; t < 64; t++) {
        float u[D], v[D];
        for (int i = 0; i < D; i++) {
            double su = 0, sv = 0;
            for (int j = 0; j < D; j++) {
                su += PLX_M(&Ai, i, j) * (X[(size_t)a[t] * D + j] - mu[j]);
                sv += PLX_M(&Ai, i, j) * (X[(size_t)b[t] * D + j] - mu[j]);
            }
            u[i] = (float)su;
            v[i] = (float)sv;
        }
        worst = fmax(worst, fabs(s[t] - vec_cos(u, v, D)));
    }
    printf("  whitened cosine vs analytic whitening: max |diff| %.2e over 64 pairs\n", worst);
    CHECK(worst < 0.03);
    /* embedded rows are unit vectors; a raw-cosine model equals the plain cosine */
    float y[D];
    plx_match_embed(&m, X, NULL, 1, false, y);
    double nn = 0;
    for (int i = 0; i < D; i++) nn += (double)y[i] * y[i];
    CHECK_NEAR(nn, 1.0, 1e-5);
    plx_match_free(&m);
    c.score = PLX_SCORE_COS;
    CHECK(plx_match_fit(&c, &L, &f, &m) == PLX_OK);
    plx_match_score(&m, &f, a, b, 1, false, s);
    CHECK_NEAR(s[0], vec_cos(X + (size_t)a[0] * D, X + (size_t)b[0] * D, D), 1e-9);
    plx_match_free(&m);
    plx_mat_free(&Am);
    plx_mat_free(&Ai);
    free(X);
}

/* Two-Gaussian pair problem: identity mean mu ~ N(0, Sb) lives in a few dims, nuisance
 * eps ~ N(0, Sw) is strong in others. Cosine in the unsupervised whitened space is fooled by the
 * nuisance; the pair-supervised models must learn to ignore it. */
static double auc_of(const double *s, const int *same, int n) {
    double win = 0;
    int np = 0, nn = 0;
    for (int i = 0; i < n; i++) {
        if (!same[i]) continue;
        np++;
        for (int j = 0; j < n; j++)
            if (!same[j]) win += s[i] > s[j] ? 1.0 : s[i] == s[j] ? 0.5 : 0.0;
    }
    for (int j = 0; j < n; j++) nn += !same[j];
    return win / ((double)np * nn);
}

TEST(supervised_separates_two_gaussians) {
    enum { D = 24, IDS = 1200, PER = 4, N = IDS * PER, NP = 3000 };
    plx_rng r;
    plx_rng_seed(&r, 11);
    float *X = PLX_NEW(float, (size_t)N * D);
    for (int id = 0; id < IDS; id++) {
        double mu[D];
        for (int j = 0; j < D; j++) mu[j] = (j < 6 ? 1.0 : 0.15) * plx_rng_normal(&r);
        for (int k = 0; k < PER; k++)
            for (int j = 0; j < D; j++)
                X[((size_t)id * PER + k) * D + j] = (float)(mu[j] + (j < 6 ? 0.5 : 1.5) * plx_rng_normal(&r));
    }
    /* training pairs from the first 800 identities, test pairs from the rest */
    int pa[2 * NP], pb[2 * NP], same[2 * NP];
    for (int t = 0; t < 2 * NP; t++) {
        int lo = t < NP ? 0 : 800, span = t < NP ? 800 : IDS - 800;
        int i1 = lo + (int)plx_rng_below(&r, (uint32_t)span), i2 = i1;
        same[t] = t % 2;
        if (!same[t])
            while (i2 == i1) i2 = lo + (int)plx_rng_below(&r, (uint32_t)span);
        int k1 = (int)plx_rng_below(&r, PER), k2 = (k1 + 1 + (int)plx_rng_below(&r, PER - 1)) % PER;
        pa[t] = i1 * PER + k1;
        pb[t] = i2 * PER + k2;
    }
    plx_desc_layout L = flat_layout(D);
    /* the unsupervised stage sees the training identities' images only */
    plx_feats train = {.n = 800 * PER, .d = D, .n_cells = 1, .x = X};
    plx_feats all = {.n = N, .d = D, .n_cells = 1, .x = X};
    const plx_sup_method methods[4] = {PLX_SUP_NONE, PLX_SUP_LDA, PLX_SUP_JB, PLX_SUP_LOGIT};
    const char *names[4] = {"unsup cosine", "pair-LDA", "joint Bayes", "logistic"};
    double auc[4];
    double *s = PLX_NEW(double, NP);
    float *Y = PLX_NEW(float, (size_t)train.n * D);
    for (int k = 0; k < 4; k++) {
        plx_match_cfg c = plx_match_default();
        c.pca_dim = D;
        c.sup = methods[k];
        c.sup_pca = D;
        c.sup_dim = 12;
        c.sup_ridge = 1e-3;
        plx_match_model m;
        CHECK(plx_match_fit(&c, &L, &train, &m) == PLX_OK);
        bool sup = methods[k] != PLX_SUP_NONE;
        if (sup) {
            plx_match_embed(&m, train.x, NULL, train.n, false, Y);
            CHECK(plx_match_fit_pairs(&m, Y, train.n, NP, pa, pb, same) == PLX_OK);
        }
        plx_match_score(&m, &all, pa + NP, pb + NP, NP, sup, s);
        auc[k] = auc_of(s, same + NP, NP);
        printf("  %-13s test AUC %.4f\n", names[k], auc[k]);
        plx_match_free(&m);
    }
    free(Y);
    for (int k = 1; k < 4; k++) CHECK(auc[k] > auc[0] + 0.05 && auc[k] > 0.9);
    free(s);
    free(X);
}

/* ---- protocol masks on the real pair lists ------------------------------------------- */

static bool exists(const char *p) {
    struct stat st;
    return stat(p, &st) == 0;
}

TEST(protocol_masks) {
    if (!exists("data/lfw") || !exists("data/protocol/pairs.txt")) {
        printf("  (skipped: LFW data not present)\n");
        return;
    }
    plx_lfw db;
    plx_pairs v2, dtr, dte;
    CHECK(plx_lfw_open(&db, "data/lfw") == PLX_OK);
    CHECK(plx_pairs_load(&v2, "data/protocol/pairs.txt", &db) == PLX_OK);
    CHECK(plx_pairs_load(&dtr, "data/protocol/pairsDevTrain.txt", &db) == PLX_OK);
    CHECK(plx_pairs_load(&dte, "data/protocol/pairsDevTest.txt", &db) == PLX_OK);
    /* independent re-derivation: identity names of each fold, compared as strings */
    int leaks = 0, fit_min = db.n_images, ntrain = 0;
    uint8_t *ids = PLX_NEW(uint8_t, db.n_ids);
    for (int k = 0; k < v2.n_folds; k++) {
        plx_rec_split s;
        plx_rec_split_view2(&v2, &db, k, &s);
        memset(ids, 0, (size_t)db.n_ids);
        for (int i = 0; i < v2.n; i++)
            if (v2.p[i].fold == k) {
                ids[plx_lfw_find_id(&db, db.id_name[db.img_id[v2.p[i].a]])] = 1;
                ids[plx_lfw_find_id(&db, db.id_name[db.img_id[v2.p[i].b]])] = 1;
            }
        int nfit = 0;
        for (int i = 0; i < db.n_images; i++) {
            leaks += s.fit_img[i] && ids[db.img_id[i]];
            nfit += s.fit_img[i];
        }
        for (int i = 0; i < s.n_test; i++) leaks += s.fit_img[s.test[i].a] + s.fit_img[s.test[i].b];
        for (int i = 0; i < s.n_train; i++) leaks += s.train[i].fold == k;
        fit_min = PLX_MIN(fit_min, nfit);
        ntrain += s.n_train;
        CHECK(s.n_test == 600 && s.n_train == 5400);
        CHECK(plx_rec_split_check(&s, &db) == 0);
        /* the checker itself must catch a planted leak */
        s.fit_img[s.test[0].a] = 1;
        CHECK(plx_rec_split_check(&s, &db) > 0);
        plx_rec_split_free(&s);
    }
    plx_rec_split s;
    plx_rec_split_view1(&dtr, &dte, &db, &s);
    CHECK(plx_rec_split_check(&s, &db) == 0);
    for (int i = 0; i < s.n_test; i++) leaks += s.fit_img[s.test[i].a] + s.fit_img[s.test[i].b];
    printf("  View 2: >= %d fit images per fold, %d training pairs total; View 1: %d train / %d test pairs\n",
           fit_min, ntrain, s.n_train, s.n_test);
    plx_rec_split_free(&s);
    CHECK(leaks == 0);
    CHECK(fit_min > db.n_images / 2);
    free(ids);
    plx_pairs_free(&v2);
    plx_pairs_free(&dtr);
    plx_pairs_free(&dte);
    plx_lfw_free(&db);
}

int main(void) {
    RUN(sim_affine_hand_points);
    RUN(identity_texture_is_centre_crop);
    RUN(photometric_invariances);
    RUN(descriptor_photometric_invariance);
    RUN(descriptor_weights);
    RUN(block_spec_parsing);
    RUN(whitening_cosine_algebra);
    RUN(supervised_separates_two_gaussians);
    RUN(protocol_masks);
    return TEST_REPORT();
}
