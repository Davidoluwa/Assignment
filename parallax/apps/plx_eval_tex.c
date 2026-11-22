// plx-deps: rec eval data la feat ops viz img core
/* plx_eval_tex: texture-descriptor face verification on LFW (ablation rungs L0 / L1).
 *
 *   plx_eval_tex --pack runs/main/pack [--sim runs/main/congeal/sim.plxt] --out runs/main/rec/<name>
 *                [descriptor / matcher flags] [--view2] [--ablation]
 *
 * Without --sim every image uses the identity similarity: the raw centre crop (L0).
 * Writes into <out>:
 *   desc.plxt, cellw.plxt, desc.cfg   descriptor cache [N, D] / [N, cells] (reused when desc.cfg matches)
 *   report.txt                        configuration + plx_verif_print blocks: View 1 (and View 2
 *                                     with --view2), unsupervised and image-restricted paradigms
 *   roc_<view>_<paradigm>.csv, scores_<view>_<paradigm>.csv
 *   textures.png                      canonical textures before / after photometric normalisation
 *   fails_<view>_<paradigm>.png       most confident false matches and false rejects
 *   ablation.txt                      with --ablation: View 1 ingredient table (View 2 untouched)
 *
 * Descriptor flags: --blocks SPEC (see plx_desc_parse_blocks), --region x0,y0,w,h, --nosqrt,
 *   --nophoto, --gamma G, --dog0 S, --dog1 S, --tau T, --ori-k K, --ori-unsigned
 * Matcher flags: --score embed|chi2|cos|cell, --pca K, --pca-n N, --global-pca, --nowhiten,
 *   --pca-iters Q, --sup lda|jb|logit, --sup-pca R, --sup-dim M, --ridge R, --l2 L
 * Defaults are the best View-1 configuration. */
#include "core/core.h"
#include "data/lfw.h"
#include "eval/verif.h"
#include "img/imgio.h"
#include "ops/ops.h"
#include "rec/match.h"
#include "rec/proto.h"
#include "rec/rec.h"
#include "viz/viz.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

enum { SIDE = 250, PIXELS = SIDE * SIDE };

/* ---- configuration --------------------------------------------------------------------- */

static const char *const DEFAULT_BLOCKS = "lbp1:8,lbp2:8,lbp2@1:8,ori:8";

static plx_desc_params desc_default(void) {
    plx_desc_params p = {.x0 = 16, .y0 = 40, .w = 96, .h = 112, .sqrt_hist = true, .photo = plx_photo_default(),
                         .orient = plx_orient_default()};
    if (plx_desc_parse_blocks(DEFAULT_BLOCKS, &p) != PLX_OK) plx_die("bad default block spec");
    return p;
}

static plx_desc_params desc_from_args(const plx_args *a) {
    plx_desc_params p = desc_default();
    if (plx_desc_parse_blocks(plx_arg_str(a, "blocks", DEFAULT_BLOCKS), &p) != PLX_OK) plx_die("bad --blocks");
    const char *reg = plx_arg_str(a, "region", NULL);
    if (reg && sscanf(reg, "%d,%d,%d,%d", &p.x0, &p.y0, &p.w, &p.h) != 4) plx_die("bad --region (x0,y0,w,h)");
    if (plx_arg_flag(a, "nosqrt")) p.sqrt_hist = false;
    if (plx_arg_flag(a, "nophoto")) p.photo = plx_photo_none();
    p.photo.gamma = (float)plx_arg_num(a, "gamma", p.photo.gamma);
    p.photo.dog_s0 = (float)plx_arg_num(a, "dog0", p.photo.dog_s0);
    p.photo.dog_s1 = (float)plx_arg_num(a, "dog1", p.photo.dog_s1);
    p.photo.ce_tau = (float)plx_arg_num(a, "tau", p.photo.ce_tau);
    p.orient.k = (int)plx_arg_int(a, "ori-k", p.orient.k);
    if (plx_arg_flag(a, "ori-unsigned")) p.orient.signed_ = false;
    return p;
}

static plx_match_cfg match_default(void) {
    plx_match_cfg c = plx_match_default();
    c.sup = PLX_SUP_JB;
    return c;
}

static plx_match_cfg match_from_args(const plx_args *a) {
    plx_match_cfg c = match_default();
    const char *sc = plx_arg_str(a, "score", "embed");
    if (!strcmp(sc, "embed")) c.score = PLX_SCORE_EMBED;
    else if (!strcmp(sc, "chi2")) c.score = PLX_SCORE_CHI2;
    else if (!strcmp(sc, "cos")) c.score = PLX_SCORE_COS;
    else if (!strcmp(sc, "cell")) c.score = PLX_SCORE_CELL;
    else plx_die("bad --score %s", sc);
    c.pca_dim = (int)plx_arg_int(a, "pca", c.pca_dim);
    c.pca_max_train = (int)plx_arg_int(a, "pca-n", c.pca_max_train);
    if (plx_arg_flag(a, "global-pca")) c.per_block = false;
    if (plx_arg_flag(a, "nowhiten")) c.whiten = false;
    const char *sup = plx_arg_str(a, "sup", NULL);
    if (sup) {
        if (!strcmp(sup, "lda")) c.sup = PLX_SUP_LDA;
        else if (!strcmp(sup, "jb")) c.sup = PLX_SUP_JB;
        else if (!strcmp(sup, "logit")) c.sup = PLX_SUP_LOGIT;
        else plx_die("bad --sup %s", sup);
    }
    c.sup_pca = (int)plx_arg_int(a, "sup-pca", c.sup_pca);
    c.sup_dim = (int)plx_arg_int(a, "sup-dim", c.sup_dim);
    c.pca_iters = (int)plx_arg_int(a, "pca-iters", c.pca_iters);
    c.sup_ridge = plx_arg_num(a, "ridge", c.sup_ridge);
    c.logit_l2 = plx_arg_num(a, "l2", c.logit_l2);
    return c;
}

/* ---- data -------------------------------------------------------------------------------- */

typedef struct {
    plx_lfw db;
    plx_pairs view2, dev_train, dev_test;
    plx_tensor gray;     /* u8 [N, 250, 250], mapped */
    float *sim;          /* [N, 4] */
    const char *sim_path;
} dataset;

static void dataset_open(dataset *ds, const plx_args *a) {
    const char *lfw = plx_arg_str(a, "lfw", "data/lfw");
    const char *proto = plx_arg_str(a, "protocol", "data/protocol");
    const char *pack = plx_arg_str(a, "pack", "runs/main/pack");
    char path[4096];
    if (plx_lfw_open(&ds->db, lfw) != PLX_OK) plx_die("cannot open LFW index at %s", lfw);
    const char *names[3] = {"pairs.txt", "pairsDevTrain.txt", "pairsDevTest.txt"};
    plx_pairs *dst[3] = {&ds->view2, &ds->dev_train, &ds->dev_test};
    for (int i = 0; i < 3; i++) {
        snprintf(path, sizeof path, "%s/%s", proto, names[i]);
        if (plx_pairs_load(dst[i], path, &ds->db) != PLX_OK) plx_die("cannot load %s", path);
    }
    snprintf(path, sizeof path, "%s/gray.plxt", pack);
    if (plx_tensor_map(path, &ds->gray) != PLX_OK) plx_die("cannot map %s", path);
    int n = ds->db.n_images;
    if (ds->gray.dtype != PLX_U8 || ds->gray.ndim != 3 || ds->gray.dim[0] != n || ds->gray.dim[1] != SIDE ||
        ds->gray.dim[2] != SIDE)
        plx_die("%s: expected u8 [%d, 250, 250]", path, n);
    ds->sim = PLX_NEW(float, (size_t)n * 4);
    ds->sim_path = plx_arg_str(a, "sim", NULL);
    if (ds->sim_path) {
        plx_tensor t;
        if (plx_tensor_load(ds->sim_path, &t) != PLX_OK) plx_die("cannot load %s", ds->sim_path);
        if (t.dtype != PLX_F32 || t.ndim != 2 || t.dim[0] != n || t.dim[1] != 4)
            plx_die("%s: expected f32 [%d, 4]", ds->sim_path, n);
        memcpy(ds->sim, t.data, sizeof(float) * (size_t)n * 4);
        plx_tensor_free(&t);
    } else {
        for (int i = 0; i < n; i++) ds->sim[4 * i] = 1.0f; /* identity: raw centre crop */
    }
}

static void dataset_close(dataset *ds) {
    plx_pairs_free(&ds->view2);
    plx_pairs_free(&ds->dev_train);
    plx_pairs_free(&ds->dev_test);
    plx_tensor_free(&ds->gray);
    free(ds->sim);
    plx_lfw_free(&ds->db);
}

/* The texture source of rungs L0 / L1. Later rungs replace exactly this function. */
static void texture_of(const dataset *ds, int i, plx_img *tex, plx_img *conf) {
    const uint8_t *g = (const uint8_t *)ds->gray.data + (size_t)i * PIXELS;
    plx_img im = plx_img_from_u8(g, SIDE, SIDE, 1, 1.0f / 255.0f);
    plx_rec_texture_sim(&im, ds->sim + 4 * (size_t)i, tex, conf);
    plx_img_free(&im);
}

/* ---- descriptors ------------------------------------------------------------------------- */

typedef struct {
    const dataset *ds;
    const plx_desc_params *p;
    plx_desc_layout lay;
    float *x, *cw;
} desc_job;

static void desc_one(int64_t i, int tid, void *ctx) {
    desc_job *j = ctx;
    plx_img tex = plx_img_new(PLX_CANON_W, PLX_CANON_H, 1), conf = plx_img_new(PLX_CANON_W, PLX_CANON_H, 1);
    texture_of(j->ds, (int)i, &tex, &conf);
    plx_desc_compute(j->p, &tex, &conf, j->x + (size_t)i * j->lay.d, j->cw + (size_t)i * j->lay.n_cells);
    plx_img_free(&tex);
    plx_img_free(&conf);
}

/* desc.cfg holds the descriptor description and the texture source; equal text = reusable cache */
static void cache_key(const dataset *ds, const plx_desc_params *p, char *buf, size_t cap) {
    char d[1024];
    plx_desc_describe(p, d, sizeof d);
    snprintf(buf, cap, "%s sim=%s\n", d, ds->sim_path ? ds->sim_path : "identity");
}

static bool cache_load(const char *out, const char *key, const plx_desc_layout *lay, int n, float **x, float **cw) {
    char path[4096];
    uint8_t *txt = NULL;
    size_t len = 0;
    snprintf(path, sizeof path, "%s/desc.cfg", out);
    if (plx_read_file(path, &txt, &len) != PLX_OK) return false;
    bool same = len == strlen(key) && memcmp(txt, key, len) == 0;
    free(txt);
    if (!same) return false;
    plx_tensor t1, t2;
    snprintf(path, sizeof path, "%s/desc.plxt", out);
    if (plx_tensor_load(path, &t1) != PLX_OK) return false;
    snprintf(path, sizeof path, "%s/cellw.plxt", out);
    if (plx_tensor_load(path, &t2) != PLX_OK) {
        plx_tensor_free(&t1);
        return false;
    }
    bool ok = t1.dtype == PLX_F32 && t1.ndim == 2 && t1.dim[0] == n && t1.dim[1] == lay->d && t2.dtype == PLX_F32 &&
              t2.ndim == 2 && t2.dim[0] == n && t2.dim[1] == lay->n_cells;
    if (ok) {
        *x = t1.data;
        *cw = t2.data;
    } else {
        plx_tensor_free(&t1);
        plx_tensor_free(&t2);
    }
    return ok;
}

static void descriptors(const dataset *ds, const plx_desc_params *p, const char *out, bool use_cache, float **x,
                        float **cw) {
    plx_desc_layout lay = plx_desc_layout_of(p);
    int n = ds->db.n_images;
    char key[1200];
    cache_key(ds, p, key, sizeof key);
    if (use_cache && cache_load(out, key, &lay, n, x, cw)) {
        plx_log("descriptor cache hit: %d x %d", n, lay.d);
        return;
    }
    desc_job j = {.ds = ds, .p = p, .lay = lay, .x = plx_malloc(sizeof(float) * (size_t)n * lay.d),
                  .cw = plx_malloc(sizeof(float) * (size_t)n * lay.n_cells)};
    double t0 = plx_now();
    plx_parallel_for(n, 16, desc_one, &j);
    plx_log("descriptors: %d images x %d dims (%d cells) in %.1fs on %d threads", n, lay.d, lay.n_cells,
            plx_now() - t0, plx_num_threads());
    if (use_cache) {
        char path[4096];
        snprintf(path, sizeof path, "%s/desc.plxt", out);
        if (plx_tensor_save(path, PLX_F32, 2, (int64_t[]){n, lay.d}, j.x) != PLX_OK) plx_die("write failed: %s", path);
        snprintf(path, sizeof path, "%s/cellw.plxt", out);
        if (plx_tensor_save(path, PLX_F32, 2, (int64_t[]){n, lay.n_cells}, j.cw) != PLX_OK)
            plx_die("write failed: %s", path);
        snprintf(path, sizeof path, "%s/desc.cfg", out);
        if (plx_write_file(path, key, strlen(key)) != PLX_OK) plx_die("write failed: %s", path);
    }
    *x = j.x;
    *cw = j.cw;
}

/* ---- diagnostics ------------------------------------------------------------------------- */

static void save_png(const char *out, const char *name, const plx_img *im) {
    char path[4096];
    snprintf(path, sizeof path, "%s/%s", out, name);
    if (plx_img_save(path, im) != PLX_OK) plx_die("write failed: %s", path);
    plx_log("wrote %s", path);
}

/* a normalised texture mapped to [0, 1] for display with its 1%..99% range */
static plx_img display_norm(const plx_img *v) {
    float lo, hi;
    plx_viz_auto_range(v->px, (size_t)v->w * v->h, 0.01f, 0.99f, &lo, &hi);
    plx_img d = plx_img_clone(v);
    plx_img_scale_offset(&d, 1.0f / (hi - lo), -lo / (hi - lo));
    for (size_t i = 0; i < (size_t)d.w * d.h; i++) d.px[i] = PLX_CLAMP(d.px[i], 0.0f, 1.0f);
    return d;
}

static void texture_montage(const dataset *ds, const plx_desc_params *p, const char *out) {
    enum { NIMG = 12 };
    plx_img tiles[2 * NIMG];
    char lab[2 * NIMG][32];
    const char *labels[2 * NIMG];
    for (int t = 0; t < NIMG; t++) {
        /* spread over the dev-test pairs: a mix of people, lighting and pose */
        const plx_pair *pr = &ds->dev_test.p[(size_t)t * ds->dev_test.n / NIMG];
        int img = t % 2 ? pr->b : pr->a;
        plx_img tex = plx_img_new(PLX_CANON_W, PLX_CANON_H, 1), conf = plx_img_new(PLX_CANON_W, PLX_CANON_H, 1);
        texture_of(ds, img, &tex, &conf);
        plx_img norm = plx_img_clone(&tex);
        plx_photo_normalize(&norm, &conf, &p->photo);
        tiles[2 * t] = tex;
        tiles[2 * t + 1] = display_norm(&norm);
        plx_img_free(&norm);
        plx_img_free(&conf);
        snprintf(lab[2 * t], sizeof lab[0], "#%d raw", img);
        snprintf(lab[2 * t + 1], sizeof lab[0], "normalised");
        labels[2 * t] = lab[2 * t];
        labels[2 * t + 1] = lab[2 * t + 1];
    }
    plx_montage_opts o = plx_montage_defaults();
    o.cols = 8;
    plx_img m = plx_viz_montage(tiles, 2 * NIMG, labels, &o);
    plx_img big = plx_viz_upscale(&m, 2, PLX_UP_NEAREST);
    save_png(out, "textures.png", &big);
    plx_img_free(&m);
    plx_img_free(&big);
    for (int t = 0; t < 2 * NIMG; t++) plx_img_free(&tiles[t]);
}

/* the two canonical textures side by side, with the descriptor region outlined */
static plx_img pair_tile(const dataset *ds, const plx_desc_params *p, int a, int b) {
    plx_img tile = plx_img_new(2 * PLX_CANON_W + 4, PLX_CANON_H, 3);
    plx_img tex = plx_img_new(PLX_CANON_W, PLX_CANON_H, 1), conf = plx_img_new(PLX_CANON_W, PLX_CANON_H, 1);
    const float red[3] = {0.9f, 0.3f, 0.2f};
    for (int s = 0; s < 2; s++) {
        texture_of(ds, s ? b : a, &tex, &conf);
        int ox = s * (PLX_CANON_W + 4);
        for (int k = 0; k < 3; k++)
            for (int y = 0; y < PLX_CANON_H; y++)
                for (int x = 0; x < PLX_CANON_W; x++) PLX_PX(&tile, ox + x, y, k) = PLX_PX(&tex, x, y, 0);
        plx_draw_rect_outline(&tile, ox + p->x0 - 0.5f, p->y0 - 0.5f, ox + p->x0 + p->w - 0.5f, p->y0 + p->h - 0.5f,
                              1.0f, red, 0.6f);
    }
    plx_img_free(&tex);
    plx_img_free(&conf);
    return tile;
}

typedef struct { int i; double margin; } ranked;
static int ranked_desc(const void *x, const void *y) {
    double a = ((const ranked *)x)->margin, b = ((const ranked *)y)->margin;
    return (a < b) - (a > b);
}

/* Most confident errors: largest |score - threshold| on the wrong side. thr(i) is the threshold
 * the protocol applied to pair i. Top rows false matches, bottom rows false rejects. */
static void failure_montage(const dataset *ds, const plx_desc_params *p, const plx_pair *pr, int n,
                            const double *score, const double *thr, const char *out, const char *name) {
    enum { SHOW = 8 };
    ranked *fm = PLX_NEW(ranked, n), *fr = PLX_NEW(ranked, n);
    int nfm = 0, nfr = 0;
    for (int i = 0; i < n; i++) {
        bool said_same = score[i] > thr[i];
        if (said_same && !pr[i].same) fm[nfm++] = (ranked){i, score[i] - thr[i]};
        if (!said_same && pr[i].same) fr[nfr++] = (ranked){i, thr[i] - score[i]};
    }
    qsort(fm, (size_t)nfm, sizeof *fm, ranked_desc);
    qsort(fr, (size_t)nfr, sizeof *fr, ranked_desc);
    plx_img tiles[2 * SHOW];
    char lab[2 * SHOW][96];
    const char *labels[2 * SHOW];
    int nt = 0;
    for (int s = 0; s < 2; s++) {
        ranked *r = s ? fr : fm;
        int cnt = PLX_MIN(SHOW, s ? nfr : nfm);
        for (int t = 0; t < cnt; t++) {
            const plx_pair *q = &pr[r[t].i];
            tiles[nt] = pair_tile(ds, p, q->a, q->b);
            char na[24], nb[24];
            snprintf(na, sizeof na, "%.20s", ds->db.id_name[ds->db.img_id[q->a]]);
            snprintf(nb, sizeof nb, "%.20s", ds->db.id_name[ds->db.img_id[q->b]]);
            snprintf(lab[nt], sizeof lab[0], "%s %+.3g\n%s\n%s", s ? "false reject" : "false match",
                     score[r[t].i] - thr[r[t].i], na, nb);
            labels[nt] = lab[nt];
            nt++;
        }
    }
    if (nt > 0) {
        plx_montage_opts o = plx_montage_defaults();
        o.cols = 4;
        plx_img m = plx_viz_montage(tiles, nt, labels, &o);
        plx_img big = plx_viz_upscale(&m, 2, PLX_UP_NEAREST);
        save_png(out, name, &big);
        plx_img_free(&m);
        plx_img_free(&big);
    }
    for (int t = 0; t < nt; t++) plx_img_free(&tiles[t]);
    free(fm);
    free(fr);
}

/* ---- evaluation ------------------------------------------------------------------------- */

typedef struct {
    const dataset *ds;
    const plx_desc_params *dp;
    plx_desc_layout lay;
    plx_feats all;
    const char *out;
    FILE *report;
    bool png;
} evaluator;

static const char *const PAR_NAME[2] = {"unsup", "restricted"}; /* paradigm 0, 1 */

static void write_outputs(const evaluator *ev, const char *view, int par, const plx_pair *p, int n,
                          const double *score, const double *thr) {
    char path[4096];
    snprintf(path, sizeof path, "%s/roc_%s_%s.csv", ev->out, view, PAR_NAME[par]);
    if (plx_verif_write_roc(path, p, n, score) != PLX_OK) plx_die("write failed: %s", path);
    snprintf(path, sizeof path, "%s/scores_%s_%s.csv", ev->out, view, PAR_NAME[par]);
    if (plx_verif_write_scores(path, p, n, score) != PLX_OK) plx_die("write failed: %s", path);
    if (ev->png) {
        char name[64];
        snprintf(name, sizeof name, "fails_%s_%s.png", view, PAR_NAME[par]);
        failure_montage(ev->ds, ev->dp, p, n, score, thr, ev->out, name);
    }
}

/* One view, both paradigms (restricted only when the matcher has a supervised stage). */
static void run_view(const evaluator *ev, const plx_match_cfg *cfg, bool view2) {
    const dataset *ds = ev->ds;
    const plx_pairs *pl = view2 ? &ds->view2 : &ds->dev_test;
    bool sup = cfg->score == PLX_SCORE_EMBED && cfg->sup != PLX_SUP_NONE;
    double *score[2] = {PLX_NEW(double, pl->n), sup ? PLX_NEW(double, pl->n) : NULL};
    plx_verif_report r[2];
    double t0 = plx_now();
    int rc = view2 ? plx_rec_eval_view2(cfg, &ev->lay, &ev->all, pl, &ds->db, score[0], score[1], &r[0], &r[1])
                   : plx_rec_eval_view1(cfg, &ev->lay, &ev->all, &ds->dev_train, pl, &ds->db, score[0], score[1],
                                        &r[0], &r[1]);
    if (rc != PLX_OK) plx_die("evaluation failed: %s", plx_la_status_str(rc));
    double dt = plx_now() - t0;
    const char *view = view2 ? "view2" : "view1";
    double *thr = PLX_NEW(double, pl->n);
    for (int par = 0; par < (sup ? 2 : 1); par++) {
        for (int i = 0; i < pl->n; i++) thr[i] = r[par].fold_thr[view2 ? pl->p[i].fold : 0];
        write_outputs(ev, view, par, pl->p, pl->n, score[par], thr);
        char label[64], mdesc[512];
        snprintf(label, sizeof label, "%s %s", view2 ? "View 2" : "View 1",
                 par ? "image-restricted" : "unsupervised");
        plx_match_describe(cfg, par == 1, mdesc, sizeof mdesc);
        fprintf(ev->report, "# matcher: %s\n", mdesc);
        plx_verif_print(ev->report, label, &r[par]);
        fprintf(ev->report, "\n");
        plx_verif_print(stderr, label, &r[par]);
    }
    fprintf(ev->report, "# %s: both paradigms in %.1fs\n\n", view, dt);
    fflush(ev->report);
    free(thr);
    free(score[0]);
    free(score[1]);
}

int main(int argc, char **argv) {
    plx_args a = {argc, argv};
    const char *out = plx_arg_str(&a, "out", NULL);
    if (!out) plx_die("usage: plx_eval_tex --pack DIR [--sim FILE] --out DIR [flags]  (see the source header)");
    if (plx_mkdir_p(out) != PLX_OK) plx_die("cannot create %s", out);
    double t_start = plx_now();

    dataset ds = {0};
    dataset_open(&ds, &a);
    plx_desc_params dp = desc_from_args(&a);
    plx_match_cfg mc = match_from_args(&a);

    float *x = NULL, *cw = NULL;
    descriptors(&ds, &dp, out, true, &x, &cw);
    plx_desc_layout lay = plx_desc_layout_of(&dp);

    char path[4096];
    snprintf(path, sizeof path, "%s/report.txt", out);
    FILE *rep = fopen(path, "w");
    if (!rep) plx_die("cannot write %s", path);
    char ddesc[1024];
    plx_desc_describe(&dp, ddesc, sizeof ddesc);
    fprintf(rep, "plx_eval_tex report\n# texture: %s\n# descriptor: %s  (D = %d, %d cells)\n\n",
            ds.sim_path ? ds.sim_path : "identity similarity (raw centre crop, L0)", ddesc, lay.d, lay.n_cells);

    evaluator ev = {.ds = &ds, .dp = &dp, .lay = lay, .all = {.n = ds.db.n_images, .d = lay.d, .n_cells = lay.n_cells,
                    .x = x, .cw = cw}, .out = out, .report = rep, .png = !plx_arg_flag(&a, "no-png")};
    if (ev.png) texture_montage(&ds, &dp, out);

    run_view(&ev, &mc, false);
    if (plx_arg_flag(&a, "view2")) run_view(&ev, &mc, true);
    fprintf(rep, "# total %.1fs\n", plx_now() - t_start);
    fclose(rep);
    plx_log("wrote %s/report.txt", out);
    free(x);
    free(cw);
    dataset_close(&ds);
    return 0;
}
