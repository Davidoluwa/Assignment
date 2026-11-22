// plx-deps: rec align render eval data la feat ops viz img core
/* plx_eval_tex: texture-descriptor face verification on LFW (ablation rungs L0 / L1).
 *
 *   plx_eval_tex --pack runs/main/pack [--sim runs/main/congeal/sim.plxt | --flow runs/main/flow]
 *                --out runs/main/rec/<name> [texture / descriptor / matcher flags] [--view2]
 *                [--ablation ladder|flow]
 *
 * Texture source: without --sim or --flow every image uses the identity similarity, the raw
 * centre crop (L0); --sim is the similarity alignment (L1); --flow DIR resamples through the
 * dense flow W(p) = T(p + alpha u(p)) with DIR's own u.plxt, sim.plxt and quality.plxt (L2).
 * Texture flags: --alpha A (fraction of the flow applied, default 1), --jconf P (confidence *=
 *   clamp(det(I + alpha grad u), 0, 1)^P; 0 = off; --jconf-mirror uses det(p) / det(mirror of p)
 *   instead, i.e. only pose-driven asymmetric stretch), --fallback BITS (images whose flow quality
 *   flags intersect BITS use alpha = 0), --conf FILE (f32 [N, 160, 128] confidence planes from
 *   outside, multiplied in: the seam for depth-derived visibility), --codes FILE (f32 [N, K]
 *   non-texture codes appended to every descriptor as one more block: the seam for geometry codes),
 *   --pose FILE (geom pose.plxt) with --vis P: confidence *= visibility of the canonical surface
 *   --depth FILE (default runs/main/geom/zbar_dense.plxt) under the photo's yaw and pitch, ^P
 *   (plx_rec_visibility; --vis-abs: absolute normal cosine, --vis-yaw-only: pitch taken as 0).
 * Writes into <out>:
 *   desc.plxt, cellw.plxt, desc.cfg   descriptor cache [N, D] / [N, cells] (reused when desc.cfg matches)
 *   report.txt                        configuration + plx_verif_print blocks: View 1 (and View 2
 *                                     with --view2), unsupervised and image-restricted paradigms
 *   roc_<view>_<paradigm>.csv, scores_<view>_<paradigm>.csv
 *   textures.png                      canonical textures, photometrically normalised, confidence
 *   fails_<view>_<paradigm>.png       most confident false matches and false rejects
 *   ablation_<table>.txt              with --ablation (a separate mode: nothing else is run or
 *                                     written): a View 1 table, View 2 untouched
 *
 * Descriptor flags: --blocks SPEC (see plx_desc_parse_blocks), --region x0,y0,w,h, --nosqrt, --nomirror, --mirror-weighted,
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

static const char *const DEFAULT_BLOCKS = "lbp1:12,lbp2:12,lbp2@1:8,ori:12";

static plx_desc_params desc_default(void) {
    plx_desc_params p = {.x0 = 16, .y0 = 40, .w = 96, .h = 112, .sqrt_hist = true, .mirror = true,
                         .photo = plx_photo_default(), .orient = plx_orient_default()};
    if (plx_desc_parse_blocks(DEFAULT_BLOCKS, &p) != PLX_OK) plx_die("bad default block spec");
    return p;
}

static plx_desc_params desc_from_args(const plx_args *a) {
    plx_desc_params p = desc_default();
    if (plx_desc_parse_blocks(plx_arg_str(a, "blocks", DEFAULT_BLOCKS), &p) != PLX_OK) plx_die("bad --blocks");
    const char *reg = plx_arg_str(a, "region", NULL);
    if (reg && sscanf(reg, "%d,%d,%d,%d", &p.x0, &p.y0, &p.w, &p.h) != 4) plx_die("bad --region (x0,y0,w,h)");
    if (plx_arg_flag(a, "nosqrt")) p.sqrt_hist = false;
    if (plx_arg_flag(a, "nomirror")) p.mirror = false;
    if (plx_arg_flag(a, "mirror-weighted")) p.mirror_weighted = true;
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
    plx_tensor gray;       /* u8 [N, 250, 250], mapped */
    float *sim;            /* [N, 4] */
    const char *sim_path;  /* NULL: identity */
    float *u;              /* [N, PLX_REC_FLOW_DOF] dense flow, or NULL (rungs L0 / L1) */
    int *qflags;           /* [N] flow quality flag bits (flow/quality.plxt column 6), or NULL */
    const char *flow_dir;
    plx_tensor conf;       /* optional f32 [N, 160, 128] confidence planes from outside (mapped) */
    const char *conf_path;
    float *codes;          /* optional f32 [N, code_dim] non-texture codes appended as one block */
    int code_dim;
    const char *codes_path;
    float *pose;           /* optional f32 [N, 9] geometry pose (geom/pose.plxt), for visibility */
    plx_img depth;         /* canonical depth map for visibility (zbar_dense.plxt layout) */
    const char *pose_path, *depth_path;
} dataset;

enum { POSE_COLS = 9, POSE_YAW = 2, POSE_PITCH = 3, POSE_FLAGS = 8 };
/* geom pose flags: 1 untrusted by flow quality, 2 rank-1 outlier: the pose estimate is not used */
enum { POSE_UNTRUSTED = 1 | 2 };

/* How a texture is made from the data: everything per-configuration (ablation rows differ). */
typedef struct {
    float alpha;     /* fraction of the flow field applied */
    float jpow;      /* > 0: confidence *= clamp(det(I + alpha grad u), 0, 1)^jpow */
    int   fallback;  /* images whose flow quality flags intersect this mask use alpha = 0 */
    bool  jmirror;   /* stretch relative to the mirror partner (pose only, not face width) */
    float vpow;      /* > 0: confidence *= visibility of the 3D surface under the photo's pose ^ vpow */
    bool  vabs;      /* absolute normal cosine instead of foreshortening relative to frontal */
    bool  vyaw;      /* yaw only (pitch set to 0: the pitch estimate is weak) */
} tex_opts;

static tex_opts tex_from_args(const plx_args *a) {
    return (tex_opts){.alpha = (float)plx_arg_num(a, "alpha", 1.0), .jpow = (float)plx_arg_num(a, "jconf", 0.0),
                      .fallback = (int)plx_arg_int(a, "fallback", 0), .jmirror = plx_arg_flag(a, "jconf-mirror"),
                      .vpow = (float)plx_arg_num(a, "vis", 0.0), .vabs = plx_arg_flag(a, "vis-abs"),
                      .vyaw = plx_arg_flag(a, "vis-yaw-only")};
}

static float *load_f32(const char *path, int64_t rows, int64_t cols, int64_t *cols_out) {
    plx_tensor t;
    if (plx_tensor_load(path, &t) != PLX_OK) plx_die("cannot load %s", path);
    int64_t c = t.ndim >= 2 ? t.nbytes / sizeof(float) / (size_t)t.dim[0] : 1;
    if (t.dtype != PLX_F32 || t.dim[0] != rows || (cols > 0 && c != cols))
        plx_die("%s: expected f32 [%lld, %lld]", path, (long long)rows, (long long)cols);
    if (cols_out) *cols_out = c;
    return t.data;
}

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

    /* --flow DIR brings its own similarity: the fields are only meaningful with the T they were
     * solved with */
    ds->flow_dir = plx_arg_str(a, "flow", NULL);
    static char flow_sim[4096];
    if (ds->flow_dir) snprintf(flow_sim, sizeof flow_sim, "%s/sim.plxt", ds->flow_dir);
    ds->sim_path = ds->flow_dir ? flow_sim : plx_arg_str(a, "sim", NULL);
    if (ds->sim_path) {
        ds->sim = load_f32(ds->sim_path, n, 4, NULL);
    } else {
        ds->sim = PLX_NEW(float, (size_t)n * 4);
        for (int i = 0; i < n; i++) ds->sim[4 * i] = 1.0f; /* identity: raw centre crop */
    }
    if (ds->flow_dir) {
        snprintf(path, sizeof path, "%s/u.plxt", ds->flow_dir);
        ds->u = load_f32(path, n, PLX_REC_FLOW_DOF, NULL);
        snprintf(path, sizeof path, "%s/quality.plxt", ds->flow_dir);
        float *q = load_f32(path, n, 7, NULL);
        ds->qflags = PLX_NEW(int, n);
        for (int i = 0; i < n; i++) ds->qflags[i] = (int)q[7 * (size_t)i + 6];
        free(q);
    }
    ds->conf_path = plx_arg_str(a, "conf", NULL);
    if (ds->conf_path) {
        if (plx_tensor_map(ds->conf_path, &ds->conf) != PLX_OK) plx_die("cannot map %s", ds->conf_path);
        if (ds->conf.dtype != PLX_F32 || ds->conf.ndim != 3 || ds->conf.dim[0] != n ||
            ds->conf.dim[1] != PLX_REC_TEX_H || ds->conf.dim[2] != PLX_REC_TEX_W)
            plx_die("%s: expected f32 [%d, %d, %d]", ds->conf_path, n, PLX_REC_TEX_H, PLX_REC_TEX_W);
    }
    ds->pose_path = plx_arg_str(a, "pose", NULL);
    if (ds->pose_path) {
        ds->pose = load_f32(ds->pose_path, n, POSE_COLS, NULL);
        ds->depth_path = plx_arg_str(a, "depth", "runs/main/geom/zbar_dense.plxt");
        plx_tensor t;
        if (plx_tensor_load(ds->depth_path, &t) != PLX_OK) plx_die("cannot load %s", ds->depth_path);
        if (t.dtype != PLX_F32 || t.ndim != 2 || t.dim[0] != PLX_REC_TEX_H || t.dim[1] != PLX_REC_TEX_W)
            plx_die("%s: expected f32 [%d, %d]", ds->depth_path, PLX_REC_TEX_H, PLX_REC_TEX_W);
        ds->depth = plx_img_new(PLX_REC_TEX_W, PLX_REC_TEX_H, 1);
        memcpy(ds->depth.px, t.data, t.nbytes);
        plx_tensor_free(&t);
    }
    ds->codes_path = plx_arg_str(a, "codes", NULL);
    if (ds->codes_path) {
        int64_t dim;
        ds->codes = load_f32(ds->codes_path, n, -1, &dim);
        ds->code_dim = (int)dim;
    }
}

static void dataset_close(dataset *ds) {
    plx_pairs_free(&ds->view2);
    plx_pairs_free(&ds->dev_train);
    plx_pairs_free(&ds->dev_test);
    plx_tensor_free(&ds->gray);
    if (ds->conf_path) plx_tensor_free(&ds->conf);
    free(ds->sim);
    free(ds->u);
    free(ds->qflags);
    free(ds->codes);
    free(ds->pose);
    plx_img_free(&ds->depth);
    plx_lfw_free(&ds->db);
}

/* The texture source: L0 / L1 resample through the similarity, L2 through the dense flow. The
 * confidence plane is the in-image mask times, optionally, the flow's stretch confidence and a
 * plane supplied from outside (the seam for depth-derived visibility). */
static void texture_of(const dataset *ds, const tex_opts *to, int i, plx_img *tex, plx_img *conf) {
    const uint8_t *g = (const uint8_t *)ds->gray.data + (size_t)i * PIXELS;
    plx_img im = plx_img_from_u8(g, SIDE, SIDE, 1, 1.0f / 255.0f);
    const float *sim = ds->sim + 4 * (size_t)i;
    if (ds->u) {
        const float *u = ds->u + (size_t)i * PLX_REC_FLOW_DOF;
        float alpha = ds->qflags[i] & to->fallback ? 0.0f : to->alpha;
        plx_rec_texture_flow(&im, sim, u, alpha, tex, conf);
        if (to->jpow > 0.0f && alpha > 0.0f) {
            plx_img jdet = plx_img_new(PLX_REC_TEX_W, PLX_REC_TEX_H, 1);
            plx_rec_flow_jdet(u, alpha, &jdet);
            if (to->jmirror) plx_rec_jdet_mirror_ratio(&jdet);
            plx_rec_conf_from_jdet(&jdet, to->jpow, conf);
            plx_img_free(&jdet);
        }
    } else {
        plx_rec_texture_sim(&im, sim, tex, conf);
    }
    if (to->vpow > 0.0f) {
        if (!ds->pose) plx_die("--vis needs --pose (and --depth)");
        const float *ps = ds->pose + (size_t)i * POSE_COLS;
        if (!((int)ps[POSE_FLAGS] & POSE_UNTRUSTED))
            plx_rec_visibility(&ds->depth, ps[POSE_YAW], to->vyaw ? 0.0 : ps[POSE_PITCH], to->vpow, !to->vabs, conf);
    }
    if (ds->conf_path) {
        const float *c = (const float *)ds->conf.data + (size_t)i * PLX_REC_TEX_W * PLX_REC_TEX_H;
        for (size_t k = 0; k < (size_t)PLX_REC_TEX_W * PLX_REC_TEX_H; k++) conf->px[k] *= PLX_CLAMP(c[k], 0.0f, 1.0f);
    }
    plx_img_free(&im);
}

/* ---- descriptors ------------------------------------------------------------------------- */

static plx_desc_layout layout_of(const dataset *ds, const plx_desc_params *p) {
    plx_desc_layout L = plx_desc_layout_of(p);
    if (ds->codes) plx_desc_layout_append_code(&L, ds->code_dim);
    return L;
}

typedef struct {
    const dataset *ds;
    const plx_desc_params *p;
    const tex_opts *to;
    plx_desc_layout lay;
    float *x, *cw;
} desc_job;

static void desc_one(int64_t i, int tid, void *ctx) {
    desc_job *j = ctx;
    plx_img tex = plx_img_new(PLX_REC_TEX_W, PLX_REC_TEX_H, 1), conf = plx_img_new(PLX_REC_TEX_W, PLX_REC_TEX_H, 1);
    texture_of(j->ds, j->to, (int)i, &tex, &conf);
    float *x = j->x + (size_t)i * j->lay.d, *cw = j->cw + (size_t)i * j->lay.n_cells;
    plx_desc_compute(j->p, &tex, &conf, x, cw);
    if (j->ds->codes) { /* the code block sits last, with full confidence */
        int cb = j->lay.n_blocks - 1;
        memcpy(x + j->lay.off[cb], j->ds->codes + (size_t)i * j->ds->code_dim, sizeof(float) * j->ds->code_dim);
        cw[j->lay.cell_off[cb]] = 1.0f;
    }
    plx_img_free(&tex);
    plx_img_free(&conf);
}

/* desc.cfg holds the descriptor description and the texture source; equal text = reusable cache */
static void cache_key(const dataset *ds, const plx_desc_params *p, const tex_opts *to, char *buf, size_t cap) {
    char d[1024];
    plx_desc_describe(p, d, sizeof d);
    int at = snprintf(buf, cap, "%s sim=%s", d, ds->sim_path ? ds->sim_path : "identity");
    if (ds->flow_dir && at > 0 && (size_t)at < cap)
        at += snprintf(buf + at, cap - (size_t)at, " flow=%s alpha=%g jconf=%g%s fallback=%d", ds->flow_dir, to->alpha,
                       to->jpow, to->jmirror ? "(mirror ratio)" : "", to->fallback);
    if (to->vpow > 0.0f && at > 0 && (size_t)at < cap)
        at += snprintf(buf + at, cap - (size_t)at, " vis=%g%s%s pose=%s depth=%s", to->vpow, to->vabs ? "(abs)" : "",
                       to->vyaw ? "(yaw only)" : "", ds->pose_path, ds->depth_path);
    if (ds->conf_path && at > 0 && (size_t)at < cap) at += snprintf(buf + at, cap - (size_t)at, " conf=%s", ds->conf_path);
    if (ds->codes_path && at > 0 && (size_t)at < cap) at += snprintf(buf + at, cap - (size_t)at, " codes=%s", ds->codes_path);
    if (at > 0 && (size_t)at < cap) snprintf(buf + at, cap - (size_t)at, "\n");
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

static void descriptors(const dataset *ds, const plx_desc_params *p, const tex_opts *to, const char *out,
                        bool use_cache, float **x, float **cw) {
    plx_desc_layout lay = layout_of(ds, p);
    int n = ds->db.n_images;
    char key[2048];
    cache_key(ds, p, to, key, sizeof key);
    if (use_cache && cache_load(out, key, &lay, n, x, cw)) {
        plx_log("descriptor cache hit: %d x %d", n, lay.d);
        return;
    }
    desc_job j = {.ds = ds, .p = p, .to = to, .lay = lay, .x = plx_malloc(sizeof(float) * (size_t)n * lay.d),
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

static void texture_montage(const dataset *ds, const plx_desc_params *p, const tex_opts *to, const char *out) {
    enum { NIMG = 12, PER = 3 };
    plx_img tiles[PER * NIMG];
    char lab[PER * NIMG][32];
    const char *labels[PER * NIMG];
    for (int t = 0; t < NIMG; t++) {
        /* spread over the dev-test pairs: a mix of people, lighting and pose */
        const plx_pair *pr = &ds->dev_test.p[(size_t)t * ds->dev_test.n / NIMG];
        int img = t % 2 ? pr->b : pr->a;
        plx_img tex = plx_img_new(PLX_REC_TEX_W, PLX_REC_TEX_H, 1), conf = plx_img_new(PLX_REC_TEX_W, PLX_REC_TEX_H, 1);
        texture_of(ds, to, img, &tex, &conf);
        plx_img norm = plx_img_clone(&tex);
        plx_photo_normalize(&norm, &conf, &p->photo);
        tiles[PER * t] = tex;
        tiles[PER * t + 1] = display_norm(&norm);
        tiles[PER * t + 2] = conf; /* [0, 1] already: black = no evidence */
        plx_img_free(&norm);
        snprintf(lab[PER * t], sizeof lab[0], "#%d raw", img);
        snprintf(lab[PER * t + 1], sizeof lab[0], "normalised");
        snprintf(lab[PER * t + 2], sizeof lab[0], "confidence");
        for (int k = 0; k < PER; k++) labels[PER * t + k] = lab[PER * t + k];
    }
    plx_montage_opts o = plx_montage_defaults();
    o.cols = 3 * PER;
    plx_img m = plx_viz_montage(tiles, PER * NIMG, labels, &o);
    plx_img big = plx_viz_upscale(&m, 2, PLX_UP_NEAREST);
    save_png(out, "textures.png", &big);
    plx_img_free(&m);
    plx_img_free(&big);
    for (int t = 0; t < PER * NIMG; t++) plx_img_free(&tiles[t]);
}

/* the two canonical textures side by side, with the descriptor region outlined */
static plx_img pair_tile(const dataset *ds, const plx_desc_params *p, const tex_opts *to, int a, int b) {
    plx_img tile = plx_img_new(2 * PLX_REC_TEX_W + 4, PLX_REC_TEX_H, 3);
    plx_img tex = plx_img_new(PLX_REC_TEX_W, PLX_REC_TEX_H, 1), conf = plx_img_new(PLX_REC_TEX_W, PLX_REC_TEX_H, 1);
    const float red[3] = {0.9f, 0.3f, 0.2f};
    for (int s = 0; s < 2; s++) {
        texture_of(ds, to, s ? b : a, &tex, &conf);
        int ox = s * (PLX_REC_TEX_W + 4);
        for (int k = 0; k < 3; k++)
            for (int y = 0; y < PLX_REC_TEX_H; y++)
                for (int x = 0; x < PLX_REC_TEX_W; x++) PLX_PX(&tile, ox + x, y, k) = PLX_PX(&tex, x, y, 0);
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
static void failure_montage(const dataset *ds, const plx_desc_params *p, const tex_opts *to, const plx_pair *pr, int n,
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
            tiles[nt] = pair_tile(ds, p, to, q->a, q->b);
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
    const tex_opts *to;
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
        failure_montage(ev->ds, ev->dp, ev->to, p, n, score, thr, ev->out, name);
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

/* ---- ablation ----------------------------------------------------------------------------
 * View 1 only (View 2 is never touched here). Each row's flags are appended to the command
 * line's (later flags win), so a table runs on top of whatever configuration the command line
 * sets; rows are cumulative where the label says "+".
 *   ladder  the ingredient table of rungs L0 / L1 (descriptor and matcher ingredients)
 *   flow    the L2 experiments: partial warps, the stretch confidence plane and the ways of
 *           using it, and flow-quality fallback (needs --flow) */
typedef struct { const char *label, *flags; } ablation_row;

static const ablation_row LADDER[] = {
    {"LBP(8,1) 8px cells, raw hist, chi-square", "--blocks lbp1:8 --nophoto --nosqrt --nomirror --score chi2"},
    {"+ photometric normalisation", "--blocks lbp1:8 --nosqrt --nomirror --score chi2"},
    {"+ Hellinger (sqrt), raw cosine", "--blocks lbp1:8 --nomirror --score cos"},
    {"+ PCA-whitening, cosine", "--blocks lbp1:8 --nomirror"},
    {"  same, PCA without whitening", "--blocks lbp1:8 --nomirror --nowhiten"},
    {"  same, no photometric normalisation", "--blocks lbp1:8 --nomirror --nophoto"},
    {"  same, no sqrt", "--blocks lbp1:8 --nomirror --nosqrt"},
    {"LBP(8,1) 12px cells alone", "--blocks lbp1:12 --nomirror"},
    {"LBP(8,2) 12px cells alone", "--blocks lbp2:12 --nomirror"},
    {"LBP(8,2) half resolution alone", "--blocks lbp2@1:8 --nomirror"},
    {"orientation histograms alone", "--blocks ori:12 --nomirror"},
    {"all four blocks", "--nomirror"},
    {"+ mirror pooling (default, unsupervised)", ""},
    {"  same, raw cell-weighted Bhattacharyya", "--score cell"},
    {"  same, one global PCA instead of per block", "--global-pca"},
    {"  same, no photometric normalisation", "--nophoto"},
    {"  same, PCA dims / 2", "--pca 125"},
    {"  same, PCA dims x 2", "--pca 500"},
    {"default + pair-LDA (restricted)", "--sup lda"},
    {"default + joint Bayes (restricted, default)", "--sup jb"},
    {"default + logistic pair model (restricted)", "--sup logit"},
    {"joint Bayes without mirror pooling", "--nomirror --sup jb"},
};

/* flow quality flag bits (align/flow.h, plx_flow): 1 untrusted by congealing, 2 fold,
 * 4 last round not converged, 8 valid < 0.9, 16 step > 1 px */
static const ablation_row FLOW[] = {
    {"alpha 0 (flow stage's similarity only)", "--sup jb --alpha 0"},
    {"alpha 0.25", "--sup jb --alpha 0.25"},
    {"alpha 0.5", "--sup jb --alpha 0.5"},
    {"alpha 0.75", "--sup jb --alpha 0.75"},
    {"alpha 1 (full frontalisation)", "--sup jb --alpha 1"},
    {"+ stretch confidence p=1, impute toward mean", "--sup jb --alpha 1 --jconf 1"},
    {"  same, p=2", "--sup jb --alpha 1 --jconf 2"},
    {"  same, p=0.5", "--sup jb --alpha 1 --jconf 0.5"},
    {"+ stretch p=1, weighted mirror pooling", "--sup jb --alpha 1 --jconf 1 --mirror-weighted"},
    {"  same, p=2", "--sup jb --alpha 1 --jconf 2 --mirror-weighted"},
    {"  same, p=4", "--sup jb --alpha 1 --jconf 4 --mirror-weighted"},
    {"+ mirror-ratio stretch p=2, weighted pooling", "--sup jb --alpha 1 --jconf 2 --jconf-mirror --mirror-weighted"},
    {"alpha 1, quality fallback fold|valid|step", "--sup jb --alpha 1 --fallback 26"},
    {"alpha 1, fallback also untrusted-by-congeal", "--sup jb --alpha 1 --fallback 27"},
};

static void ablation(const dataset *ds, const plx_args *cmd, const char *table, const char *out) {
    const ablation_row *rows = LADDER;
    size_t nrows = sizeof LADDER / sizeof LADDER[0];
    if (!strcmp(table, "flow")) {
        if (!ds->u) plx_die("--ablation flow needs --flow");
        rows = FLOW;
        nrows = sizeof FLOW / sizeof FLOW[0];
    } else if (strcmp(table, "ladder") != 0) {
        plx_die("--ablation: unknown table '%s' (ladder | flow)", table);
    }
    char path[4096];
    snprintf(path, sizeof path, "%s/ablation_%s.txt", out, table);
    FILE *f = fopen(path, "w");
    if (!f) plx_die("cannot write %s", path);
    fprintf(f, "# View 1 ablation '%s' (train pairsDevTrain, test pairsDevTest), texture: %s%s%s\n# command line:", table,
            ds->sim_path ? ds->sim_path : "identity (L0)", ds->flow_dir ? " + flow " : "",
            ds->flow_dir ? ds->flow_dir : "");
    for (int k = 1; k < cmd->argc; k++) fprintf(f, " %s", cmd->argv[k]);
    fprintf(f, "\n# %-50s %-17s %-17s  flags\n", "configuration", "unsup acc / AUC", "restr. acc / AUC");
    float *x = NULL, *cw = NULL;
    char have[2048] = "";
    for (size_t r = 0; r < nrows; r++) {
        enum { MAXARG = 128 };
        char buf[512], *argv[MAXARG];
        int argc = 0;
        for (int k = 0; k < cmd->argc && argc < MAXARG; k++) argv[argc++] = cmd->argv[k];
        snprintf(buf, sizeof buf, "%s", rows[r].flags);
        for (char *t = strtok(buf, " "); t && argc < MAXARG; t = strtok(NULL, " ")) argv[argc++] = t;
        plx_args a = {argc, argv};
        plx_desc_params dp = desc_from_args(&a);
        plx_match_cfg mc = match_from_args(&a);
        tex_opts to = tex_from_args(&a);
        bool sup = plx_arg_str(&a, "sup", NULL) != NULL;
        if (!sup) mc.sup = PLX_SUP_NONE;
        char key[2048];
        cache_key(ds, &dp, &to, key, sizeof key);
        if (strcmp(key, have) != 0) {
            free(x);
            free(cw);
            descriptors(ds, &dp, &to, out, false, &x, &cw);
            snprintf(have, sizeof have, "%s", key);
        }
        plx_desc_layout lay = layout_of(ds, &dp);
        plx_feats all = {.n = ds->db.n_images, .d = lay.d, .n_cells = lay.n_cells, .x = x, .cw = cw};
        double *score[2] = {PLX_NEW(double, ds->dev_test.n), sup ? PLX_NEW(double, ds->dev_test.n) : NULL};
        plx_verif_report rep[2];
        int rc = plx_rec_eval_view1(&mc, &lay, &all, &ds->dev_train, &ds->dev_test, &ds->db, score[0], score[1],
                                    &rep[0], &rep[1]);
        if (rc != PLX_OK) plx_die("ablation row %zu failed", r);
        fprintf(f, "%-52s %.4f / %.4f   ", rows[r].label, rep[0].mean_acc, rep[0].auc);
        if (sup) fprintf(f, "%.4f / %.4f   ", rep[1].mean_acc, rep[1].auc);
        else fprintf(f, "%-17s  ", "-");
        fprintf(f, "%s\n", rows[r].flags[0] ? rows[r].flags : "(command line only)");
        fflush(f);
        plx_log("ablation %zu: %s  unsup %.4f", r, rows[r].label, rep[0].mean_acc);
        free(score[0]);
        free(score[1]);
    }
    free(x);
    free(cw);
    fclose(f);
    plx_log("wrote %s", path);
}

int main(int argc, char **argv) {
    plx_args a = {argc, argv};
    const char *out = plx_arg_str(&a, "out", NULL);
    if (!out) plx_die("usage: plx_eval_tex --pack DIR [--sim FILE] --out DIR [flags]  (see the source header)");
    if (plx_mkdir_p(out) != PLX_OK) plx_die("cannot create %s", out);
    double t_start = plx_now();

    dataset ds = {0};
    dataset_open(&ds, &a);
    const char *table = plx_arg_str(&a, "ablation", NULL);
    if (table || plx_arg_flag(&a, "ablation")) { /* a mode of its own: leaves report.txt and the cache alone */
        ablation(&ds, &a, table ? table : "ladder", out);
        dataset_close(&ds);
        return 0;
    }
    plx_desc_params dp = desc_from_args(&a);
    plx_match_cfg mc = match_from_args(&a);
    tex_opts to = tex_from_args(&a);

    float *x = NULL, *cw = NULL;
    descriptors(&ds, &dp, &to, out, true, &x, &cw);
    plx_desc_layout lay = layout_of(&ds, &dp);

    char path[4096];
    snprintf(path, sizeof path, "%s/report.txt", out);
    FILE *rep = fopen(path, "w");
    if (!rep) plx_die("cannot write %s", path);
    char key[2048];
    cache_key(&ds, &dp, &to, key, sizeof key);
    fprintf(rep, "plx_eval_tex report\n# texture: %s\n# descriptor + texture source: %s# D = %d, %d cells\n\n",
            ds.flow_dir ? "dense flow (L2)" : ds.sim_path ? "similarity (L1)" : "identity similarity (raw centre crop, L0)",
            key, lay.d, lay.n_cells);

    evaluator ev = {.ds = &ds, .dp = &dp, .to = &to, .lay = lay, .all = {.n = ds.db.n_images, .d = lay.d, .n_cells = lay.n_cells,
                    .x = x, .cw = cw}, .out = out, .report = rep, .png = !plx_arg_flag(&a, "no-png")};
    if (ev.png) texture_montage(&ds, &dp, &to, out);

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
