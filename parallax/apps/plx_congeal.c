// plx-deps: align feat ops viz img core
/* plx_congeal: stage 1 of the pipeline. Jointly aligns every packed face crop to a template
 * learnt from the crops themselves (similarity congealing, align/congeal.h), then measures how
 * trustworthy the result is.
 *
 *   plx_congeal --pack runs/main/pack --out runs/main/congeal [options]
 *
 * Writes into <out>:
 *   sim.plxt            f32 [N, 4]  (a, b, tx, ty) per image, align/canon.h convention
 *   template_mean.plxt  f32 [K, 192, 160]  final template mean on Ω+ (margin 16, level 0)
 *   template_var.plxt   f32 [K, 192, 160]  population variance, same layout
 *   quality.plxt        f32 [N, 8]   columns (QUALITY_COLUMNS below):
 *                       0 cost      Σωe²/Σω vs the final template (≈ K for a typical face)
 *                       1 inlier    Huber weight mass in [0, 1]
 *                       2 valid     window fraction backed by real photo pixels
 *                       3 disp      px a fresh full-res solve would still move the face
 *                       4 converged last alignment round converged (0/1)
 *                       5 bound     a bound was hit in some round (0/1)
 *                       6 scale     |T| (s)
 *                       7 rot_deg   roll of T in degrees
 *   mean_face.plxt      f32 [160, 128] mean canonical gray face at the final T
 *   diag/ (PNG diagnostics and summary.txt)
 *
 * Options (defaults reproduce the reference run): --schedule "L:blur:iters:prior,..."
 * --eps-frac --huber --ramp --tol --k --unsigned --sigma-d --sigma-n --sigma-s --bicubic
 * --n (first n images only, for experiments) --eval-n (mirror / perturbation sample size)
 * --seed --skip-eval. */
#include "align/canon.h"
#include "align/congeal.h"
#include "core/core.h"
#include "feat/feat.h"
#include "img/image.h"
#include "img/imgio.h"
#include "ops/ops.h"
#include "viz/viz.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_SCHEDULE "2:1:10:0.1,2:0.5:10:0,2:0:10:0,2:0:10:0,1:0:10:0,1:0:10:0,1:0:10:0,0:0:10:0,0:0:10:0,0:0:10:0"
enum { QCOLS = 8, MAX_STAGES = 64 };
static const char *const QUALITY_COLUMNS[QCOLS] = {"cost", "inlier", "valid", "disp", "converged", "bound", "scale",
                                                    "rot_deg"};
static const float GUIDE[3] = {1.0f, 0.35f, 0.2f};

static int parse_schedule(const char *s, plx_cg_stage *st) {
    int n = 0;
    while (*s) {
        if (n == MAX_STAGES) plx_die("schedule: more than %d stages", MAX_STAGES);
        int used = 0;
        if (sscanf(s, "%d:%f:%d:%f%n", &st[n].level, &st[n].blur, &st[n].iters, &st[n].prior, &used) != 4)
            plx_die("schedule: cannot parse '%s' (want level:blur:iters:prior,...)", s);
        n++;
        s += used;
        if (*s == ',') s++;
    }
    return n;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* q-quantile of v[0..n) (sorts v in place). */
static double quantile(double *v, int n, double q) {
    qsort(v, n, sizeof(double), cmp_double);
    double pos = q * (n - 1);
    int i = (int)pos;
    return i + 1 < n ? v[i] + (pos - i) * (v[i + 1] - v[i]) : v[n - 1];
}

static void describe(FILE *f, const char *name, const double *v0, int n) {
    double *v = PLX_NEW(double, n), m = 0, s = 0;
    memcpy(v, v0, sizeof(double) * n);
    for (int i = 0; i < n; i++) m += v[i] / n;
    for (int i = 0; i < n; i++) s += (v[i] - m) * (v[i] - m) / n;
    double p[7] = {0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99};
    fprintf(f, "  %-10s mean %8.4f  std %7.4f |", name, m, sqrt(s));
    for (int j = 0; j < 7; j++) fprintf(f, " p%02d %8.4f", (int)lrint(100 * p[j]), quantile(v, n, p[j]));
    fprintf(f, "\n");
    free(v);
}

/* Mean squared gradient of a mean gray face (Ω) over the face core, x 1e4. */
static double sharpness(const plx_img *face) {
    plx_img gx = plx_img_new(face->w, face->h, 1), gy = plx_img_new(face->w, face->h, 1);
    plx_gradient_scharr(face, &gx, &gy);
    double s = 0;
    int c = 0;
    for (int y = 21; y <= 136; y++)
        for (int x = 24; x <= 104; x++, c++) s += PLX_PX(&gx, x, y, 0) * PLX_PX(&gx, x, y, 0) + PLX_PX(&gy, x, y, 0) * PLX_PX(&gy, x, y, 0);
    plx_img_free(&gx);
    plx_img_free(&gy);
    return 1e4 * s / c;
}

/* ---- discrepancy between two alignments of the same photo ------------------------------ */
typedef struct { double t, s, r; } discrepancy; /* px at c0, % scale, degrees */
static discrepancy compare(plx_sim A, plx_sim B) {
    double dr = plx_sim_angle(A) - plx_sim_angle(B);
    dr = atan2(sin(dr), cos(dr));
    return (discrepancy){hypot(A.tx - B.tx, A.ty - B.ty), 100 * fabs(plx_sim_scale(A) / plx_sim_scale(B) - 1),
                         fabs(dr) * 180 / M_PI};
}

static void report_discrepancy(FILE *f, const char *title, const discrepancy *d, int n, bool fractions) {
    double *t = PLX_NEW(double, n), *s = PLX_NEW(double, n), *r = PLX_NEW(double, n);
    int ok_t = 0, ok_s = 0, ok_r = 0, ok_all = 0;
    for (int i = 0; i < n; i++) {
        t[i] = d[i].t;
        s[i] = d[i].s;
        r[i] = d[i].r;
        ok_t += d[i].t <= 1;
        ok_s += d[i].s <= 1;
        ok_r += d[i].r <= 1;
        ok_all += d[i].t <= 1 && d[i].s <= 1 && d[i].r <= 1;
    }
    char line[512];
    snprintf(line, sizeof line,
             "%s (n=%d)\n  translation px: median %.3f  p95 %.3f\n  scale %%:       median %.3f  p95 %.3f\n"
             "  rotation deg:   median %.3f  p95 %.3f\n",
             title, n, quantile(t, n, 0.5), quantile(t, n, 0.95), quantile(s, n, 0.5), quantile(s, n, 0.95),
             quantile(r, n, 0.5), quantile(r, n, 0.95));
    fputs(line, f);
    fputs(line, stderr);
    if (fractions) {
        snprintf(line, sizeof line,
                 "  within 1 px: %.1f%%  within 1%%: %.1f%%  within 1 deg: %.1f%%  all three: %.1f%%\n",
                 100.0 * ok_t / n, 100.0 * ok_s / n, 100.0 * ok_r / n, 100.0 * ok_all / n);
        fputs(line, f);
        fputs(line, stderr);
    }
    free(t);
    free(s);
    free(r);
}

/* ---- evaluation: mirror consistency and perturbation recovery -------------------------- */
typedef struct {
    const plx_cg_source *src;
    const plx_cg_params *p;
    const plx_cg_stage  *st;
    const plx_cg_ref    *refs;
    int                  ns;
    const plx_sim       *T;
    const int           *idx;
    const plx_sim       *perturb;    /* NULL: mirror test */
    discrepancy         *d_flip;     /* flipped from identity vs twin of the population result */
    discrepancy         *d_orig;     /* original from identity vs the population result */
    discrepancy         *d_equi;     /* flipped vs twin of original, both from identity */
    discrepancy         *d_pert;     /* perturbed start vs the population result */
} eval_ctx;

static void eval_one(int64_t j, int tid, void *pc) {
    eval_ctx *c = pc;
    int i = c->idx[j];
    plx_img g = plx_cg_source_image(c->src, i);
    if (c->perturb) {
        plx_sim T0 = plx_sim_compose(c->T[i], c->perturb[j]);
        plx_sim T = plx_cg_align_image(&g, T0, c->refs, c->st, c->ns, c->p, NULL, NULL);
        c->d_pert[j] = compare(T, c->T[i]);
    } else {
        plx_img gf = plx_img_new(g.w, g.h, 1);
        plx_flip_h(&g, &gf);
        plx_sim Tf = plx_cg_align_image(&gf, plx_sim_identity(), c->refs, c->st, c->ns, c->p, NULL, NULL);
        plx_sim To = plx_cg_align_image(&g, plx_sim_identity(), c->refs, c->st, c->ns, c->p, NULL, NULL);
        c->d_flip[j] = compare(Tf, plx_sim_twin(c->T[i]));
        c->d_orig[j] = compare(To, c->T[i]);
        c->d_equi[j] = compare(Tf, plx_sim_twin(To));
        plx_img_free(&gf);
    }
    plx_img_free(&g);
}

/* ---- diagnostics ----------------------------------------------------------------------- */
static void save_png(const char *dir, const char *name, const plx_img *im) {
    char path[4096];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    if (plx_img_save(path, im) != PLX_OK) plx_die("cannot write %s", path);
    plx_log("wrote %s", path);
}

/* Canonical crop of photo i at T as RGB, with guide lines at the DESIGN eye / mouth rows. */
static plx_img canon_tile(const plx_cg_source *src, int i, plx_sim T, bool guides) {
    plx_img g = plx_cg_source_image(src, i), c = plx_canon_new(1, 0);
    plx_canon_warp(&g, T, 0, PLX_INTERP_BILINEAR, &c, NULL);
    plx_img rgb = plx_viz_to_rgb(&c);
    if (guides) {
        plx_draw_line(&rgb, 0, 63, PLX_CANON_W - 1, 63, 1, GUIDE, 0.35f);
        plx_draw_line(&rgb, 0, 103, PLX_CANON_W - 1, 103, 1, GUIDE, 0.35f);
        plx_draw_line(&rgb, 63.5f, 0, 63.5f, PLX_CANON_H - 1, 1, GUIDE, 0.2f);
    }
    plx_img_free(&g);
    plx_img_free(&c);
    return rgb;
}

static void free_tiles(plx_img *t, int n) {
    for (int i = 0; i < n; i++) plx_img_free(&t[i]);
}

static void diag_rounds(const char *dir, const plx_cg_round *rounds, int nr) {
    plx_img *tiles = PLX_NEW(plx_img, nr);
    char **lab = PLX_NEW(char *, nr);
    for (int r = 0; r < nr; r++) {
        plx_img up = plx_viz_upscale(&rounds[r].mean_gray, 2, PLX_UP_BILINEAR);
        float lo = 1, hi = 0; /* stretch each mean for visibility; same map for every round */
        for (size_t k = 0; k < (size_t)up.w * up.h; k++) lo = fminf(lo, up.px[k]), hi = fmaxf(hi, up.px[k]);
        plx_img_scale_offset(&up, 1.0f / fmaxf(hi - lo, 1e-6f), -lo / fmaxf(hi - lo, 1e-6f));
        tiles[r] = plx_viz_to_rgb(&up);
        plx_img_free(&up);
        lab[r] = plx_malloc(32);
        snprintf(lab[r], 32, r == nr - 1 ? "final" : "round %d", r);
    }
    plx_montage_opts o = plx_montage_defaults();
    o.cols = nr <= 6 ? nr : (nr + 1) / 2;
    o.text_scale = 2;
    plx_img m = plx_viz_montage(tiles, nr, (const char *const *)lab, &o);
    save_png(dir, "rounds_mean_face.png", &m);
    plx_img_free(&m);
    free_tiles(tiles, nr);
    for (int r = 0; r < nr; r++) free(lab[r]);
    free(tiles);
    free(lab);
}

static void diag_before_after(const char *dir, const plx_cg_source *src, const plx_sim *T, int n, plx_rng *rng) {
    enum { PAIRS = 60 };
    int m = PLX_MIN(PAIRS, n);
    plx_img tiles[2 * PAIRS];
    int *perm = PLX_NEW(int, n);
    for (int i = 0; i < n; i++) perm[i] = i;
    plx_rng_shuffle(rng, perm, n);
    for (int j = 0; j < m; j++) {
        tiles[2 * j] = canon_tile(src, perm[j], plx_sim_identity(), true);
        tiles[2 * j + 1] = canon_tile(src, perm[j], T[perm[j]], true);
    }
    plx_montage_opts o = plx_montage_defaults();
    o.cols = 12;
    o.pad = 3;
    plx_img im = plx_viz_montage(tiles, 2 * m, NULL, &o);
    save_png(dir, "before_after.png", &im);
    plx_img_free(&im);
    free_tiles(tiles, 2 * m);
    free(perm);
}

typedef struct { float key; int idx; } keyed;
static int cmp_keyed_desc(const void *a, const void *b) {
    float x = ((const keyed *)a)->key, y = ((const keyed *)b)->key;
    return (x < y) - (x > y);
}

static void diag_worst(const char *dir, const plx_cg_source *src, const plx_sim *T, const float *q, int n) {
    enum { WORST = 40 };
    int m = PLX_MIN(WORST, n);
    keyed *order = PLX_NEW(keyed, n);
    for (int i = 0; i < n; i++) order[i] = (keyed){q[(size_t)i * QCOLS], i};
    qsort(order, n, sizeof(keyed), cmp_keyed_desc);
    plx_img tiles[2 * WORST];
    char *lab[2 * WORST];
    for (int j = 0; j < m; j++) {
        int i = order[j].idx;
        tiles[2 * j] = canon_tile(src, i, plx_sim_identity(), false);
        tiles[2 * j + 1] = canon_tile(src, i, T[i], true);
        lab[2 * j] = plx_malloc(32);
        lab[2 * j + 1] = plx_malloc(32);
        snprintf(lab[2 * j], 32, "#%d raw", i);
        snprintf(lab[2 * j + 1], 32, "c%.1f%s%s", q[i * QCOLS + 0], q[i * QCOLS + 5] > 0 ? " B" : "",
                 q[i * QCOLS + 4] > 0 ? "" : " U");
    }
    plx_montage_opts o = plx_montage_defaults();
    o.cols = 10;
    o.pad = 3;
    plx_img im = plx_viz_montage(tiles, 2 * m, (const char *const *)lab, &o);
    save_png(dir, "worst.png", &im);
    plx_img_free(&im);
    free_tiles(tiles, 2 * m);
    for (int j = 0; j < 2 * m; j++) free(lab[j]);
    free(order);
}

/* Template channels (shared colour range), their variances, total variance, the solver weight
 * Σ_k 1/(σ²+ε) at full resolution, the mean face and its mirror asymmetry. */
static void diag_template(const char *dir, const plx_cg_template *t, const plx_cg_params *p, const plx_img *mean_face) {
    int K = t->K;
    plx_img mu = plx_canon_crop(&t->mean[0], t->margin), var = plx_canon_crop(&t->var[0], t->margin);
    size_t plane = (size_t)mu.w * mu.h;
    float lo, hi, vlo, vhi;
    plx_viz_auto_range(mu.px, plane * K, 0.0f, 0.995f, &lo, &hi);
    plx_viz_auto_range(var.px, plane * K, 0.0f, 0.995f, &vlo, &vhi);
    int nt = 2 * K + 5;
    plx_img *tiles = PLX_NEW(plx_img, nt);
    char **lab = PLX_NEW(char *, nt);
    for (int k = 0; k < K; k++) {
        float ang = plx_orient_bin_center(&p->feat, k) * 180.0f / (float)M_PI;
        tiles[k] = plx_viz_colormap(plx_img_chan(&mu, k), mu.w, mu.h, lo, hi, PLX_CMAP_SEQ);
        tiles[K + k] = plx_viz_colormap(plx_img_chan(&var, k), mu.w, mu.h, vlo, vhi, PLX_CMAP_SEQ);
        lab[k] = plx_malloc(32);
        lab[K + k] = plx_malloc(32);
        snprintf(lab[k], 32, "mean %d (%.0fd)", k, ang);
        snprintf(lab[K + k], 32, "var %d", k);
    }
    plx_cg_ref ref = plx_cg_ref_build(t, p, 0, 0);
    plx_img tot = plx_img_new(mu.w, mu.h, 1), wt = plx_img_new(mu.w, mu.h, 1);
    for (int y = 0; y < mu.h; y++)
        for (int x = 0; x < mu.w; x++)
            for (int k = 0; k < K; k++) {
                float v = PLX_PX(&var, x, y, k);
                PLX_PX(&tot, x, y, 0) += v;
                PLX_PX(&wt, x, y, 0) += (float)(1.0 / (v + ref.eps));
            }
    plx_cg_ref_free(&ref);
    tiles[2 * K] = plx_viz_scalar(&tot, PLX_CMAP_SEQ);
    tiles[2 * K + 1] = plx_viz_scalar(&wt, PLX_CMAP_SEQ);
    plx_img mf = plx_viz_to_rgb(mean_face), fl = plx_img_new(mean_face->w, mean_face->h, 1);
    plx_flip_h(mean_face, &fl);
    for (size_t i = 0; i < (size_t)fl.w * fl.h; i++) fl.px[i] = mean_face->px[i] - fl.px[i];
    float m = 0;
    for (size_t i = 0; i < (size_t)fl.w * fl.h; i++) m = fmaxf(m, fabsf(fl.px[i]));
    tiles[2 * K + 2] = mf;
    tiles[2 * K + 3] = plx_viz_colormap(fl.px, fl.w, fl.h, -m, m, PLX_CMAP_DIV);
    plx_img sm = plx_img_new(mu.w, mu.h, 1);
    for (int k = 0; k < K; k++)
        for (size_t i = 0; i < plane; i++) sm.px[i] += mu.px[k * plane + i];
    tiles[2 * K + 4] = plx_viz_scalar(&sm, PLX_CMAP_GRAY);
    const char *extra[5] = {"total var", "weight", "mean face", "face - mirror", "sum of means"};
    for (int j = 0; j < 5; j++) lab[2 * K + j] = plx_strdup(extra[j]);
    for (int j = 0; j < nt; j++) {
        plx_img up = plx_viz_upscale(&tiles[j], 2, PLX_UP_NEAREST);
        plx_img_free(&tiles[j]);
        tiles[j] = up;
    }
    plx_montage_opts o = plx_montage_defaults();
    o.cols = K;
    o.text_scale = 2;
    plx_img im = plx_viz_montage(tiles, nt, (const char *const *)lab, &o);
    save_png(dir, "template.png", &im);
    plx_img_free(&im);
    plx_img big = plx_viz_upscale(mean_face, 3, PLX_UP_BILINEAR);
    save_png(dir, "mean_face.png", &big);
    plx_img_free(&big);
    free_tiles(tiles, nt);
    for (int j = 0; j < nt; j++) free(lab[j]);
    free(tiles);
    free(lab);
    plx_img_free(&mu);
    plx_img_free(&var);
    plx_img_free(&tot);
    plx_img_free(&wt);
    plx_img_free(&fl);
    plx_img_free(&sm);
}

int main(int argc, char **argv) {
    plx_args a = {argc, argv};
    const char *pack = plx_arg_str(&a, "pack", "runs/main/pack");
    const char *out = plx_arg_str(&a, "out", "runs/main/congeal");
    plx_cg_params p = plx_cg_default();
    p.eps_frac = (float)plx_arg_num(&a, "eps-frac", p.eps_frac);
    p.huber = (float)plx_arg_num(&a, "huber", p.huber);
    p.ramp = (float)plx_arg_num(&a, "ramp", p.ramp);
    p.tol = (float)plx_arg_num(&a, "tol", p.tol);
    p.feat.k = (int)plx_arg_int(&a, "k", p.feat.k);
    p.feat.signed_ = !plx_arg_flag(&a, "unsigned");
    p.feat.sigma_d = (float)plx_arg_num(&a, "sigma-d", p.feat.sigma_d);
    p.feat.sigma_n = (float)plx_arg_num(&a, "sigma-n", p.feat.sigma_n);
    p.feat.sigma_s = (float)plx_arg_num(&a, "sigma-s", p.feat.sigma_s);
    if (plx_arg_flag(&a, "bicubic")) p.interp = PLX_INTERP_BICUBIC;
    plx_cg_stage sched[MAX_STAGES];
    const char *sched_str = plx_arg_str(&a, "schedule", DEFAULT_SCHEDULE);
    int ns = parse_schedule(sched_str, sched);
    for (int s = 0; s < ns; s++)
        if (sched[s].level < 0 || sched[s].level >= p.levels || sched[s].iters < 1) plx_die("schedule: bad stage %d", s);
    int eval_n = (int)plx_arg_int(&a, "eval-n", 500);
    uint64_t seed = (uint64_t)plx_arg_int(&a, "seed", 1);
    bool skip_eval = plx_arg_flag(&a, "skip-eval");

    char path[8192], dir[4096];
    snprintf(path, sizeof path, "%s/gray.plxt", pack);
    plx_tensor gt;
    if (plx_tensor_map(path, &gt) != PLX_OK) plx_die("cannot map %s", path);
    if (gt.dtype != PLX_U8 || gt.ndim != 3 || gt.dim[1] != PLX_CANON_IMG_SIDE || gt.dim[2] != PLX_CANON_IMG_SIDE)
        plx_die("%s: want u8 [N, 250, 250]", path);
    int n = (int)PLX_MIN(gt.dim[0], plx_arg_int(&a, "n", gt.dim[0]));
    plx_cg_source src = {gt.data, n, (int)gt.dim[2], (int)gt.dim[1]};
    snprintf(dir, sizeof dir, "%s/diag", out);
    if (plx_mkdir_p(dir) != PLX_OK) plx_die("cannot create %s", dir);
    plx_log("congealing %d images, %d threads, schedule %s", n, plx_num_threads(), sched_str);

    /* ---- the stage itself ---- */
    double t_start = plx_now();
    plx_sim *T = PLX_NEW(plx_sim, n);
    for (int i = 0; i < n; i++) T[i] = plx_sim_identity();
    plx_cg_round *rounds = PLX_NEW(plx_cg_round, ns + 2);
    plx_cg_quality *qual = PLX_NEW(plx_cg_quality, n);
    plx_cg_template tmpl;
    plx_cg_congeal(&src, &p, sched, ns, T, &tmpl, rounds, qual);
    double t_congeal = plx_now() - t_start;

    /* ---- outputs ---- */
    float *sim = PLX_NEW(float, (size_t)n * 4), *q = PLX_NEW(float, (size_t)n * QCOLS);
    for (int i = 0; i < n; i++) {
        float v[4] = {(float)T[i].a, (float)T[i].b, (float)T[i].tx, (float)T[i].ty};
        memcpy(sim + 4 * i, v, sizeof v);
        float r[QCOLS] = {qual[i].cost, qual[i].inlier, qual[i].valid, qual[i].disp, qual[i].converged,
                          qual[i].bound, (float)plx_sim_scale(T[i]), (float)(plx_sim_angle(T[i]) * 180 / M_PI)};
        memcpy(q + (size_t)QCOLS * i, r, sizeof r);
    }
    const plx_img *tm = &tmpl.mean[0], *tv = &tmpl.var[0];
    const plx_img *mean_face = &rounds[ns + 1].mean_gray;
    struct { const char *name; int ndim; int64_t dim[3]; const float *data; } outs[] = {
        {"sim.plxt", 2, {n, 4}, sim},
        {"quality.plxt", 2, {n, QCOLS}, q},
        {"template_mean.plxt", 3, {tm->c, tm->h, tm->w}, tm->px},
        {"template_var.plxt", 3, {tv->c, tv->h, tv->w}, tv->px},
        {"mean_face.plxt", 2, {mean_face->h, mean_face->w}, mean_face->px},
    };
    for (size_t j = 0; j < sizeof outs / sizeof outs[0]; j++) {
        snprintf(path, sizeof path, "%s/%s", out, outs[j].name);
        if (plx_tensor_save(path, PLX_F32, outs[j].ndim, outs[j].dim, outs[j].data) != PLX_OK)
            plx_die("cannot write %s", path);
        plx_log("wrote %s", path);
    }

    /* ---- summary: parameters, rounds, distributions ---- */
    snprintf(path, sizeof path, "%s/summary.txt", dir);
    FILE *f = fopen(path, "w");
    if (!f) plx_die("cannot write %s", path);
    fprintf(f, "plx_congeal summary: %d images, congealing %.1f s on %d threads\n\n", n, t_congeal, plx_num_threads());
    fprintf(f, "features: K=%d %s sigma_d=%.2f sigma_n=%.2f sigma_s=%.2f eps=%.4f; margin %d, %s warp\n", p.feat.k,
            p.feat.signed_ ? "signed" : "unsigned", p.feat.sigma_d, p.feat.sigma_n, p.feat.sigma_s, p.feat.eps,
            p.margin, p.interp == PLX_INTERP_BICUBIC ? "bicubic" : "bilinear");
    fprintf(f, "solver: eps_frac=%.3f huber=%.2f*sqrt(K) ramp=%.1f tol=%.3f; bounds scale [%.2f, %.2f], |rot| <= %.0f deg,"
               " |t| <= %.0f px\nschedule (level:blur:iters:prior): %s\n\n",
            p.eps_frac, p.huber, p.ramp, p.tol, p.scale_min, p.scale_max, p.rot_max * 180 / M_PI, p.shift_max, sched_str);
    fprintf(f, "rounds (template of round r = population at the start of round r):\n");
    fprintf(f, "  round level  blur   core_var  var/2nd-moment  sharpness  mean_cost  median_update_px  unconverged  bound"
               "   sec\n");
    for (int r = 0; r < ns + 2; r++) {
        const plx_cg_round *ri = &rounds[r];
        bool al = r >= 1 && r <= ns;
        fprintf(f, "  %5d %5d %5.2f %10.2f %15.4f %10.3f %10.3f %17.3f %12d %6d %5.1f\n", r,
                al ? sched[r - 1].level : -1, al ? sched[r - 1].blur : 0.0f, ri->core_var, ri->core_ratio,
                sharpness(&ri->mean_gray), ri->mean_cost, ri->median_disp, ri->n_unconverged, ri->n_bound, ri->seconds);
    }
    fprintf(f, "(sharpness = 1e4 * mean |grad|^2 of the mean gray face over the face core: an independent check, it\n"
               " does not use the features the alignment optimises)\n");
    fprintf(f, "variance drop, first/last core variance: %.3f  (var/2nd-moment %.4f -> %.4f); sharpness %.3f -> %.3f\n\n",
            rounds[0].core_var / rounds[ns + 1].core_var, rounds[0].core_ratio, rounds[ns + 1].core_ratio,
            sharpness(&rounds[0].mean_gray), sharpness(&rounds[ns + 1].mean_gray));
    double *col = PLX_NEW(double, n);
    fprintf(f, "parameter distributions:\n");
    for (int c = 0; c < 4; c++) {
        static const char *names[4] = {"scale", "rot_deg", "tx_px", "ty_px"};
        for (int i = 0; i < n; i++) {
            double v[4] = {plx_sim_scale(T[i]), plx_sim_angle(T[i]) * 180 / M_PI, T[i].tx, T[i].ty};
            col[i] = v[c];
        }
        describe(f, names[c], col, n);
    }
    fprintf(f, "quality distributions:\n");
    for (int c = 0; c < 4; c++) {
        for (int i = 0; i < n; i++) col[i] = q[(size_t)i * QCOLS + c];
        describe(f, QUALITY_COLUMNS[c], col, n);
    }
    int n_bound = 0, n_unconv = 0, n_disp1 = 0;
    for (int i = 0; i < n; i++) {
        n_bound += qual[i].bound;
        n_unconv += !qual[i].converged;
        n_disp1 += qual[i].disp > 1;
    }
    fprintf(f, "bound hit: %d images; unconverged in the last round: %d; would still move > 1 px: %d\n", n_bound,
            n_unconv, n_disp1);
    if (n_bound) {
        fprintf(f, "bound-hit images:");
        for (int i = 0, c = 0; i < n; i++)
            if (qual[i].bound) fprintf(f, "%s%d", c++ % 16 ? " " : "\n  ", i);
        fprintf(f, "\n");
    }
    fprintf(f, "\n");

    /* ---- acceptance metrics ---- */
    plx_rng rng;
    plx_rng_seed(&rng, seed);
    if (!skip_eval) {
        double t0 = plx_now();
        plx_cg_ref *refs = PLX_NEW(plx_cg_ref, ns);
        for (int s = 0; s < ns; s++) refs[s] = plx_cg_ref_build(&tmpl, &p, sched[s].level, sched[s].blur);
        int m = PLX_MIN(eval_n, n);
        int *perm = PLX_NEW(int, n), *idx = PLX_NEW(int, m);
        for (int i = 0; i < n; i++) perm[i] = i;
        plx_rng_shuffle(&rng, perm, n);
        memcpy(idx, perm, sizeof(int) * m);
        discrepancy *d1 = PLX_NEW(discrepancy, m), *d2 = PLX_NEW(discrepancy, m), *d3 = PLX_NEW(discrepancy, m);
        eval_ctx c = {.src = &src, .p = &p, .st = sched, .refs = refs, .ns = ns, .T = T, .idx = idx,
                      .d_flip = d1, .d_orig = d2, .d_equi = d3};
        plx_parallel_for(m, 1, eval_one, &c);
        fprintf(f, "MIRROR CONSISTENCY: flipped photo aligned from identity against the final template (full schedule)\n"
                   "vs the twin (a, -b, -tx, ty) of the photo's congealed T\n");
        report_discrepancy(f, "mirror: flipped-from-identity vs twin(congealed)", d1, m, true);
        fprintf(f, "path dependence (same single-image procedure on the unflipped photo vs its congealed T):\n");
        report_discrepancy(f, "original-from-identity vs congealed", d2, m, true);
        fprintf(f, "solver equivariance sanity check (both from identity; should be ~0 by construction):\n");
        report_discrepancy(f, "flipped vs twin(original)", d3, m, false);

        /* perturbation recovery on converged, unclipped images */
        int np = 0;
        for (int j = 0; j < n && np < m; j++) {
            int i = perm[n - 1 - j]; /* disjoint from the mirror sample where possible */
            if (qual[i].converged && !qual[i].bound) idx[np++] = i;
        }
        plx_sim *P = PLX_NEW(plx_sim, np);
        for (int j = 0; j < np; j++) {
            double s = 1 + 0.10 * (2 * plx_rng_uniform(&rng) - 1), rho = 10 * M_PI / 180 * (2 * plx_rng_uniform(&rng) - 1);
            P[j] = plx_sim_make(s, rho, 8 * (2 * plx_rng_uniform(&rng) - 1), 8 * (2 * plx_rng_uniform(&rng) - 1));
        }
        eval_ctx cp = {.src = &src, .p = &p, .st = sched, .refs = refs, .ns = ns, .T = T, .idx = idx, .perturb = P,
                       .d_pert = d1};
        plx_parallel_for(np, 1, eval_one, &cp);
        fprintf(f, "\nPERTURBATION RECOVERY: converged images, T o P with P ~ scale +-10%%, rotation +-10 deg, "
                   "shift +-8 px, re-aligned (full schedule) against the final template, vs the congealed T\n");
        report_discrepancy(f, "perturbed vs congealed", d1, np, true);
        fprintf(f, "\nevaluation time %.1f s\n", plx_now() - t0);
        for (int s = 0; s < ns; s++) plx_cg_ref_free(&refs[s]);
        free(refs);
        free(perm);
        free(idx);
        free(d1);
        free(d2);
        free(d3);
        free(P);
    }
    fclose(f);
    plx_log("wrote %s", path);

    /* ---- pictures ---- */
    diag_rounds(dir, rounds, ns + 2);
    diag_before_after(dir, &src, T, n, &rng);
    diag_worst(dir, &src, T, q, n);
    diag_template(dir, &tmpl, &p, mean_face);
    plx_log("total %.1f s", plx_now() - t_start);

    for (int r = 0; r < ns + 2; r++) plx_img_free(&rounds[r].mean_gray);
    plx_cg_template_free(&tmpl);
    free(rounds);
    free(qual);
    free(T);
    free(sim);
    free(q);
    free(col);
    plx_tensor_free(&gt);
    return 0;
}
