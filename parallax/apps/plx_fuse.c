// plx-deps: fuse align render viz feat ops eval data img la core
/* plx_fuse: multi-frame fusion (docs/DESIGN.md §8): many views of one person -> one shape.
 *
 *   plx_fuse --clip <clip.plxv | frame dir> --out <record dir>     a clip -> record + diag.png
 *   plx_fuse --synth [--out runs/main/fuse/synth] [--quick]         synthetic clips vs ground truth
 *   plx_fuse --lfw-identity [--out runs/main/fuse/lfw]              LFW identities as multi-view sets
 *   plx_fuse --locate-test [--out runs/main/fuse/locate]            localisation on pasted LFW faces
 *
 * Inputs (all modes): --flow runs/main/flow (frozen model under model/, and u, sim, quality,
 * node_weight for the LFW modes), --geom runs/main/geom (zbar.plxt: the population mean depth,
 * the sign / gauge / bas-relief prior; depth_basis.plxt: its per-node personal variance for the
 * posterior), --pack runs/main/pack (gray, rgb: textures and LFW photos), --lfw data/lfw.
 * Options: --step 8|4 (personal lattice), --lambda (personal bending weight), --rounds (personal
 * template rounds), --boot (bootstrap resamples), --threads.
 *
 * The record layout is documented in fuse/fuse.h. Every report line goes to stdout and to
 * <out>/summary.txt. */
#include "align/canon.h"
#include "align/flow.h"
#include "align/lattice.h"
#include "core/core.h"
#include "data/lfw.h"
#include "eval/verif.h"
#include "fuse/fuse.h"
#include "img/image.h"
#include "img/imgio.h"
#include "la/la.h"
#include "ops/ops.h"
#include "render/render.h"
#include "viz/viz.h"

#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#define DEG (M_PI / 180.0)
static const float WHITE[3] = {1, 1, 1}, GREY[3] = {0.6f, 0.6f, 0.6f}, RED[3] = {1, 0.3f, 0.2f},
                   GREEN[3] = {0.3f, 1, 0.4f}, CYAN[3] = {0.3f, 0.85f, 1}, YELLOW[3] = {1, 0.85f, 0.2f};

/* Region of the lattice that is tracked and factorised (a little larger than the face core,
 * so the occluding contour is in, hair and background mostly out), and the core ellipse that
 * carries the canonical gauge and every reported number (the population's core, geom.h). */
#define TRACK_AX 48.0
#define TRACK_AY 62.0
#define TRACK_YC 4.0
#define CORE_AX 40.0
#define CORE_AY 50.0
#define CORE_YC 6.0

/* ---- reporting ------------------------------------------------------------------------------------ */
static FILE *g_sum;
static void say(const char *fmt, ...) PLX_PRINTF(1, 2);
static void say(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    fflush(stdout);
    if (g_sum) {
        va_start(ap, fmt);
        vfprintf(g_sum, fmt, ap);
        va_end(ap);
        fflush(g_sum);
    }
}

static void open_summary(const char *dir) {
    char path[4096];
    if (plx_mkdir_p(dir) != PLX_OK) plx_die("cannot create %s", dir);
    snprintf(path, sizeof path, "%s/summary.txt", dir);
    g_sum = fopen(path, "w");
    if (!g_sum) plx_die("cannot write %s", path);
}

static void save_png(const char *dir, const char *name, const plx_img *im) {
    char path[4096];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    if (plx_img_save(path, im) != PLX_OK) plx_die("cannot write %s", path);
    plx_log("wrote %s", path);
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median_of(const double *v, int n) {
    double *t = PLX_NEW(double, n + 1);
    int k = 0;
    for (int i = 0; i < n; i++)
        if (isfinite(v[i])) t[k++] = v[i];
    qsort(t, k, sizeof(double), cmp_double);
    double m = k ? (k % 2 ? t[k / 2] : 0.5 * (t[k / 2 - 1] + t[k / 2])) : NAN;
    free(t);
    return m;
}

/* Weighted centred correlation and RMS difference of a and b over w > 0 (finite entries). */
static void compare(int n, const double *a, const double *b, const double *w, double *corr, double *rms) {
    double W = 0, ma = 0, mb = 0;
    for (int i = 0; i < n; i++)
        if (w[i] > 0 && isfinite(a[i]) && isfinite(b[i])) W += w[i], ma += w[i] * a[i], mb += w[i] * b[i];
    ma /= W, mb /= W;
    double ab = 0, aa = 0, bb = 0, d2 = 0;
    for (int i = 0; i < n; i++)
        if (w[i] > 0 && isfinite(a[i]) && isfinite(b[i])) {
            double x = a[i] - ma, y = b[i] - mb;
            ab += w[i] * x * y, aa += w[i] * x * x, bb += w[i] * y * y, d2 += w[i] * (a[i] - b[i]) * (a[i] - b[i]);
        }
    if (corr) *corr = ab / sqrt(aa * bb);
    if (rms) *rms = sqrt(d2 / W);
}

/* ---- context: the frozen model, the population depth prior -------------------------------------- */
typedef struct {
    plx_fl_model     model;
    plx_fl_fitter    fitter;
    plx_fu_locator   loc;
    double           zbar8[17 * 21];   /* population mean depth, 8-px lattice */
    double           prior_var8[17 * 21]; /* per-node prior variance of personal depth, px² */
    plx_fu_track_opts topt;
    plx_fu_factor_opts fopt;
    const char      *flow, *geom, *pack, *lfw;
} context;

static void load_context(const plx_args *a, context *cx) {
    memset(cx, 0, sizeof *cx);
    cx->flow = plx_arg_str(a, "flow", "runs/main/flow");
    cx->geom = plx_arg_str(a, "geom", "runs/main/geom");
    cx->pack = plx_arg_str(a, "pack", "runs/main/pack");
    cx->lfw = plx_arg_str(a, "lfw", "data/lfw");
    char path[4096];
    snprintf(path, sizeof path, "%s/model", cx->flow);
    if (plx_fl_model_load(path, &cx->model) != PLX_OK) plx_die("cannot load the flow model %s", path);
    plx_fl_fitter_init(&cx->fitter, &cx->model);
    plx_fu_locator_init(&cx->loc, &cx->model);
    plx_tensor t;
    snprintf(path, sizeof path, "%s/zbar.plxt", cx->geom);
    if (plx_tensor_load(path, &t) != PLX_OK || t.dtype != PLX_F32 || plx_tensor_count(&t) != 357)
        plx_die("cannot read %s (f32 [21, 17])", path);
    for (int k = 0; k < 357; k++) cx->zbar8[k] = ((float *)t.data)[k];
    plx_tensor_free(&t);
    snprintf(path, sizeof path, "%s/depth_basis.plxt", cx->geom);
    for (int k = 0; k < 357; k++) cx->prior_var8[k] = 0.25; /* floor: (0.5 px)² */
    if (plx_tensor_load(path, &t) == PLX_OK && t.dtype == PLX_F32 && t.ndim == 3 && t.dim[1] * t.dim[2] == 357) {
        const float *B = t.data;
        for (int j = 0; j < t.dim[0]; j++)
            for (int k = 0; k < 357; k++)
                if (isfinite(B[j * 357 + k])) cx->prior_var8[k] += (double)B[j * 357 + k] * B[j * 357 + k];
        plx_tensor_free(&t);
    } else {
        plx_log("no usable %s: prior variance floor only", path);
    }
    cx->topt = plx_fu_track_defaults();
    cx->topt.step = (int)plx_arg_int(a, "step", cx->topt.step);
    cx->topt.lambda = (float)plx_arg_num(a, "lambda", cx->topt.lambda);
    cx->topt.rounds = (int)plx_arg_int(a, "rounds", cx->topt.rounds);
    cx->fopt = plx_fu_factor_defaults();
    cx->fopt.boot = (int)plx_arg_int(a, "boot", cx->fopt.boot);
    if (plx_arg_int(a, "threads", 0) > 0) plx_set_num_threads((int)plx_arg_int(a, "threads", 0));
}

static void free_context(context *cx) {
    plx_fu_locator_free(&cx->loc);
    plx_fl_fitter_free(&cx->fitter);
    plx_fl_model_free(&cx->model);
}

/* ---- lattice nodes ------------------------------------------------------------------------------------ */
typedef struct {
    plx_lattice L;
    int     P;
    double *X0, *Y0, *zp, *gw;
    uint8_t *track, *core;
} nodes;

static bool in_ellipse(double X, double Y, double ax, double ay, double yc) {
    double ex = X / ax, ey = (Y - yc) / ay;
    return ex * ex + ey * ey <= 1;
}

static void nodes_make(const context *cx, const plx_lattice *L, nodes *nd) {
    nd->L = *L;
    nd->P = plx_lat_nodes(L);
    int P = nd->P;
    nd->X0 = PLX_NEW(double, P), nd->Y0 = PLX_NEW(double, P), nd->zp = PLX_NEW(double, P), nd->gw = PLX_NEW(double, P);
    nd->track = PLX_NEW(uint8_t, P), nd->core = PLX_NEW(uint8_t, P);
    plx_lattice L8 = plx_lattice_make(PLX_FL_STEP);
    float zf[2 * 357] = {0};
    for (int k = 0; k < 357; k++) zf[k] = (float)cx->zbar8[k];
    for (int n = 0; n < P; n++) {
        double x = plx_lat_x(L, n % L->nx), y = plx_lat_y(L, n / L->nx), v[2];
        nd->X0[n] = x - PLX_CANON_CX, nd->Y0[n] = y - PLX_CANON_CY;
        plx_lat_eval(&L8, zf, x, y, v); /* exact on 8-px nodes, bilinear between them */
        nd->zp[n] = v[0];
        nd->track[n] = in_ellipse(nd->X0[n], nd->Y0[n], TRACK_AX, TRACK_AY, TRACK_YC);
        nd->core[n] = in_ellipse(nd->X0[n], nd->Y0[n], CORE_AX, CORE_AY, CORE_YC);
        nd->gw[n] = nd->core[n];
    }
}

static void nodes_free(nodes *nd) {
    free(nd->X0), free(nd->Y0), free(nd->zp), free(nd->gw), free(nd->track), free(nd->core);
}

/* ---- the clip pipeline --------------------------------------------------------------------------- */
typedef struct {
    int           F;
    plx_fu_window win;
    plx_fu_located loc;
    plx_img      *crops, *grays; /* [F] 250 x 250 */
    double       *t_ms;
    plx_fu_track  tr;
    nodes         nd;
    double       *x, *x_pop, *w; /* [F][2][P], [F][P] */
    plx_fu_shape  sh, sh_pop;    /* from the personal tracks, and from the population tracks */
    double       *dense;         /* [3][128 * 160] canonical-gauge surface */
    plx_img       texture, texture_w;
    double        seconds[4];    /* locate, track, factor, fuse */
} clip_run;

/* Base factorisation weights: tracked region x data share of the node's personal fit x the
 * frame's local fit quality at the node (Cauchy on its residual relative to the clip median). */
static void base_weights(const clip_run *c, double *w) {
    int F = c->F, P = c->nd.P;
    double *nc = PLX_NEW(double, (size_t)F * P);
    for (size_t k = 0; k < (size_t)F * P; k++) nc[k] = c->tr.node_cost[k];
    double med = median_of(nc, F * P);
    free(nc);
    for (int f = 0; f < F; f++)
        for (int n = 0; n < P; n++) {
            double cst = c->tr.node_cost[(size_t)f * P + n], q = isfinite(cst) ? cst / (2 * med) : 1e9;
            double d = fmax(c->tr.node_data[n], 0.05);
            w[(size_t)f * P + n] = c->nd.track[n] ? d / (1 + q * q) : 0;
        }
}

static plx_fu_tracks tracks_of(const clip_run *c, const double *x) {
    return (plx_fu_tracks){.F = c->F, .P = c->nd.P, .nx = c->nd.L.nx, .ny = c->nd.L.ny, .img_w = PLX_CANON_IMG_SIDE,
                           .img_h = PLX_CANON_IMG_SIDE, .x = x, .w = c->w, .X0 = c->nd.X0, .Y0 = c->nd.Y0,
                           .zp = c->nd.zp, .gw = c->nd.gw};
}

/* frames: F frames of any size and 1 or 3 channels. */
static void run_clip(const context *cx, const plx_img *frames, int F, const double *t_ms, clip_run *c) {
    memset(c, 0, sizeof *c);
    c->F = F;
    c->t_ms = PLX_NEW(double, F);
    memcpy(c->t_ms, t_ms, sizeof(double) * F);
    double t0 = plx_now();
    plx_img g0 = plx_img_luma(&frames[0]);
    plx_fu_locate_opts lo = plx_fu_locate_defaults();
    c->loc = plx_fu_locate(&cx->loc, &g0, &lo);
    plx_img_free(&g0);
    c->win = plx_fu_window_for(c->loc.T);
    c->crops = PLX_NEW(plx_img, F);
    c->grays = PLX_NEW(plx_img, F);
    for (int f = 0; f < F; f++) {
        c->crops[f] = plx_fu_window_extract(&frames[f], c->win);
        c->grays[f] = plx_img_luma(&c->crops[f]);
    }
    c->seconds[0] = plx_now() - t0;
    t0 = plx_now();
    plx_fu_track_clip(&cx->fitter, c->grays, F, plx_fu_sim_to_crop(c->win, c->loc.T), &cx->topt, &c->tr);
    c->seconds[1] = plx_now() - t0;
    t0 = plx_now();
    nodes_make(cx, &c->tr.lat, &c->nd);
    int P = c->nd.P;
    c->x = PLX_NEW(double, (size_t)F * 2 * P);
    c->x_pop = PLX_NEW(double, (size_t)F * 2 * P);
    c->w = PLX_NEW(double, (size_t)F * P);
    plx_fu_track_positions(&c->tr, false, c->x);
    plx_fu_track_positions(&c->tr, true, c->x_pop);
    base_weights(c, c->w);
    {   /* DEBUG */
        int bad = 0, badw = 0;
        for (size_t k = 0; k < (size_t)F * 2 * P; k++) bad += !isfinite(c->x[k]) || !isfinite(c->x_pop[k]);
        for (size_t k = 0; k < (size_t)F * P; k++) badw += !isfinite(c->w[k]);
        plx_log("DEBUG loc T %g %g %g %g cost %g; bad x %d bad w %d; T_pop0 %g %g %g %g", c->loc.T.a, c->loc.T.b, c->loc.T.tx,
                c->loc.T.ty, c->loc.score, bad, badw, c->tr.T_pop[0].a, c->tr.T_pop[0].b, c->tr.T_pop[0].tx, c->tr.T_pop[0].ty);
    }
    plx_fu_tracks tp = tracks_of(c, c->x), tq = tracks_of(c, c->x_pop);
    if (plx_fu_factorise(&tp, &cx->fopt, &c->sh) != PLX_OK) plx_die("factorisation failed (personal tracks)");
    plx_fu_factor_opts fq = cx->fopt;
    fq.boot = 0;
    if (plx_fu_factorise(&tq, &fq, &c->sh_pop) != PLX_OK) plx_die("factorisation failed (population tracks)");
    c->seconds[2] = plx_now() - t0;
    t0 = plx_now();
    c->dense = PLX_NEW(double, (size_t)3 * PLX_CANON_W * PLX_CANON_H);
    plx_fu_dense_surface(&c->nd.L, c->sh.S, c->sh.used, true, c->dense);
    plx_fu_fuse_texture(c->crops, F, &c->tr, &c->sh, c->dense, &c->texture, &c->texture_w);
    c->seconds[3] = plx_now() - t0;
}

static void free_clip(clip_run *c) {
    for (int f = 0; f < c->F; f++) plx_img_free(&c->crops[f]), plx_img_free(&c->grays[f]);
    free(c->crops), free(c->grays), free(c->t_ms);
    plx_fu_track_free(&c->tr);
    nodes_free(&c->nd);
    free(c->x), free(c->x_pop), free(c->w), free(c->dense);
    plx_fu_shape_free(&c->sh), plx_fu_shape_free(&c->sh_pop);
    plx_img_free(&c->texture), plx_img_free(&c->texture_w);
}

static double mean_of(const double *v, int n) {
    double s = 0;
    int k = 0;
    for (int i = 0; i < n; i++)
        if (isfinite(v[i])) s += v[i], k++;
    return k ? s / k : NAN;
}

/* Depth protrusion of the nose: z at the tip node (nearest (0, 10)) minus the mean of the
 * cheek nodes (nearest (±30, 10)); the same nodes for any z on these nodes. */
static void nose_nodes(const nodes *nd, int *tip, int *cl, int *cr) {
    double bt = 1e9, bl = 1e9, br = 1e9;
    for (int n = 0; n < nd->P; n++) {
        double dt = hypot(nd->X0[n], nd->Y0[n] - 10), dl = hypot(nd->X0[n] + 30, nd->Y0[n] - 10),
               dr = hypot(nd->X0[n] - 30, nd->Y0[n] - 10);
        if (dt < bt) bt = dt, *tip = n;
        if (dl < bl) bl = dl, *cl = n;
        if (dr < br) br = dr, *cr = n;
    }
}

static double protrusion(const nodes *nd, const double *z) {
    int t = 0, l = 0, r = 0;
    nose_nodes(nd, &t, &l, &r);
    return z[t] - 0.5 * (z[l] + z[r]);
}

/* Record contents from a run. */
static void make_record(const context *cx, const clip_run *c, plx_fu_record *r) {
    memset(r, 0, sizeof *r);
    const plx_fu_shape *sh = &c->sh;
    int F = c->F, P = c->nd.P, P8 = 357;
    plx_lattice L8 = plx_lattice_make(PLX_FL_STEP);
    r->F = F;
    r->C = c->texture.c;
    r->lat = c->nd.L;
    r->shape_lat = PLX_NEW(float, (size_t)7 * P);
    for (int n = 0; n < P; n++) {
        double fw = 0;
        for (int f = 0; f < F; f++) fw += sh->w[(size_t)f * P + n] / F;
        float v[7] = {(float)(sh->S[n] - c->nd.X0[n]), (float)(sh->S[P + n] - c->nd.Y0[n]), (float)sh->S[2 * P + n],
                      (float)sh->sd[n], (float)sh->sd[P + n], (float)sh->sd[2 * P + n], (float)fw};
        for (int k = 0; k < 7; k++) r->shape_lat[(size_t)k * P + n] = sh->used[n] || k == 6 ? v[k] : NAN;
    }
    /* 8-px lattice: a 4-px lattice contains its nodes (every second row and column) */
    int ratio = PLX_FL_STEP / (int)c->nd.L.step;
    r->shape = PLX_NEW(float, (size_t)7 * P8);
    r->post = PLX_NEW(float, (size_t)2 * P8);
    for (int n = 0; n < P8; n++) {
        int i = n / L8.nx, j = n % L8.nx, m = (i * ratio) * c->nd.L.nx + j * ratio;
        for (int k = 0; k < 7; k++) r->shape[(size_t)k * P8 + n] = r->shape_lat[(size_t)k * P + m];
        /* DESIGN §5.5 / §8: precisions add (per node: the record keeps the diagonal) */
        double z = r->shape[2 * P8 + n], sd = r->shape[5 * P8 + n], v0 = cx->prior_var8[n];
        if (isfinite(z) && isfinite(sd) && sd > 0) {
            double prec = 1 / v0 + 1 / (sd * sd);
            r->post[n] = (float)((cx->zbar8[n] / v0 + z / (sd * sd)) / prec);
            r->post[P8 + n] = (float)sqrt(1 / prec);
        } else {
            r->post[n] = (float)cx->zbar8[n];
            r->post[P8 + n] = (float)sqrt(v0);
        }
    }
    r->depth = plx_img_new(PLX_CANON_W, PLX_CANON_H, 1);
    for (int k = 0; k < PLX_CANON_W * PLX_CANON_H; k++) r->depth.px[k] = (float)c->dense[2 * PLX_CANON_W * PLX_CANON_H + k];
    r->texture = plx_img_clone(&c->texture);
    r->texture_w = plx_img_clone(&c->texture_w);
    r->tmpl = plx_img_clone(&c->tr.tmpl_gray);
    r->pose = PLX_NEW(float, (size_t)F * 9);
    for (int f = 0; f < F; f++) {
        double fw = 0;
        for (int n = 0; n < P; n++) fw += sh->w[(size_t)f * P + n];
        float v[9] = {(float)c->t_ms[f], (float)sh->yaw[f], (float)sh->pitch[f], (float)sh->roll[f], (float)sh->scale[f],
                      (float)sh->tx[f], (float)sh->ty[f], (float)sh->frame_rms[f], (float)fw};
        memcpy(r->pose + (size_t)f * 9, v, sizeof v);
    }
    r->tracks = PLX_NEW(float, (size_t)F * 3 * P);
    for (int f = 0; f < F; f++)
        for (int n = 0; n < P; n++) {
            r->tracks[((size_t)f * 3) * P + n] = (float)c->x[((size_t)f * 2) * P + n];
            r->tracks[((size_t)f * 3 + 1) * P + n] = (float)c->x[((size_t)f * 2 + 1) * P + n];
            r->tracks[((size_t)f * 3 + 2) * P + n] = (float)sh->w[(size_t)f * P + n];
        }
    char *info = plx_malloc(8192);
    double zc[357];
    for (int n = 0; n < P8; n++) zc[n] = r->shape[2 * P8 + n];
    nodes n8;
    nodes_make(cx, &L8, &n8);
    double cpp = mean_of(c->tr.cost_pp, F), cp = mean_of(c->tr.cost, F);
    int n_used = 0;
    for (int n = 0; n < P; n++) n_used += sh->used[n];
    snprintf(info, 8192,
             "plx_fuse_record 1\nframes %d\nchannels %d\nlattice_step %d\nwindow %.6f %.6f %.6f\n"
             "locate_score %.4f\nlocate_runner_score %.4f\nreproj_rms_px %.4f\ntrack_sigma_px %.4f\n"
             "metric_gap %.6g\nL_min_eig %.6g\nrotation_spread_deg %.3f %.3f %.3f\ndegenerate %d\n"
             "depth_ratio %.4f\nprior_scale %.4f\ndepth_scale_sd %.4f\nmirror_corr %.4f %.4f\ngauge_rms_px %.4f\n"
             "nose_protrusion_px %.3f\ntrack_cost_population %.4f\ntrack_cost_personal %.4f\nnodes_used %d\n",
             F, r->C, (int)c->nd.L.step, c->win.s, c->win.x0, c->win.y0, c->loc.score, c->loc.runner_score, sh->rms,
             sh->sigma, sh->gap, sh->L_min_eig, sh->spread_deg[0], sh->spread_deg[1], sh->spread_deg[2],
             (int)sh->degenerate, sh->depth_ratio, sh->prior_scale, sh->scale_sd, sh->corr, sh->corr_mirror,
             sh->gauge_rms, protrusion(&n8, zc), cpp, cp, n_used);
    nodes_free(&n8);
    r->info = info;
}

/* ---- drawing ---------------------------------------------------------------------------------- */
static plx_img canvas(int w, int h) {
    plx_img im = plx_img_new(w, h, 3);
    for (size_t k = 0; k < (size_t)3 * w * h; k++) im.px[k] = 0.1f;
    return im;
}

static void blit(plx_img *dst, const plx_img *src, int x0, int y0) {
    plx_img rgb = plx_viz_to_rgb(src);
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < rgb.h; y++)
            for (int x = 0; x < rgb.w; x++)
                if (x0 + x >= 0 && y0 + y >= 0 && x0 + x < dst->w && y0 + y < dst->h)
                    PLX_PX(dst, x0 + x, y0 + y, c) = PLX_CLAMP(PLX_PX(&rgb, x, y, c), 0.0f, 1.0f);
    plx_img_free(&rgb);
}

static plx_img vstack(const plx_img *im, const char *const *titles, int n) {
    int W = 0, H = 0, pad = 6, th = 14;
    for (int i = 0; i < n; i++) W = PLX_MAX(W, im[i].w), H += im[i].h + pad + (titles && titles[i] ? th : 0);
    plx_img out = canvas(W + 2 * pad, H + pad);
    int y = pad;
    for (int i = 0; i < n; i++) {
        if (titles && titles[i]) plx_draw_text(&out, pad, y, titles[i], 1, WHITE), y += th;
        blit(&out, &im[i], pad, y);
        y += im[i].h + pad;
    }
    return out;
}

typedef struct { double x0, x1, y0, y1; int L, R, T, B, w, h; } plot_frame;
static float px_x(const plot_frame *f, double x) { return (float)(f->L + (x - f->x0) / (f->x1 - f->x0) * (f->w - f->L - f->R)); }
static float px_y(const plot_frame *f, double y) { return (float)(f->h - f->B - (y - f->y0) / (f->y1 - f->y0) * (f->h - f->T - f->B)); }

static plx_img plot_axes(const plot_frame *f, const char *title, const char *xl, const char *yl) {
    plx_img im = canvas(f->w, f->h);
    plx_draw_line(&im, (float)f->L, (float)(f->h - f->B), (float)(f->w - f->R), (float)(f->h - f->B), 1, GREY, 1);
    plx_draw_line(&im, (float)f->L, (float)f->T, (float)f->L, (float)(f->h - f->B), 1, GREY, 1);
    char buf[64];
    for (int k = 0; k <= 4; k++) {
        double x = f->x0 + k * (f->x1 - f->x0) / 4, y = f->y0 + k * (f->y1 - f->y0) / 4;
        snprintf(buf, sizeof buf, "%.3g", x);
        plx_draw_text(&im, (int)px_x(f, x) - 10, f->h - f->B + 6, buf, 1, GREY);
        snprintf(buf, sizeof buf, "%.3g", y);
        plx_draw_text(&im, 4, (int)px_y(f, y) - 4, buf, 1, GREY);
    }
    if (f->y0 < 0 && f->y1 > 0)
        plx_draw_line(&im, (float)f->L, px_y(f, 0), (float)(f->w - f->R), px_y(f, 0), 1, GREY, 0.4f);
    plx_draw_text(&im, f->L, 6, title, 1, WHITE);
    plx_draw_text(&im, f->w / 2 - 40, f->h - 14, xl, 1, WHITE);
    plx_draw_text(&im, 4, f->T - 14, yl, 1, WHITE);
    return im;
}

static void plot_series(plx_img *im, const plot_frame *f, const double *x, const double *y, int n, const float *col,
                        bool dashed) {
    for (int i = 0; i + 1 < n; i++) {
        if (!isfinite(y[i]) || !isfinite(y[i + 1]) || (dashed && i % 2)) continue;
        plx_draw_line(im, px_x(f, x[i]), px_y(f, y[i]), px_x(f, x[i + 1]), px_y(f, y[i + 1]), dashed ? 1.0f : 1.6f, col, 1);
    }
}

/* Frame with the tracked lattice: grid lines, nodes coloured by final weight (green = trusted). */
static plx_img draw_tracks(const clip_run *c, int f) {
    plx_img im = plx_viz_to_rgb(&c->crops[f]);
    int P = c->nd.P, nx = c->nd.L.nx;
    const double *x = c->x + (size_t)f * 2 * P, *y = x + P;
    double wmax = 0;
    for (int n = 0; n < P; n++) wmax = fmax(wmax, c->sh.w[(size_t)f * P + n]);
    for (int n = 0; n < P; n++) {
        if (!c->nd.track[n]) continue;
        if (n % nx + 1 < nx && c->nd.track[n + 1]) plx_draw_line(&im, (float)x[n], (float)y[n], (float)x[n + 1], (float)y[n + 1], 0.6f, CYAN, 0.5f);
        if (n + nx < P && c->nd.track[n + nx]) plx_draw_line(&im, (float)x[n], (float)y[n], (float)x[n + nx], (float)y[n + nx], 0.6f, CYAN, 0.5f);
    }
    for (int n = 0; n < P; n++) {
        if (!c->nd.track[n]) continue;
        float t = (float)(wmax > 0 ? c->sh.w[(size_t)f * P + n] / wmax : 0), col[3] = {1 - t, 0.3f + 0.7f * t, 0.2f};
        plx_draw_disc(&im, (float)x[n], (float)y[n], 1.6f, col, 1);
    }
    char buf[64];
    snprintf(buf, sizeof buf, "frame %d yaw %+.0f pitch %+.0f", f, c->sh.yaw[f] / DEG, c->sh.pitch[f] / DEG);
    plx_draw_text(&im, 4, 4, buf, 1, YELLOW);
    return im;
}

/* Lattice field (P values on nodes) as an Ω-sized image, nearest node, NaN where unused. */
static plx_img node_image(const nodes *nd, const double *v, const uint8_t *used) {
    plx_img im = plx_img_new(PLX_CANON_W, PLX_CANON_H, 1);
    for (int y = 0; y < PLX_CANON_H; y++)
        for (int x = 0; x < PLX_CANON_W; x++) {
            int j = (int)lrint((x - PLX_LAT_ORIGIN) / nd->L.step), i = (int)lrint((y - PLX_LAT_ORIGIN) / nd->L.step);
            int n = PLX_CLAMP(i, 0, nd->L.ny - 1) * nd->L.nx + PLX_CLAMP(j, 0, nd->L.nx - 1);
            PLX_PX(&im, x, y, 0) = used[n] ? (float)v[n] : NAN;
        }
    return im;
}

static plx_img labelled_row(const plx_img *tiles, const char *const *labels, int n, int scale) {
    plx_img *up = PLX_NEW(plx_img, n);
    for (int i = 0; i < n; i++) up[i] = scale > 1 ? plx_viz_upscale(&tiles[i], scale, PLX_UP_BILINEAR) : plx_img_clone(&tiles[i]);
    plx_montage_opts mo = plx_montage_defaults();
    mo.cols = n;
    plx_img out = plx_viz_montage(up, n, labels, &mo);
    for (int i = 0; i < n; i++) plx_img_free(&up[i]);
    free(up);
    return out;
}

/* The clip's diagnostic picture. truth (nullable): true yaw / pitch / roll per frame (rad). */
static plx_img clip_diag(const clip_run *c, const double *true_ang) {
    int F = c->F, P = c->nd.P;
    plx_img rows[5];
    const char *titles[5];
    /* 1: tracked lattice over four frames */
    {
        plx_img t[4];
        int pick[4] = {0, F / 4, F / 2, 3 * F / 4};
        for (int k = 0; k < 4; k++) t[k] = draw_tracks(c, PLX_MIN(pick[k], F - 1));
        rows[0] = labelled_row(t, NULL, 4, 1);
        for (int k = 0; k < 4; k++) plx_img_free(&t[k]);
        titles[0] = "tracked lattice (personal fit); node colour = final factorisation weight (green high, red low)";
    }
    /* 2: faces and maps */
    {
        plx_img t[7];
        t[0] = plx_img_clone(&c->tr.pop_gray);
        t[1] = plx_img_clone(&c->tr.tmpl_gray);
        t[2] = plx_img_clone(&c->texture);
        for (size_t k = 0; k < (size_t)t[2].w * t[2].h * t[2].c; k++) t[2].px[k] = PLX_CLAMP(t[2].px[k], 0.0f, 1.0f);
        float *z = PLX_NEW(float, PLX_CANON_W * PLX_CANON_H);
        for (int k = 0; k < PLX_CANON_W * PLX_CANON_H; k++) z[k] = (float)c->dense[2 * PLX_CANON_W * PLX_CANON_H + k];
        float lo, hi;
        plx_viz_auto_range(z, PLX_CANON_W * PLX_CANON_H, 0.01f, 0.99f, &lo, &hi);
        t[3] = plx_viz_colormap(z, PLX_CANON_W, PLX_CANON_H, lo, hi, PLX_CMAP_SEQ);
        free(z);
        double *sdz = PLX_NEW(double, P), *nr = PLX_NEW(double, P), *zd = PLX_NEW(double, P);
        for (int n = 0; n < P; n++) sdz[n] = c->sh.sd[2 * P + n], nr[n] = c->sh.node_rms[n], zd[n] = c->sh.S[2 * P + n] - c->nd.zp[n];
        plx_img a = node_image(&c->nd, sdz, c->sh.used), b = node_image(&c->nd, nr, c->sh.used), d = node_image(&c->nd, zd, c->sh.used);
        plx_viz_auto_range(d.px, (size_t)d.w * d.h, 0.01f, 0.99f, &lo, &hi);
        t[4] = plx_viz_colormap(d.px, d.w, d.h, lo, hi, PLX_CMAP_DIV);
        plx_viz_auto_range(a.px, (size_t)a.w * a.h, 0.0f, 0.98f, &lo, &hi);
        t[5] = plx_viz_colormap(a.px, a.w, a.h, 0, hi, PLX_CMAP_SEQ);
        plx_viz_auto_range(b.px, (size_t)b.w * b.h, 0.0f, 0.98f, &lo, &hi);
        t[6] = plx_viz_colormap(b.px, b.w, b.h, 0, hi, PLX_CMAP_SEQ);
        plx_img_free(&a), plx_img_free(&b), plx_img_free(&d);
        free(sdz), free(nr), free(zd);
        const char *lab[7] = {"population fits", "personal template", "fused texture", "depth z", "z - zbar", "sd(z)", "non-rigid rms"};
        rows[1] = labelled_row(t, lab, 7, 2);
        for (int k = 0; k < 7; k++) plx_img_free(&t[k]);
        titles[1] = "mean faces, fused texture, depth (Ω), depth minus the population mean, bootstrap sd, per-node residual";
    }
    /* 3: renders at novel poses */
    {
        plx_img z = plx_img_new(PLX_CANON_W, PLX_CANON_H, 1);
        for (int k = 0; k < PLX_CANON_W * PLX_CANON_H; k++) z.px[k] = (float)c->dense[2 * PLX_CANON_W * PLX_CANON_H + k];
        plx_img tex = plx_img_clone(&c->texture);
        for (size_t k = 0; k < (size_t)tex.w * tex.h * tex.c; k++) tex.px[k] = PLX_CLAMP(tex.px[k], 0.0f, 1.0f);
        plx_pose poses[7] = {{-45 * DEG, 0, 0}, {-25 * DEG, 0, 0}, {0, 0, 0}, {25 * DEG, 0, 0}, {45 * DEG, 0, 0}, {0, 25 * DEG, 0}, {0, -25 * DEG, 0}};
        rows[2] = plx_render_views(&z, NULL, &tex, poses, 7);
        plx_img_free(&z), plx_img_free(&tex);
        titles[2] = "reconstructed head at novel poses: lit geometry (top), fused texture (bottom)";
    }
    /* 4: pose track */
    {
        double *fx = PLX_NEW(double, F), *v = PLX_NEW(double, F), lim = 10;
        for (int f = 0; f < F; f++) {
            fx[f] = f;
            lim = fmax(lim, fmax(fabs(c->sh.yaw[f]), fabs(c->sh.pitch[f])) / DEG + 5);
        }
        plot_frame pf = {.x0 = 0, .x1 = PLX_MAX(F - 1, 1), .y0 = -lim, .y1 = lim, .L = 50, .R = 20, .T = 30, .B = 36, .w = 900, .h = 260};
        rows[3] = plot_axes(&pf, "pose track: yaw (yellow) pitch (cyan) roll (red); dashed = truth", "frame", "degrees");
        const double *ang[3] = {c->sh.yaw, c->sh.pitch, c->sh.roll};
        const float *col[3] = {YELLOW, CYAN, RED};
        for (int k = 0; k < 3; k++) {
            for (int f = 0; f < F; f++) v[f] = ang[k][f] / DEG;
            plot_series(&rows[3], &pf, fx, v, F, col[k], false);
            if (true_ang) {
                for (int f = 0; f < F; f++) v[f] = true_ang[f * 3 + k] / DEG;
                plot_series(&rows[3], &pf, fx, v, F, col[k], true);
            }
        }
        free(fx), free(v);
        titles[3] = NULL;
    }
    /* 5: reprojection residual per frame */
    {
        double *fx = PLX_NEW(double, F), mx = 0.5;
        for (int f = 0; f < F; f++) fx[f] = f, mx = fmax(mx, 1.2 * (isfinite(c->sh.frame_rms[f]) ? c->sh.frame_rms[f] : 0));
        for (int f = 0; f < F; f++) mx = fmax(mx, 1.2 * (isfinite(c->sh_pop.frame_rms[f]) ? c->sh_pop.frame_rms[f] : 0));
        plot_frame pf = {.x0 = 0, .x1 = PLX_MAX(F - 1, 1), .y0 = 0, .y1 = mx, .L = 50, .R = 20, .T = 30, .B = 36, .w = 900, .h = 200};
        rows[4] = plot_axes(&pf, "rigid reprojection rms per frame: personal tracks (green), population tracks (red)", "frame", "px");
        plot_series(&rows[4], &pf, fx, c->sh.frame_rms, F, GREEN, false);
        plot_series(&rows[4], &pf, fx, c->sh_pop.frame_rms, F, RED, false);
        free(fx);
        titles[4] = NULL;
    }
    plx_img out = vstack(rows, titles, 5);
    for (int k = 0; k < 5; k++) plx_img_free(&rows[k]);
    return out;
}

static void report_clip(const clip_run *c) {
    const plx_fu_shape *s = &c->sh, *q = &c->sh_pop;
    int nu = 0;
    for (int n = 0; n < c->nd.P; n++) nu += s->used[n];
    say("  window s %.3f at (%.1f, %.1f); locate score %.3f (runner-up elsewhere %.3f)\n", c->win.s, c->win.x0,
        c->win.y0, c->loc.score, c->loc.runner_score);
    say("  tracking cost vs the final personal template: population warps %.3f -> personal warps %.3f "
        "(population fit vs its own template %.3f); drift removed per round:",
        mean_of(c->tr.cost_pp, c->F), mean_of(c->tr.cost, c->F), mean_of(c->tr.cost_pop, c->F));
    for (int r = 0; r < c->tr.F && r < 8 && c->tr.drift[r] > 0; r++) say(" %.3f", c->tr.drift[r]);
    say(" px\n");
    say("  rigid reprojection rms: population tracks %.3f px -> personal tracks %.3f px (robust sigma %.3f -> %.3f)\n",
        q->rms, s->rms, q->sigma, s->sigma);
    say("  factorisation: %d nodes; rotation spreads %.2f / %.2f / %.2f deg; metric gap %.2e; L min eig %.3f; "
        "mirror corr %.3f (rejected twin %.3f)\n",
        nu, s->spread_deg[0], s->spread_deg[1], s->spread_deg[2], s->gap, s->L_min_eig, s->corr, s->corr_mirror);
    say("  depth ratio to zbar %.3f; degenerate %s (prior scale %.3f); bootstrap depth-scale sd %.3f (%d resamples)\n",
        s->depth_ratio, s->degenerate ? "YES" : "no", s->prior_scale, s->scale_sd, s->n_boot);
    say("  nose protrusion %.2f px (population zbar %.2f px); time locate %.1f s, track %.1f s, factor %.1f s, fuse %.1f s\n",
        protrusion(&c->nd, s->S + 2 * c->nd.P), protrusion(&c->nd, c->nd.zp), c->seconds[0], c->seconds[1],
        c->seconds[2], c->seconds[3]);
}

/* ---- mode: a clip ----------------------------------------------------------------------------- */
static int run_clip_mode(const plx_args *a) {
    const char *path = plx_arg_str(a, "clip", NULL), *out = plx_arg_str(a, "out", "runs/main/fuse/clip");
    context cx;
    load_context(a, &cx);
    plx_fu_clip clip;
    int st = plx_fu_clip_read(path, &clip);
    if (st != PLX_OK) plx_die("cannot read clip %s: %s", path, plx_status_str(st));
    plx_img *frames = PLX_NEW(plx_img, clip.n);
    for (int f = 0; f < clip.n; f++)
        if ((st = plx_fu_clip_decode(&clip, f, &frames[f])) != PLX_OK) plx_die("frame %d of %s: %s", f, path, plx_status_str(st));
    open_summary(out);
    say("plx_fuse --clip %s: %d frames of %d x %d (%d channels)\n", path, clip.n, frames[0].w, frames[0].h, frames[0].c);
    clip_run c;
    run_clip(&cx, frames, clip.n, clip.t_ms, &c);
    report_clip(&c);
    plx_fu_record r;
    make_record(&cx, &c, &r);
    if (plx_fu_record_save(out, &r) != PLX_OK) plx_die("cannot write the record to %s", out);
    plx_img d = clip_diag(&c, NULL);
    save_png(out, "diag.png", &d);
    plx_img_free(&d);
    say("record in %s\n", out);
    plx_fu_record_free(&r);
    free_clip(&c);
    for (int f = 0; f < clip.n; f++) plx_img_free(&frames[f]);
    free(frames);
    plx_fu_clip_free(&clip);
    free_context(&cx);
    fclose(g_sum);
    return 0;
}

/* ---- synthetic ground truth ------------------------------------------------------------------- */
typedef struct {
    double depth_corr, depth_rms, hp_corr, hp_rms;  /* core nodes, after similarity alignment to truth */
    double nose_true, nose_rec;
    double yaw_rms, pitch_rms, roll_rms;            /* deg, in the truth frame */
    double yaw_rms_gauge;                           /* deg, canonical gauge as output (no alignment) */
    double scale_ratio;                             /* ⟨z_rec, z_true⟩ / ⟨z_true, z_true⟩ (centred) */
    double track_pop, track_pers;                   /* RMS track error vs consistent true points, px */
    double zscore_rms;                              /* RMS of (z_rec - z_true) / sd */
    double loc_err, loc_scale;                      /* localisation: centre error px, scale ratio */
} synth_eval;

typedef struct {
    const plx_fu_head *head;
    const plx_fu_synth_opts *o;
    plx_fu_window win;
    const double *x;  /* [F][2][P] crop positions */
    const double *w;  /* [F][P] */
    int F, P;
    double *XY;       /* out [2][P] true canonical correspondences */
    double *err;      /* out [P] RMS track error */
} corr_ctx;

/* The true surface point a node track follows: the (X, Y) on the true head whose projections
 * best fit the track over the frames where it is visible (Gauss-Newton, numerical Jacobian). */
static void corr_one(int64_t n, int tid, void *pc) {
    (void)tid;
    corr_ctx *c = pc;
    int F = c->F, P = c->P;
    double X = NAN, Y = NAN;
    /* start: invert the most frontal frame's projection approximately (pose known) */
    int f0 = 0;
    double best = 1e9;
    for (int f = 0; f < F; f++) {
        plx_fu_pose p = plx_fu_synth_pose(c->o, f);
        double d = fabs(p.yaw) + fabs(p.pitch);
        if (d < best && c->w[(size_t)f * P + n] > 0) best = d, f0 = f;
    }
    {
        plx_fu_pose p = plx_fu_synth_pose(c->o, f0);
        double fx = c->win.x0 + c->win.s * c->x[((size_t)f0 * 2) * P + n], fy = c->win.y0 + c->win.s * c->x[((size_t)f0 * 2 + 1) * P + n];
        double R[9];
        plx_rot_ypr(p.yaw, p.pitch, p.roll, R);
        /* first order: ignore depth, undo roll and scale */
        double dx = (fx - p.x) / p.s, dy = (fy - p.y) / p.s;
        X = R[0] * dx + R[3] * dy, Y = R[1] * dx + R[4] * dy;
    }
    double e2 = NAN;
    for (int it = 0; it < 15; it++) {
        double H[3] = {0}, g[2] = {0}, ss = 0;
        int m = 0;
        for (int f = 0; f < F; f++) {
            if (c->w[(size_t)f * P + n] <= 0) continue;
            plx_fu_pose p = plx_fu_synth_pose(c->o, f);
            double fx = c->win.x0 + c->win.s * c->x[((size_t)f * 2) * P + n], fy = c->win.y0 + c->win.s * c->x[((size_t)f * 2 + 1) * P + n];
            double px, py, ax, ay, bx, by, h = 0.25;
            plx_fu_synth_project(c->head, c->o, p, X, Y, &px, &py);
            plx_fu_synth_project(c->head, c->o, p, X + h, Y, &ax, &ay);
            plx_fu_synth_project(c->head, c->o, p, X, Y + h, &bx, &by);
            if (!isfinite(px) || !isfinite(ax) || !isfinite(bx)) continue;
            double J[4] = {(ax - px) / h, (bx - px) / h, (ay - py) / h, (by - py) / h}, r[2] = {fx - px, fy - py};
            H[0] += J[0] * J[0] + J[2] * J[2], H[1] += J[0] * J[1] + J[2] * J[3], H[2] += J[1] * J[1] + J[3] * J[3];
            g[0] += J[0] * r[0] + J[2] * r[1], g[1] += J[1] * r[0] + J[3] * r[1];
            ss += r[0] * r[0] + r[1] * r[1];
            m++;
        }
        if (m == 0) break;
        e2 = ss / m;
        double det = H[0] * H[2] - H[1] * H[1];
        if (!(fabs(det) > 1e-12)) break;
        double dX = (H[2] * g[0] - H[1] * g[1]) / det, dY = (H[0] * g[1] - H[1] * g[0]) / det;
        X += dX, Y += dY;
        if (hypot(dX, dY) < 1e-4) break;
    }
    c->XY[n] = X, c->XY[P + n] = Y;
    c->err[n] = sqrt(e2) / c->win.s; /* in crop pixels */
}

static void true_correspondences(const plx_fu_head *h, const plx_fu_synth_opts *o, const clip_run *c, const double *x,
                                 double *XY, double *err) {
    corr_ctx cc = {.head = h, .o = o, .win = c->win, .x = x, .w = c->sh.w, .F = c->F, .P = c->nd.P, .XY = XY, .err = err};
    plx_parallel_for(c->nd.P, 4, corr_one, &cc);
}

/* Residual of z after removing the best quadratic in (X, Y) (the smooth dome): what is left is
 * the nose, the eye sockets, the lips: "is the nose there". */
static void highpass(int P, const double *X, const double *Y, const double *z, const double *w, double *out) {
    plx_mat A = plx_mat_new(P, 6), B = plx_mat_new(P, 1), C = plx_mat_new(6, 1);
    for (int n = 0; n < P; n++) {
        double s = w[n] > 0 && isfinite(z[n]) ? 1 : 0, b[6] = {1, X[n], Y[n], X[n] * X[n], X[n] * Y[n], Y[n] * Y[n]};
        for (int k = 0; k < 6; k++) PLX_M(&A, n, k) = s * b[k] / (k >= 3 ? 1000 : k ? 30 : 1);
        PLX_M(&B, n, 0) = s ? z[n] : 0;
    }
    plx_lstsq(&A, &B, 0, 0, &C);
    for (int n = 0; n < P; n++) {
        double b[6] = {1, X[n], Y[n], X[n] * X[n], X[n] * Y[n], Y[n] * Y[n]}, q = 0;
        for (int k = 0; k < 6; k++) q += PLX_M(&C, k, 0) * b[k] / (k >= 3 ? 1000 : k ? 30 : 1);
        out[n] = z[n] - q;
    }
    plx_mat_free(&A), plx_mat_free(&B), plx_mat_free(&C);
}

/* Compares a recovered shape with a head (truth at the given correspondences); fills the depth
 * fields of ev. Rq: the alignment truth = s R rec + t is returned for the pose comparison. */
static void eval_depth(const plx_fu_head *h, const nodes *nd, const plx_fu_shape *sh, const double *XY, synth_eval *ev,
                       double Rq[9]) {
    int P = nd->P;
    double *T = PLX_NEW(double, (size_t)3 * P), *S = PLX_NEW(double, (size_t)3 * P), *w = PLX_NEW(double, P);
    for (int n = 0; n < P; n++) {
        T[n] = XY[n], T[P + n] = XY[P + n], T[2 * P + n] = plx_fu_head_depth(h, XY[n], XY[P + n]);
        w[n] = sh->used[n] && nd->core[n] && isfinite(T[2 * P + n]) && isfinite(XY[n]);
        for (int r = 0; r < 3; r++) S[r * P + n] = w[n] ? sh->S[r * P + n] : 0;
        if (!w[n]) T[n] = T[P + n] = T[2 * P + n] = 0;
    }
    double s, t[3];
    plx_fu_procrustes(P, S, T, w, &s, Rq, t);
    double *za = PLX_NEW(double, P), *zt = T + 2 * P, *ha = PLX_NEW(double, P), *ht = PLX_NEW(double, P);
    for (int n = 0; n < P; n++)
        za[n] = s * (Rq[6] * S[n] + Rq[7] * S[P + n] + Rq[8] * S[2 * P + n]) + t[2];
    compare(P, za, zt, w, &ev->depth_corr, &ev->depth_rms);
    highpass(P, T, T + P, zt, w, ht);
    highpass(P, T, T + P, za, w, ha);
    compare(P, ha, ht, w, &ev->hp_corr, &ev->hp_rms);
    ev->nose_true = protrusion(nd, zt);
    ev->nose_rec = protrusion(nd, za);
    double W = 0, ma = 0, mt = 0, ab = 0, bb = 0, z2 = 0;
    int nz = 0;
    for (int n = 0; n < P; n++)
        if (w[n] > 0) W += 1, ma += za[n], mt += zt[n];
    ma /= W, mt /= W;
    for (int n = 0; n < P; n++)
        if (w[n] > 0) {
            ab += (za[n] - ma) * (zt[n] - mt), bb += (zt[n] - mt) * (zt[n] - mt);
            double sd = s * sh->sd[2 * P + n];
            if (isfinite(sd) && sd > 0) z2 += (za[n] - zt[n]) * (za[n] - zt[n]) / (sd * sd), nz++;
        }
    ev->scale_ratio = ab / bb;
    ev->zscore_rms = nz ? sqrt(z2 / nz) : NAN;
    free(T), free(S), free(w), free(za), free(ha), free(ht);
}

static void mat3_mul(const double *A, const double *B, double *C) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) C[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
}

static void eval_clip(const plx_fu_head *h, const plx_fu_synth_opts *o, const clip_run *c, synth_eval *ev) {
    memset(ev, 0, sizeof *ev);
    int P = c->nd.P, F = c->F;
    double *XY = PLX_NEW(double, (size_t)2 * P), *err = PLX_NEW(double, P), *XYp = PLX_NEW(double, (size_t)2 * P);
    true_correspondences(h, o, c, c->x_pop, XYp, err);
    double s2 = 0;
    int m = 0;
    for (int n = 0; n < P; n++)
        if (c->sh.used[n] && c->nd.core[n] && isfinite(err[n])) s2 += err[n] * err[n], m++;
    ev->track_pop = sqrt(s2 / m);
    true_correspondences(h, o, c, c->x, XY, err);
    s2 = 0, m = 0;
    for (int n = 0; n < P; n++)
        if (c->sh.used[n] && c->nd.core[n] && isfinite(err[n])) s2 += err[n] * err[n], m++;
    ev->track_pers = sqrt(s2 / m);
    double Rq[9];
    eval_depth(h, &c->nd, &c->sh, XY, ev, Rq);
    /* pose: truth = s Rq rec  =>  R_true ≈ R_rec Rqᵀ */
    double Rt[9], e[3] = {0};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) Rt[i * 3 + j] = Rq[j * 3 + i];
    double eg = 0;
    for (int f = 0; f < F; f++) {
        plx_fu_pose p = plx_fu_synth_pose(o, f);
        double Rr[9], yv, pv, rv;
        mat3_mul(c->sh.R + f * 9, Rt, Rr);
        plx_fu_angles(Rr, &yv, &pv, &rv);
        e[0] += (yv - p.yaw) * (yv - p.yaw) / F, e[1] += (pv - p.pitch) * (pv - p.pitch) / F, e[2] += (rv - p.roll) * (rv - p.roll) / F;
        eg += (c->sh.yaw[f] - p.yaw) * (c->sh.yaw[f] - p.yaw) / F;
    }
    ev->yaw_rms = sqrt(e[0]) / DEG, ev->pitch_rms = sqrt(e[1]) / DEG, ev->roll_rms = sqrt(e[2]) / DEG;
    ev->yaw_rms_gauge = sqrt(eg) / DEG;
    /* localisation: the window centre vs the head origin of frame 0 */
    plx_fu_pose p0 = plx_fu_synth_pose(o, 0);
    double gx, gy;
    plx_sim_apply(c->loc.T, PLX_CANON_CX, PLX_CANON_CY, &gx, &gy);
    ev->loc_err = hypot(gx - p0.x, gy - p0.y);
    ev->loc_scale = plx_sim_scale(c->loc.T) / p0.s;
    free(XY), free(err), free(XYp);
}

/* ---- synthetic mode ------------------------------------------------------------------------------ */
typedef struct {
    context  *cx;
    plx_img   tex[3];     /* frontalised RGB textures (Ω+, margin 16) */
    int       tex_img[3]; /* their global LFW indices */
    const char *out;
} synth_env;

/* Three near-frontal, trusted LFW photos of different people, frontalised by their population fit. */
static void pick_textures(synth_env *se) {
    context *cx = se->cx;
    char path[4096];
    plx_tensor pose, q, sim, u, rgb;
    snprintf(path, sizeof path, "%s/pose.plxt", cx->geom);
    if (plx_tensor_load(path, &pose) != PLX_OK) plx_die("cannot read %s", path);
    snprintf(path, sizeof path, "%s/quality.plxt", cx->flow);
    if (plx_tensor_load(path, &q) != PLX_OK) plx_die("cannot read %s", path);
    snprintf(path, sizeof path, "%s/sim.plxt", cx->flow);
    if (plx_tensor_load(path, &sim) != PLX_OK) plx_die("cannot read %s", path);
    snprintf(path, sizeof path, "%s/u.plxt", cx->flow);
    if (plx_tensor_map(path, &u) != PLX_OK) plx_die("cannot read %s", path);
    snprintf(path, sizeof path, "%s/rgb.plxt", cx->pack);
    if (plx_tensor_map(path, &rgb) != PLX_OK) plx_die("cannot read %s", path);
    int N = (int)pose.dim[0], k = 0;
    const float *pv = pose.data, *qv = q.data, *sv = sim.data;
    /* fixed, spread-out picks so the textures are different people (index order is by name) */
    int start[3] = {N / 7, N / 2, (5 * N) / 6};
    plx_lattice L8 = plx_lattice_make(PLX_FL_STEP);
    for (int j = 0; j < 3; j++) {
        for (int i = start[j]; i < N; i++) {
            const float *p = pv + (size_t)i * 9;
            if (((int)qv[(size_t)i * 7 + 6] & 27) || fabs(p[2]) > 2.5 * DEG || fabs(p[3]) > 3 * DEG) continue;
            plx_img im = plx_img_from_u8((const uint8_t *)rgb.data + (size_t)i * 250 * 250 * 3, 250, 250, 3, 1.0f / 255);
            plx_sim T = {sv[i * 4], sv[i * 4 + 1], sv[i * 4 + 2], sv[i * 4 + 3]};
            se->tex[k] = plx_canon_new(3, PLX_CANON_MARGIN);
            plx_fl_render_gray(&im, T, &L8, (const float *)u.data + (size_t)i * 714, PLX_CANON_MARGIN, PLX_INTERP_BILINEAR,
                               &se->tex[k], NULL);
            plx_img_free(&im);
            se->tex_img[k++] = i;
            break;
        }
    }
    if (k < 3) plx_die("could not find three frontal textures");
    plx_tensor_free(&pose), plx_tensor_free(&q), plx_tensor_free(&sim), plx_tensor_free(&u), plx_tensor_free(&rgb);
}

static clip_run synth_clip(synth_env *se, const plx_fu_head *h, int tex, const plx_fu_synth_opts *o, synth_eval *ev,
                           const char *diag_name) {
    plx_img *frames = PLX_NEW(plx_img, o->F);
    double *t_ms = PLX_NEW(double, o->F);
    plx_img bg = plx_fu_synth_background(o, 3);
    for (int f = 0; f < o->F; f++) {
        frames[f] = plx_fu_synth_frame(h, &se->tex[tex], &bg, o, f);
        t_ms[f] = f * 1000.0 / 30;
    }
    plx_img_free(&bg);
    clip_run c;
    run_clip(se->cx, frames, o->F, t_ms, &c);
    eval_clip(h, o, &c, ev);
    if (diag_name) {
        double *ang = PLX_NEW(double, (size_t)o->F * 3);
        for (int f = 0; f < o->F; f++) {
            plx_fu_pose p = plx_fu_synth_pose(o, f);
            ang[f * 3] = p.yaw, ang[f * 3 + 1] = p.pitch, ang[f * 3 + 2] = p.roll;
        }
        plx_img d = clip_diag(&c, ang);
        save_png(se->out, diag_name, &d);
        plx_img_free(&d);
        free(ang);
        /* the first frame as the pipeline saw it */
        char name[256];
        snprintf(name, sizeof name, "%.*s_frame0.png", (int)(strlen(diag_name) - 4), diag_name);
        save_png(se->out, name, &frames[0]);
    }
    for (int f = 0; f < o->F; f++) plx_img_free(&frames[f]);
    free(frames), free(t_ms);
    return c;
}

static void eval_header(void) {
    say("  %-22s %6s %6s %6s %6s %6s %6s %6s %6s %6s %6s %6s %6s %6s %5s\n", "clip", "corr", "rms", "hpcorr", "hprms",
        "nose", "true", "scale", "yaw", "pitch", "roll", "reproj", "trkpop", "trkper", "zsc");
}

static void eval_line(const char *label, const synth_eval *e, const clip_run *c) {
    say("  %-22s %6.3f %6.2f %6.3f %6.2f %6.1f %6.1f %6.3f %6.2f %6.2f %6.2f %6.3f %6.3f %6.3f %5.2f%s\n", label,
        e->depth_corr, e->depth_rms, e->hp_corr, e->hp_rms, e->nose_rec, e->nose_true, e->scale_ratio, e->yaw_rms,
        e->pitch_rms, e->roll_rms, c->sh.rms, e->track_pop, e->track_pers, e->zscore_rms,
        c->sh.degenerate ? "  [degenerate: prior scale]" : "");
}

static int run_synth_mode(const plx_args *a) {
    synth_env se = {.out = plx_arg_str(a, "out", "runs/main/fuse/synth")};
    context cx;
    load_context(a, &cx);
    se.cx = &cx;
    open_summary(se.out);
    pick_textures(&se);
    bool quick = plx_arg_flag(a, "quick");
    say("plx_fuse --synth: textures = frontalised LFW photos %d, %d, %d; personal lattice %d px, lambda %.3g, %d rounds\n",
        se.tex_img[0], se.tex_img[1], se.tex_img[2], cx.topt.step, cx.topt.lambda, cx.topt.rounds);
    for (int k = 0; k < 3; k++) {
        char name[64];
        snprintf(name, sizeof name, "texture_%d.png", k);
        plx_img t = plx_canon_crop(&se.tex[k], PLX_CANON_MARGIN);
        save_png(se.out, name, &t);
        plx_img_free(&t);
    }
    plx_fu_head heads[3] = {plx_fu_head_preset(0), plx_fu_head_preset(1), plx_fu_head_preset(2)};
    say("columns: depth corr / rms (px) over the core after similarity alignment to truth; the same for the\n"
        "high-pass part (best quadratic removed: nose, eyes, lips); nose protrusion recovered / true (px);\n"
        "depth scale (regression of recovered on true depth); pose rms (deg, truth frame); rigid reprojection rms;\n"
        "track error vs truth for population / personal tracks (px); rms z-score of the depth error\n\n");

    /* base clip, with diagnostics */
    plx_fu_synth_opts o = plx_fu_synth_defaults();
    o.shade = plx_arg_num(a, "shade", o.shade);
    o.noise = plx_arg_num(a, "noise", o.noise);
    o.F = (int)plx_arg_int(a, "frames", o.F);
    synth_eval ev;
    say("BASE: 40 frames, yaw +-30, nod +-12, roll +-3, 250 x 250, head 0, texture 0\n");
    eval_header();
    clip_run c = synth_clip(&se, &heads[0], 0, &o, &ev, "base_diag.png");
    eval_line("base", &ev, &c);
    report_clip(&c);
    free_clip(&c);
    if (quick) goto done;

    say("\nERROR VS NUMBER OF FRAMES (yaw +-30, nod +-12, head 0, texture 0)\n");
    eval_header();
    int nf[6] = {3, 5, 10, 20, 40, 80};
    for (int k = 0; k < 6; k++) {
        o = plx_fu_synth_defaults();
        o.F = nf[k];
        char lab[64];
        snprintf(lab, sizeof lab, "F = %d", nf[k]);
        c = synth_clip(&se, &heads[0], 0, &o, &ev, NULL);
        eval_line(lab, &ev, &c);
        free_clip(&c);
    }
    say("\nERROR VS YAW RANGE (40 frames, nod = 0.4 x yaw range, head 0, texture 0)\n");
    eval_header();
    double yr[4] = {5, 10, 20, 30};
    for (int k = 0; k < 4; k++) {
        o = plx_fu_synth_defaults();
        o.yaw_deg = yr[k], o.pitch_deg = 0.4 * yr[k];
        char lab[64];
        snprintf(lab, sizeof lab, "yaw +-%.0f nod +-%.0f", yr[k], 0.4 * yr[k]);
        c = synth_clip(&se, &heads[0], 0, &o, &ev, NULL);
        eval_line(lab, &ev, &c);
        free_clip(&c);
    }
    say("\nDEGENERATE MOTION: yaw +-30 with nod amplitude varied, roll 0 (40 frames, head 0, texture 0);\n"
        "metric-only scale = depth scale of the metric upgrade without the prior fallback\n");
    say("  %-16s %8s %8s %8s %8s %8s %8s  %s\n", "nod", "spread2", "gap", "metric", "boot_sd", "final", "hpcorr",
        "decision");
    double nods[6] = {0, 1, 2, 4, 8, 12};
    for (int k = 0; k < 6; k++) {
        o = plx_fu_synth_defaults();
        o.pitch_deg = nods[k], o.roll_deg = 0;
        c = synth_clip(&se, &heads[0], 0, &o, &ev, k == 0 ? "yaw_only_diag.png" : NULL);
        /* the same tracks through the metric upgrade alone */
        plx_fu_factor_opts fo = cx.fopt;
        fo.prior_scale = 0;
        plx_fu_tracks tp = tracks_of(&c, c.x);
        plx_fu_shape sm;
        plx_fu_factorise(&tp, &fo, &sm);
        synth_eval em;
        double *XY = PLX_NEW(double, (size_t)2 * c.nd.P), *err = PLX_NEW(double, c.nd.P), Rq[9];
        true_correspondences(&heads[0], &o, &c, c.x, XY, err);
        eval_depth(&heads[0], &c.nd, &sm, XY, &em, Rq);
        say("  nod +-%-10.0f %8.2f %8.1e %8.3f %8.3f %8.3f %8.3f  %s\n", nods[k], c.sh.spread_deg[1], c.sh.gap,
            em.scale_ratio, sm.scale_sd, ev.scale_ratio, ev.hp_corr, c.sh.degenerate ? "degenerate -> prior scale" : "metric");
        free(XY), free(err);
        plx_fu_shape_free(&sm);
        free_clip(&c);
    }

    say("\nSHAPES x TEXTURES (40 frames, yaw +-30, nod +-12): high-pass depth correlation of the reconstruction\n"
        "with each head's truth (rows: true head of the clip; columns: head compared against)\n");
    double conf[3][3] = {{0}};
    int ncf[3] = {0};
    for (int hd = 0; hd < 3; hd++)
        for (int tx = 0; tx < 3; tx++) {
            o = plx_fu_synth_defaults();
            o.seed = 10 + 3 * hd + tx;
            c = synth_clip(&se, &heads[hd], tx, &o, &ev, NULL);
            char lab[64];
            snprintf(lab, sizeof lab, "head %d texture %d", hd, tx);
            if (hd == 0 && tx == 0) eval_header();
            eval_line(lab, &ev, &c);
            double *XY = PLX_NEW(double, (size_t)2 * c.nd.P), *err = PLX_NEW(double, c.nd.P), Rq[9];
            true_correspondences(&heads[hd], &o, &c, c.x, XY, err);
            for (int j = 0; j < 3; j++) {
                synth_eval ej;
                eval_depth(&heads[j], &c.nd, &c.sh, XY, &ej, Rq);
                conf[hd][j] += ej.hp_corr / 3;
            }
            ncf[hd]++;
            free(XY), free(err);
            free_clip(&c);
        }
    say("  %-8s %8s %8s %8s\n", "", "head 0", "head 1", "head 2");
    for (int hd = 0; hd < 3; hd++) say("  head %d   %8.3f %8.3f %8.3f\n", hd, conf[hd][0], conf[hd][1], conf[hd][2]);
    {
        /* the truths themselves: how different are the three heads' high-pass depth patterns? */
        nodes nd;
        plx_lattice L = plx_lattice_make(cx.topt.step);
        nodes_make(&cx, &L, &nd);
        double *z[3], *hp[3], *w = PLX_NEW(double, nd.P);
        for (int j = 0; j < 3; j++) {
            z[j] = PLX_NEW(double, nd.P), hp[j] = PLX_NEW(double, nd.P);
            for (int n = 0; n < nd.P; n++) z[j][n] = plx_fu_head_depth(&heads[j], nd.X0[n], nd.Y0[n]);
        }
        for (int n = 0; n < nd.P; n++) w[n] = nd.core[n];
        for (int j = 0; j < 3; j++) highpass(nd.P, nd.X0, nd.Y0, z[j], w, hp[j]);
        say("  truth vs truth (same measure, canonical nodes):\n");
        for (int i = 0; i < 3; i++) {
            say("  head %d  ", i);
            for (int j = 0; j < 3; j++) {
                double cr;
                compare(nd.P, hp[i], hp[j], w, &cr, NULL);
                say(" %8.3f", cr);
            }
            say("   nose %.1f px\n", protrusion(&nd, z[i]));
        }
        for (int j = 0; j < 3; j++) free(z[j]), free(hp[j]);
        free(w);
        nodes_free(&nd);
    }

    say("\nLARGER FRAMES WITH CLUTTER (480 x 360; localisation, then the same pipeline)\n");
    eval_header();
    for (int k = 0; k < 2; k++) {
        o = plx_fu_synth_defaults();
        o.frame_w = 480, o.frame_h = 360, o.clutter = true, o.seed = 50 + k;
        o.face_scale = k ? 0.85 : 1.4, o.face_x = k ? 140 : 300, o.face_y = k ? 150 : 190;
        char lab[64], dn[64];
        snprintf(lab, sizeof lab, "clutter s %.2f", o.face_scale);
        snprintf(dn, sizeof dn, "clutter%d_diag.png", k);
        c = synth_clip(&se, &heads[k], k, &o, &ev, dn);
        eval_line(lab, &ev, &c);
        say("    localisation: centre error %.2f px, scale ratio %.3f, score %.3f vs runner-up %.3f\n", ev.loc_err,
            ev.loc_scale, c.loc.score, c.loc.runner_score);
        free_clip(&c);
    }
done:
    say("outputs in %s\n", se.out);
    for (int k = 0; k < 3; k++) plx_img_free(&se.tex[k]);
    free_context(&cx);
    fclose(g_sum);
    return 0;
}

int main(int argc, char **argv) {
    plx_args a = {argc, argv};
    if (plx_arg_str(&a, "clip", NULL)) return run_clip_mode(&a);
    if (plx_arg_flag(&a, "synth")) return run_synth_mode(&a);
    fprintf(stderr, "usage: plx_fuse --clip <clip.plxv | dir> --out <dir> | --synth | --lfw-identity | --locate-test\n");
    return 2;
}
