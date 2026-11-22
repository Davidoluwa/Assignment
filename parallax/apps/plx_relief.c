// plx-deps: geom align render viz feat data la ops img core
/* plx_relief: the depth-relief stage (geom/relief.h). The population-shared, full-resolution
 * layer w on top of the emerge stage's smooth depth pattern v, estimated by regressing every
 * canonical pixel's residual horizontal misalignment on the photos' known yaw parallax.
 *
 *   plx_relief --pack runs/main/pack --flow runs/main/flow --geom runs/main/geom --out runs/main/relief
 *
 * Inputs: <pack>/gray.plxt u8 [N, 250, 250]; <flow>/{u, sim, mean_face}.plxt and <flow>/model/
 * (feature parameters and the final flow template); <geom>/pose.plxt (s, robust weight; columns
 * in apps/plx_emerge.c), <geom>/zbar.plxt (κ v on the lattice) and <geom>/kappa.txt.
 * Photos with robust weight >= 0.5 enter (emerge's trusted set). Options:
 *   --sched L:σ:iters,...   schedule (pyramid level, window px, passes); default relief.h's
 *   --lambda, --eta         regularisation and damping
 *   --n K                   every (N/K)-th photo only, for quick experiments
 *   --no-controls --no-refine --no-synth   skip parts
 *   --synth-n, --synth-only the synthetic known-relief check (acceptance 1)
 *
 * Writes into <out>: w.plxt, vhr.plxt, zbar_hr.plxt, support.plxt, zscore.plxt (f32 [160, 128];
 * vhr / zbar_hr NaN outside the core), s_refined.plxt (f32 [N], when refined), mean_face.plxt,
 * model/ (the sharpened template in flow-model layout: plx_fl_model_load reads it), summary.txt
 * and diag/<n>_<name>.png. */
#include "align/canon.h"
#include "align/congeal.h"
#include "align/flow.h"
#include "align/lattice.h"
#include "core/core.h"
#include "data/lfw.h"
#include "geom/relief.h"
#include "img/image.h"
#include "img/imgio.h"
#include "ops/ops.h"
#include "render/render.h"
#include "viz/viz.h"

#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#define DEG (M_PI / 180.0)
enum { NLAT = PLX_RL_LAT_NODES, DOF = 2 * NLAT, PCOLS = 9, W = PLX_CANON_W, H = PLX_CANON_H };
static const float WHITE[3] = {1, 1, 1}, GREY[3] = {0.6f, 0.6f, 0.6f}, CYAN[3] = {0.3f, 0.85f, 1},
                   YELLOW[3] = {1, 0.85f, 0.2f};

/* ---- small utilities ------------------------------------------------------------------------- */
static FILE *g_sum; /* summary.txt; every report line also goes to stdout */
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

static void save_map(const char *dir, const char *name, const plx_img *im) {
    char path[4096];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    if (plx_tensor_save(path, PLX_F32, 2, (int64_t[]){im->h, im->w}, im->px) != PLX_OK) plx_die("cannot write %s", path);
}

static void save_png(const char *dir, const char *name, const plx_img *im) {
    char path[4096];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    if (plx_img_save(path, im) != PLX_OK) plx_die("cannot write %s", path);
}

static plx_img load_map(const char *path, int w, int h) {
    plx_tensor t;
    if (plx_tensor_load(path, &t) != PLX_OK || t.dtype != PLX_F32 || plx_tensor_count(&t) != (int64_t)w * h)
        plx_die("%s: want f32 [%d, %d]", path, h, w);
    plx_img im = plx_img_new(w, h, 1);
    memcpy(im.px, t.data, sizeof(float) * w * h);
    plx_tensor_free(&t);
    return im;
}

static double read_kappa(const char *path) {
    FILE *f = fopen(path, "r");
    double k = NAN;
    if (!f || fscanf(f, "kappa %lf", &k) != 1) plx_die("cannot read kappa from %s", path);
    fclose(f);
    return k;
}

/* "L:sigma:iters,..." */
static void parse_sched(const char *s, plx_rl_params *p) {
    p->n_stages = 0;
    while (*s) {
        plx_rl_stage st;
        int used = 0;
        if (p->n_stages >= PLX_RL_MAX_STAGES || sscanf(s, "%d:%f:%d%n", &st.level, &st.sigma, &st.iters, &used) != 3)
            plx_die("--sched: want L:sigma:iters[,...]");
        p->stages[p->n_stages++] = st;
        s += used;
        if (*s == ',') s++;
    }
}

static void sched_str(const plx_rl_params *p, char *buf, size_t n) {
    size_t k = 0;
    buf[0] = 0;
    for (int s = 0; s < p->n_stages && k < n; s++)
        k += (size_t)snprintf(buf + k, n - k, "%s%d:%.2g:%d", s ? "," : "", p->stages[s].level, p->stages[s].sigma,
                              p->stages[s].iters);
}

static double rms_core(const plx_img *a, plx_core_shape cs) {
    double s = 0;
    int n = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            if (plx_rl_in_core(cs, x, y)) s += (double)PLX_PX(a, x, y, 0) * PLX_PX(a, x, y, 0), n++;
    return sqrt(s / n);
}

/* a · scale + b · bscale, NaN outside the core when masked. */
static plx_img combine(const plx_img *a, double as, const plx_img *b, double bs, plx_core_shape cs, bool mask) {
    plx_img o = plx_img_new(W, H, 1);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            double v = as * PLX_PX(a, x, y, 0) + (b ? bs * PLX_PX(b, x, y, 0) : 0);
            PLX_PX(&o, x, y, 0) = mask && !plx_rl_in_core(cs, x, y) ? NAN : (float)v;
        }
    return o;
}

/* ---- the shape numbers ------------------------------------------------------------------------ */
/* Nose protrusion of a dense depth map (px): max over the face-centre box (|X0| <= 0.3 ax,
 * |Y0 - yc| <= 0.4 ay, emerge's definition) minus the mean over the cheeks (|X0| in [20, 36],
 * Y0 in [-4, 20]). */
static double protrusion(const plx_img *z, plx_core_shape cs, int *tip_y) {
    double nose = -INFINITY, c = 0;
    int nc = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            double X = x - PLX_CANON_CX, Y = y - PLX_CANON_CY, v = PLX_PX(z, x, y, 0);
            if (fabs(X) <= 0.3 * cs.ax && fabs(Y - cs.yc) <= 0.4 * cs.ay && v > nose) {
                nose = v;
                if (tip_y) *tip_y = y;
            }
            if (fabs(X) >= 20 && fabs(X) <= 36 && Y >= -4 && Y <= 20) c += v, nc++;
        }
    return nose - c / nc;
}

/* Eye row and inter-eye distance from a mean face (darkest 7x7 blob on the right half, emerge's
 * search window); mouth row = darkest midline band below the nose tip. */
static void landmarks(const plx_img *face, int tip_y, int *eye_y, double *eye_dist, int *mouth_y) {
    double best = INFINITY;
    int bx = 0;
    for (int y = 52; y <= 72; y++)
        for (int x = 66; x <= 104; x++) {
            double s = 0;
            for (int dy = -3; dy <= 3; dy++)
                for (int dx = -3; dx <= 3; dx++) s += PLX_PX(face, x + dx, y + dy, 0);
            if (s < best) best = s, bx = x, *eye_y = y;
        }
    *eye_dist = 2 * (bx - PLX_CANON_CX);
    best = INFINITY;
    for (int y = tip_y + 12; y <= PLX_MIN(tip_y + 40, H - 4); y++) {
        double s = 0;
        for (int dy = -1; dy <= 1; dy++)
            for (int x = 54; x <= 73; x++) s += PLX_PX(face, x, y + dy, 0);
        if (s < best) best = s, *mouth_y = y;
    }
}

/* ---- drawing ------------------------------------------------------------------------------------ */
static plx_img canvas(int w, int h) {
    plx_img im = plx_img_new(w, h, 3);
    for (size_t k = 0; k < (size_t)3 * w * h; k++) im.px[k] = 0.1f;
    return im;
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
        plx_draw_line(&im, (float)f->L, px_y(f, y), (float)(f->w - f->R), px_y(f, y), 1, GREY, 0.15f);
    }
    plx_draw_text(&im, f->L, 6, title, 1, WHITE);
    plx_draw_text(&im, f->w / 2 - 40, f->h - 14, xl, 1, WHITE);
    plx_draw_text(&im, 4, f->T - 14, yl, 1, WHITE);
    return im;
}

/* Curves y[c][k] over x[k] (NaN gaps), with a legend. */
static plx_img line_plot(const char *title, const char *xl, const char *yl, const double *x, int n, const double *const *y,
                         const char *const *names, const float *const *cols, int nc, const double *marks, int nm) {
    double lo = INFINITY, hi = -INFINITY;
    for (int c = 0; c < nc; c++)
        for (int k = 0; k < n; k++)
            if (isfinite(y[c][k])) lo = fmin(lo, y[c][k]), hi = fmax(hi, y[c][k]);
    double pad = 0.08 * (hi - lo + 1e-9);
    plot_frame f = {.x0 = x[0], .x1 = x[n - 1], .y0 = lo - pad, .y1 = hi + pad, .L = 60, .R = 20, .T = 36, .B = 40, .w = 640, .h = 300};
    plx_img im = plot_axes(&f, title, xl, yl);
    for (int m = 0; m < nm; m++) plx_draw_line(&im, px_x(&f, marks[m]), (float)f.T, px_x(&f, marks[m]), (float)(f.h - f.B), 1, GREY, 0.5f);
    for (int c = 0; c < nc; c++) {
        for (int k = 1; k < n; k++)
            if (isfinite(y[c][k - 1]) && isfinite(y[c][k]))
                plx_draw_line(&im, px_x(&f, x[k - 1]), px_y(&f, y[c][k - 1]), px_x(&f, x[k]), px_y(&f, y[c][k]), 2, cols[c], 1);
        plx_draw_text(&im, f.w - f.R - 230, f.T + 4 + 12 * c, names[c], 1, cols[c]);
    }
    return im;
}

static plx_img stack(plx_img *ims, int n, int cols) {
    plx_montage_opts mo = plx_montage_defaults();
    mo.cols = cols;
    return plx_viz_montage(ims, n, NULL, &mo);
}

static plx_img cmap_tile(const plx_img *a, float lo, float hi, plx_cmap m) {
    plx_img up = plx_viz_upscale(a, 2, PLX_UP_NEAREST);
    plx_img t = plx_viz_colormap(up.px, up.w, up.h, lo, hi, m);
    plx_img_free(&up);
    return t;
}

/* plx_render_views of a dense Ω depth map (px; NaN outside the core) at 2x, textured by face. */
static plx_img render_set(const plx_img *z, const plx_img *face) {
    const int f = 2;
    plx_img zd = plx_viz_upscale(z, f, PLX_UP_BILINEAR);
    for (size_t k = 0; k < (size_t)zd.w * zd.h; k++) zd.px[k] *= f; /* heights in enlarged-grid pixels */
    /* NaN does not survive bilinear blending at the border: mask from the nearest source pixel */
    for (int y = 0; y < zd.h; y++)
        for (int x = 0; x < zd.w; x++)
            if (!isfinite(PLX_PX(z, x / f, y / f, 0))) PLX_PX(&zd, x, y, 0) = NAN;
    plx_img tex = plx_viz_upscale(face, f, PLX_UP_BILINEAR);
    plx_pose poses[9];
    for (int k = 0; k < 9; k++) poses[k] = (plx_pose){(-40 + 10 * k) * DEG, 0, 0};
    plx_img out = plx_render_views(&zd, NULL, &tex, poses, 9);
    plx_img_free(&zd), plx_img_free(&tex);
    return out;
}

static plx_img titled(const plx_img *im, const char *title) {
    plx_img rgb = plx_viz_to_rgb(im), out = canvas(rgb.w, rgb.h + 18);
    plx_draw_text(&out, 6, 4, title, 1, WHITE);
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < rgb.h; y++)
            for (int x = 0; x < rgb.w; x++) PLX_PX(&out, x, y + 18, c) = PLX_PX(&rgb, x, y, c);
    plx_img_free(&rgb);
    return out;
}

/* ---- data ------------------------------------------------------------------------------------- */
typedef struct {
    int           n;
    plx_cg_source src;
    plx_sim      *T;
    float        *u, *s;
    uint8_t      *use;
    int          *ident;  /* [n] LFW identity, NULL without the index */
    float         v_lat[NLAT];
    double        kappa;
    plx_img       face0;  /* flow's mean face (Ω) */
} inputs;

static plx_rl_data as_data(const inputs *in, const float *s, const uint8_t *use, const float *shift) {
    return (plx_rl_data){.src = &in->src, .T = in->T, .u = in->u, .s = s ? s : in->s, .s_shift = shift,
                         .use = use ? use : in->use, .v_lat = in->v_lat};
}

/* ---- synthetic known relief (acceptance 1) -------------------------------------------------- */
typedef struct { double corr, corr_hp, rms, rms_true, rms_w; } synth_score;

static synth_score synth_eval(const plx_rl_synth *s, const plx_rl_result *r, const plx_rl_params *p, plx_img *wt_out) {
    synth_score sc;
    plx_img v = plx_rl_v_dense(s->v_lat), wt = plx_img_clone(&s->w_true), wg = plx_img_clone(&r->w);
    plx_rl_gauge(&wt, &v, &r->support, p->core, NULL);
    plx_rl_gauge(&wg, &v, &r->support, p->core, NULL);
    plx_rl_compare(&wg, &wt, p->core, 4.0f, &sc.corr, &sc.corr_hp, &sc.rms);
    sc.rms_true = rms_core(&wt, p->core);
    sc.rms_w = rms_core(&wg, p->core);
    if (wt_out) *wt_out = wt;
    else plx_img_free(&wt);
    plx_img_free(&v), plx_img_free(&wg);
    return sc;
}

static void run_synth(const plx_args *a, const plx_rl_params *p, const inputs *in, const char *diag) {
    int n = (int)plx_arg_int(a, "synth-n", 1200), ntex = 0;
    say("\n[1] SYNTHETIC KNOWN RELIEF: %d photos rendered by the z-buffer renderer from one surface (plx_synth_depth:\n"
        "    dome + nose + eye sockets + brow, lips, chin), yaw N(0, 12 deg) clipped at 35, each with a different\n"
        "    texture; the estimator gets the dome only (half-ellipsoid on the lattice), u with a smooth 0.5 px\n"
        "    error, and s = sin(yaw) (+ noise); same schedule. Truth = surface - dome, in the estimator's gauge.\n", n);
    /* frontalised real faces (through the photos' own T, u) as the textures */
    plx_img *tex = PLX_NEW(plx_img, n);
    plx_lattice L = plx_lattice_make(PLX_FL_STEP);
    for (int i = 0; i < in->n && ntex < n; i += PLX_MAX(1, in->n / n)) {
        if (!in->use[i]) continue;
        plx_img g = plx_cg_source_image(&in->src, i);
        tex[ntex] = plx_canon_new(1, PLX_CANON_MARGIN);
        plx_fl_render_gray(&g, in->T[i], &L, in->u + (size_t)i * DOF, PLX_CANON_MARGIN, PLX_INTERP_BILINEAR, &tex[ntex],
                           NULL);
        plx_img_free(&g);
        ntex++;
    }
    say("    %-34s %8s %8s %10s %10s %10s\n", "setting", "corr", "corr_hp", "rms err", "rms w", "rms true");
    struct { const char *name; double s_noise; bool random_tex, permute; } runs[] = {
        {"face textures, exact s", 0, false, false},
        {"face textures, s noise 10%", 0.1, false, false},
        {"face textures, s noise 20%", 0.2, false, false},
        {"face textures, s permuted", 0, false, true},
        {"random textures, exact s", 0, true, false},
    };
    plx_img tiles[8], wt_keep = {0}, w_keep[2] = {{0}}, v_keep = {0};
    int nr = (int)(sizeof runs / sizeof runs[0]);
    for (int r = 0; r < nr; r++) {
        plx_rl_synth_opts so = plx_rl_synth_defaults();
        so.n = n;
        so.s_noise = runs[r].s_noise;
        so.u_noise_px = 0.5;
        so.seed = 11;
        if (!runs[r].random_tex) so.tex = tex, so.n_tex = ntex;
        plx_rl_synth s;
        plx_rl_synth_make(&so, &s);
        if (runs[r].permute) {
            plx_rng rng;
            plx_rng_seed(&rng, 41);
            int *perm = PLX_NEW(int, n);
            float *sp = PLX_NEW(float, n);
            for (int i = 0; i < n; i++) perm[i] = i;
            plx_rng_shuffle(&rng, perm, n);
            for (int i = 0; i < n; i++) sp[i] = s.s[perm[i]];
            memcpy(s.s, sp, sizeof(float) * n);
            free(perm), free(sp);
        }
        plx_rl_data d = {.src = &s.src, .T = s.T, .u = s.u, .s = s.s, .v_lat = s.v_lat};
        plx_rl_result res;
        plx_rl_estimate(&d, p, NULL, NULL, false, &res);
        plx_img wt;
        synth_score sc = synth_eval(&s, &res, p, &wt);
        say("    %-34s %8.3f %8.3f %10.3f %10.3f %10.3f   (px; core ellipse)\n", runs[r].name, sc.corr, sc.corr_hp, sc.rms,
            sc.rms_w, sc.rms_true);
        if (r == 0) wt_keep = wt, w_keep[0] = plx_img_clone(&res.w), v_keep = plx_rl_v_dense(s.v_lat);
        else plx_img_free(&wt);
        if (r == 2) w_keep[1] = plx_img_clone(&res.w);
        plx_rl_result_free(&res);
        plx_rl_synth_free(&s);
    }
    /* picture: truth, recovered (exact s, 20% noise) as maps; midline profiles of dome + w */
    float m = 0;
    for (int k = 0; k < W * H; k++)
        if (plx_rl_in_core(p->core, k % W, k / W)) m = fmaxf(m, fabsf(wt_keep.px[k]));
    plx_img ma = combine(&wt_keep, 1, NULL, 0, p->core, true), mb = combine(&w_keep[0], 1, NULL, 0, p->core, true),
            mc = combine(&w_keep[1], 1, NULL, 0, p->core, true);
    plx_img t0 = cmap_tile(&ma, -m, m, PLX_CMAP_DIV), t1 = cmap_tile(&mb, -m, m, PLX_CMAP_DIV),
            t2 = cmap_tile(&mc, -m, m, PLX_CMAP_DIV);
    tiles[0] = titled(&t0, "true relief (gauged)");
    tiles[1] = titled(&t1, "recovered w, exact s");
    tiles[2] = titled(&t2, "recovered w, s noise 20%");
    plx_img maps = stack(tiles, 3, 3);
    double ys[H], *prof[3];
    for (int k = 0; k < 3; k++) prof[k] = PLX_NEW(double, H);
    for (int y = 0; y < H; y++) {
        ys[y] = y - PLX_CANON_CY;
        bool in_c = plx_rl_in_core(p->core, 63, y);
        double vv = PLX_PX(&v_keep, 63, y, 0);
        prof[0][y] = in_c ? vv : NAN;
        prof[1][y] = in_c ? vv + PLX_PX(&wt_keep, 63, y, 0) : NAN;
        prof[2][y] = in_c ? vv + PLX_PX(&w_keep[0], 63, y, 0) : NAN;
    }
    const char *names[3] = {"dome (given)", "true surface (gauged)", "dome + recovered w"};
    const float *cols[3] = {GREY, CYAN, YELLOW};
    plx_img pl = line_plot("synthetic midline profile (depth px vs Y0)", "Y0 (px, down)", "depth", ys, H,
                           (const double *const *)prof, names, cols, 3, NULL, 0);
    plx_img both[2] = {maps, pl};
    plx_img all = stack(both, 2, 1);
    save_png(diag, "0_synthetic.png", &all);
    for (int k = 0; k < 3; k++) free(prof[k]), plx_img_free(&tiles[k]);
    plx_img_free(&t0), plx_img_free(&t1), plx_img_free(&t2), plx_img_free(&ma), plx_img_free(&mb), plx_img_free(&mc);
    plx_img_free(&maps), plx_img_free(&pl), plx_img_free(&all);
    plx_img_free(&wt_keep), plx_img_free(&w_keep[0]), plx_img_free(&w_keep[1]), plx_img_free(&v_keep);
    for (int k = 0; k < ntex; k++) plx_img_free(&tex[k]);
    free(tex);
}

/* ---- real data ------------------------------------------------------------------------------------ */
static void load_inputs(const plx_args *a, inputs *in, plx_fl_model *model, plx_tensor *gray, uint8_t **own_px) {
    const char *pack = plx_arg_str(a, "pack", "runs/main/pack"), *flow = plx_arg_str(a, "flow", "runs/main/flow"),
               *geom = plx_arg_str(a, "geom", "runs/main/geom");
    char path[4096];
    snprintf(path, sizeof path, "%s/model", flow);
    if (plx_fl_model_load(path, model) != PLX_OK) plx_die("cannot load the flow model %s", path);
    snprintf(path, sizeof path, "%s/gray.plxt", pack);
    if (plx_tensor_map(path, gray) != PLX_OK || gray->dtype != PLX_U8 || gray->ndim != 3) plx_die("cannot read %s", path);
    int N = (int)gray->dim[0], side = (int)gray->dim[1];
    plx_tensor u, sim, pose;
    snprintf(path, sizeof path, "%s/u.plxt", flow);
    if (plx_tensor_map(path, &u) != PLX_OK || u.dtype != PLX_F32 || plx_tensor_count(&u) != (int64_t)N * DOF)
        plx_die("%s: want f32 [%d, 2, 21, 17]", path, N);
    snprintf(path, sizeof path, "%s/sim.plxt", flow);
    if (plx_tensor_load(path, &sim) != PLX_OK || sim.dtype != PLX_F32 || plx_tensor_count(&sim) != (int64_t)N * 4)
        plx_die("%s: want f32 [%d, 4]", path, N);
    snprintf(path, sizeof path, "%s/pose.plxt", geom);
    if (plx_tensor_load(path, &pose) != PLX_OK || pose.dtype != PLX_F32 || plx_tensor_count(&pose) != (int64_t)N * PCOLS)
        plx_die("%s: want f32 [%d, %d]", path, N, PCOLS);
    snprintf(path, sizeof path, "%s/kappa.txt", geom);
    in->kappa = read_kappa(path);
    snprintf(path, sizeof path, "%s/zbar.plxt", geom);
    plx_img zb = load_map(path, 17, 21);
    for (int k = 0; k < NLAT; k++) in->v_lat[k] = (float)(zb.px[k] / in->kappa);
    plx_img_free(&zb);
    snprintf(path, sizeof path, "%s/mean_face.plxt", flow);
    in->face0 = load_map(path, W, H);

    int want = (int)plx_arg_int(a, "n", 0), stride = want > 0 ? PLX_MAX(1, N / want) : 1;
    int n = (N + stride - 1) / stride;
    in->n = n;
    in->T = PLX_NEW(plx_sim, n);
    in->u = PLX_NEW(float, (size_t)n * DOF);
    in->s = PLX_NEW(float, n);
    in->use = PLX_NEW(uint8_t, n);
    *own_px = stride > 1 ? PLX_NEW(uint8_t, (size_t)n * side * side) : NULL;
    plx_lfw db;
    bool db_open = plx_lfw_open(&db, plx_arg_str(a, "lfw", "data/lfw")) == PLX_OK, have_db = db_open && db.n_images == N;
    in->ident = have_db ? PLX_NEW(int, n) : NULL;
    const float *sv = sim.data, *pv = pose.data;
    for (int j = 0; j < n; j++) {
        int i = j * stride;
        in->T[j] = (plx_sim){sv[4 * i], sv[4 * i + 1], sv[4 * i + 2], sv[4 * i + 3]};
        memcpy(in->u + (size_t)j * DOF, (const float *)u.data + (size_t)i * DOF, sizeof(float) * DOF);
        in->s[j] = pv[(size_t)i * PCOLS];
        in->use[j] = pv[(size_t)i * PCOLS + 7] >= 0.5f;
        if (*own_px) memcpy(*own_px + (size_t)j * side * side, (const uint8_t *)gray->data + (size_t)i * side * side, (size_t)side * side);
        if (have_db) in->ident[j] = db.img_id[i];
    }
    in->src = (plx_cg_source){.px = *own_px ? *own_px : gray->data, .n = n, .w = side, .h = side};
    if (!have_db) plx_log("no LFW index matching the pack: no identity split");
    if (db_open) plx_lfw_free(&db);
    plx_tensor_free(&u), plx_tensor_free(&sim), plx_tensor_free(&pose);
}

static void free_inputs(inputs *in) {
    free(in->T), free(in->u), free(in->s), free(in->use), free(in->ident);
    plx_img_free(&in->face0);
}

static void iter_table(const plx_rl_result *r, double kappa) {
    say("  pass stage level sigma | template: core_var var/2nd sharpness  cost  inlier | lambda    step_rms step_max "
        "  w_rms  gauge(a, g, c)            sec\n");
    for (int k = 0; k < r->n_iter; k++) {
        const plx_rl_iter *it = &r->iter[k];
        say("  %4d %5d %5d %5.2f | %17.2f %7.4f %9.3f %6.3f %6.4f | %8.3g %8.3f %8.3f %7.3f  %+.4f %+.5f %+.4f %5.1f\n",
            k + 1, it->stage, it->level, it->sigma, it->core_var, it->core_ratio, it->sharpness, it->cost, it->inlier,
            it->lambda_abs, kappa * it->step_rms, kappa * it->step_max, kappa * it->w_rms, kappa * it->gauge[0],
            kappa * it->gauge[1], it->gauge[2], it->seconds);
    }
    say("  (step_rms, step_max, w_rms in px of depth = kappa x units of v, over the core ellipse; the template of pass k\n"
        "   is the population rendered with the w of pass k - 1)\n");
}

/* The depth part of a warp layer: w with span{1, Y0, v} removed over the core, weights `support`
 * (relief.h step 6). The removed parts are not depth: α + γ Y0 is a translation / shear of every
 * photo proportional to its yaw, c v a rescaling of s along the lattice pattern. */
static plx_img gauged(const plx_img *w, const plx_img *v, const plx_img *support, plx_core_shape cs, double coef[3]) {
    plx_img g = plx_img_clone(w);
    plx_rl_gauge(&g, v, support, cs, coef);
    return g;
}

typedef struct { double corr, corr_hp, rms; } agreement;

/* Both halves from w = 0; their depth parts (gauged with the common weights `support`) compared. */
static agreement split_half(const inputs *in, const plx_rl_params *p, const plx_cg_template *init, const uint8_t *side,
                            const float *s, const float *shift, const plx_img *v, const plx_img *support, plx_img *wa,
                            plx_img *wb, int *na, int *nb) {
    uint8_t *ua = PLX_NEW(uint8_t, in->n), *ub = PLX_NEW(uint8_t, in->n);
    *na = *nb = 0;
    for (int i = 0; i < in->n; i++) {
        ua[i] = in->use[i] && !side[i], ub[i] = in->use[i] && side[i];
        *na += ua[i], *nb += ub[i];
    }
    plx_rl_result A, B;
    plx_rl_data da = as_data(in, s, ua, shift), db = as_data(in, s, ub, shift);
    plx_rl_estimate(&da, p, init, NULL, false, &A);
    plx_rl_estimate(&db, p, init, NULL, false, &B);
    agreement g;
    *wa = gauged(&A.w, v, support, p->core, NULL), *wb = gauged(&B.w, v, support, p->core, NULL);
    plx_rl_compare(wa, wb, p->core, 4.0f, &g.corr, &g.corr_hp, &g.rms);
    plx_rl_result_free(&A), plx_rl_result_free(&B);
    free(ua), free(ub);
    return g;
}

static double corr_f(const float *a, const float *b, const uint8_t *m, int n) {
    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, k = 0;
    for (int i = 0; i < n; i++)
        if (m[i]) k++, sa += a[i], sb += b[i], saa += (double)a[i] * a[i], sbb += (double)b[i] * b[i], sab += (double)a[i] * b[i];
    return (sab - sa * sb / k) / sqrt((saa - sa * sa / k) * (sbb - sb * sb / k));
}

/* Shape report and the profile / section plots of κ v (before) and κ v_hr (after). */
static plx_img shape_report(const plx_img *zb, const plx_img *za, const plx_img *face, plx_core_shape cs) {
    int tip = 0, eye_y = 0, mouth_y = 0;
    double pb = protrusion(zb, cs, NULL), pa = protrusion(za, cs, &tip), ed;
    landmarks(face, tip, &eye_y, &ed, &mouth_y);
    say("\n[4] SHAPE (kappa v_hr, px)\n");
    say("  nose protrusion over the cheeks: before %.2f px (%.3f of the inter-eye distance %.1f px) -> after %.2f px "
        "(%.3f)\n", pb, pb / ed, ed, pa, pa / ed);
    say("  landmark rows (canonical y): eyes %d, nose tip %d (max of kappa v_hr in the face centre), mouth %d\n", eye_y, tip,
        mouth_y);
    say("  midline profile kappa v(X0 = 0, y) before -> after, px:\n   ");
    for (int y = 16; y < H; y += 6)
        if (plx_rl_in_core(cs, 63, y)) say(" y%d %.1f>%.1f", y, PLX_PX(zb, 63, y, 0), PLX_PX(za, 63, y, 0));
    say("\n");
    double xs[H], *c[2];
    c[0] = PLX_NEW(double, H), c[1] = PLX_NEW(double, H);
    for (int y = 0; y < H; y++) {
        xs[y] = y;
        bool ok = plx_rl_in_core(cs, 63, y);
        c[0][y] = ok ? 0.5 * (PLX_PX(zb, 63, y, 0) + PLX_PX(zb, 64, y, 0)) : NAN;
        c[1][y] = ok ? 0.5 * (PLX_PX(za, 63, y, 0) + PLX_PX(za, 64, y, 0)) : NAN;
    }
    const char *nm[2] = {"before: kappa v (dome)", "after: kappa (v + w)"};
    const float *col[2] = {GREY, YELLOW};
    double marks[3] = {eye_y, tip, mouth_y};
    plx_img plots[4];
    plots[0] = line_plot("midline profile X0 = 0 (marks: eyes, nose tip, mouth)", "canonical y (down)", "depth px", xs, H,
                         (const double *const *)c, nm, col, 2, marks, 3);
    int rows[3] = {eye_y, tip, mouth_y};
    const char *rn[3] = {"horizontal section at eye height", "horizontal section at nose-tip height",
                         "horizontal section at mouth height"};
    for (int r = 0; r < 3; r++) {
        double xx[W];
        for (int x = 0; x < W; x++) {
            xx[x] = x - PLX_CANON_CX;
            bool ok = plx_rl_in_core(cs, x, rows[r]);
            c[0][x] = ok ? PLX_PX(zb, x, rows[r], 0) : NAN;
            c[1][x] = ok ? PLX_PX(za, x, rows[r], 0) : NAN;
        }
        char t[96];
        snprintf(t, sizeof t, "%s (y = %d)", rn[r], rows[r]);
        plots[1 + r] = line_plot(t, "X0 (px)", "depth px", xx, W, (const double *const *)c, nm, col, 2, NULL, 0);
    }
    plx_img out = stack(plots, 4, 2);
    for (int k = 0; k < 4; k++) plx_img_free(&plots[k]);
    free(c[0]), free(c[1]);
    return out;
}

/* Everything after the synthetic check: estimate, controls, refinement, outputs, pictures. */
static void run_estimate(const plx_args *a, const inputs *inp, const plx_fl_model *mp, const plx_rl_params *pp,
                         const char *out, const char *diag) {
    const inputs in = *inp;
    const plx_fl_model model = *mp;
    const plx_rl_params p = *pp;
    char path[4096];
    int nu = 0;
    for (int i = 0; i < in.n; i++) nu += in.use[i];
    double kap = in.kappa;

    plx_rl_data d = as_data(&in, NULL, NULL, NULL);
    plx_rl_result R;
    double t0 = plx_now();
    plx_rl_estimate(&d, &p, &model.tmpl, NULL, true, &R);
    say("\nESTIMATE (%.0f s): starts from w = 0 against the flow model's template\n", plx_now() - t0);
    iter_table(&R, kap);
    plx_img v = plx_rl_v_dense(in.v_lat);
    double gc[3], wmax = 0;
    plx_img wg = gauged(&R.w, &v, &R.support, p.core, gc);
    for (int k = 0; k < W * H; k++)
        if (plx_rl_in_core(p.core, k % W, k / W)) wmax = fmax(wmax, fabs(kap * wg.px[k]));
    say("  depth part of w (gauged, step 6) over the core: rms %.3f px (kappa w), max %.2f px; the dome kappa v: rms "
        "%.3f px\n  removed by the gauge: alpha %+.3f px, gamma %+.5f px/px (a yaw-proportional translation / shear), "
        "c %+.4f (w along v: the fine regression asks for %.1f%% more depth along the dome than emerge's kappa v)\n",
        kap * rms_core(&wg, p.core), wmax, kap * rms_core(&v, p.core), kap * gc[0], kap * gc[1], gc[2], 100 * gc[2]);

    /* ---- controls ---- */
    plx_img wperm = {0}, zperm = {0}, z = plx_img_new(W, H, 1), wsp[4] = {{0}};
    for (int k = 0; k < W * H; k++) z.px[k] = R.se.px[k] > 0 ? wg.px[k] / R.se.px[k] : 0;
    agreement ag[2] = {{0}};
    if (!plx_arg_flag(a, "no-controls")) {
        say("\n[2] PERMUTATION CONTROL: s shuffled across the trusted photos (same schedule, same everything else)\n");
        float *sp = PLX_NEW(float, in.n);
        int *idx = PLX_NEW(int, in.n), m = 0;
        for (int i = 0; i < in.n; i++)
            if (in.use[i]) idx[m++] = i;
        int *perm = PLX_NEW(int, m);
        for (int k = 0; k < m; k++) perm[k] = idx[k];
        plx_rng rng;
        plx_rng_seed(&rng, 0x5eed);
        plx_rng_shuffle(&rng, perm, m);
        memcpy(sp, in.s, sizeof(float) * in.n);
        for (int k = 0; k < m; k++) sp[idx[k]] = in.s[perm[k]];
        plx_rl_data dp = as_data(&in, sp, NULL, NULL);
        plx_rl_result RP;
        plx_rl_estimate(&dp, &p, &model.tmpl, NULL, false, &RP);
        wperm = gauged(&RP.w, &v, &R.support, p.core, NULL);
        zperm = plx_img_new(W, H, 1);
        for (int k = 0; k < W * H; k++) zperm.px[k] = RP.se.px[k] > 0 ? wperm.px[k] / RP.se.px[k] : 0;
        double rz = rms_core(&z, p.core), rzp = rms_core(&zperm, p.core);
        int n3 = 0, n3p = 0, nc = 0;
        for (int k = 0; k < W * H; k++)
            if (plx_rl_in_core(p.core, k % W, k / W)) nc++, n3 += fabsf(z.px[k]) > 3, n3p += fabsf(zperm.px[k]) > 3;
        say("  rms kappa w (depth part) over the core: real %.3f px, permuted %.3f px (ratio %.1f)\n",
            kap * rms_core(&wg, p.core), kap * rms_core(&wperm, p.core), rms_core(&wg, p.core) / rms_core(&wperm, p.core));
        say("  z = w / se (se: photo-level sandwich standard error of a regression step): rms over the core real %.2f, "
            "permuted %.2f (1 if se is calibrated); |z| > 3 on %.1f%% of core pixels (permuted %.1f%%)\n",
            rz, rzp, 100.0 * n3 / nc, 100.0 * n3p / nc);
        plx_rl_result_free(&RP);
        free(sp), free(idx), free(perm);

        say("\n[3] SPLIT-HALF REPRODUCIBILITY (each half estimated independently from w = 0, same schedule;\n"
            "    high-pass = minus a normalised Gaussian blur of 4 px inside the core)\n");
        uint8_t *side = PLX_NEW(uint8_t, in.n);
        for (int split = 0; split < 2; split++) {
            if (split == 1 && !in.ident) break;
            plx_rng_seed(&rng, 0x5911 + split);
            if (split == 0) {
                for (int i = 0; i < in.n; i++) side[i] = (uint8_t)plx_rng_below(&rng, 2);
            } else {
                int ni = 0;
                for (int i = 0; i < in.n; i++) ni = PLX_MAX(ni, in.ident[i] + 1);
                uint8_t *is = PLX_NEW(uint8_t, ni);
                for (int k = 0; k < ni; k++) is[k] = (uint8_t)plx_rng_below(&rng, 2);
                for (int i = 0; i < in.n; i++) side[i] = is[in.ident[i]];
                free(is);
            }
            int na, nb;
            ag[split] = split_half(&in, &p, &model.tmpl, side, NULL, NULL, &v, &R.support, &wsp[2 * split],
                                   &wsp[2 * split + 1], &na, &nb);
            say("  %-12s %5d / %5d photos: corr %.4f, high-pass corr %.4f, rms difference %.3f px (kappa w)\n",
                split ? "by identity" : "random", na, nb, ag[split].corr, ag[split].corr_hp, kap * ag[split].rms);
        }
        free(side);
    }

    /* ---- yaw refinement (step 8) ---- */
    float *s_ref = NULL;
    plx_rl_result R2 = {0};
    if (!plx_arg_flag(a, "no-refine")) {
        double eta_s = plx_arg_num(a, "eta-s", 0.5);
        float *ds = PLX_NEW(float, in.n), *shift = PLX_NEW(float, in.n);
        s_ref = PLX_NEW(float, in.n);
        plx_rl_refine_yaw(&d, &p, &R.tmpl, &R.w, 0, ds);
        /* the gauge in s: a common rescaling of all s would trade against the depth scale (κ), so
         * the component of the steps along s itself is removed */
        double s2 = 0, sd = 0, d2 = 0, r2 = 0;
        for (int i = 0; i < in.n; i++)
            if (in.use[i]) s2 += (double)in.s[i] * in.s[i], sd += (double)in.s[i] * ds[i];
        double gs = eta_s * sd / s2;
        for (int i = 0; i < in.n; i++) {
            shift[i] = in.use[i] ? (float)(eta_s * ds[i] - gs * in.s[i]) : 0.0f;
            s_ref[i] = in.s[i] + shift[i];
            if (in.use[i]) d2 += (double)shift[i] * shift[i], r2 += (double)s_ref[i] * s_ref[i];
        }
        say("\n[8] YAW REFINEMENT: one Gauss-Newton step per photo against the sharpened model, damped %.2f; the common\n"
            "    rescaling of s in the steps (%+.2f%%) is removed (scale gauge)\n", eta_s, 100 * gs);
        say("  rms s %.4f px; rms change %.4f px (%.1f%% of rms s); corr(s, s') %.4f; rms s' / rms s %.4f\n", sqrt(s2 / nu),
            sqrt(d2 / nu), 100 * sqrt(d2 / s2), corr_f(in.s, s_ref, in.use, in.n), sqrt(r2 / s2));
        plx_rl_params pr = p;
        pr.n_stages = 1;
        pr.stages[0] = p.stages[p.n_stages - 1];
        pr.stages[0].iters = (int)plx_arg_int(a, "refine-iters", 2);
        plx_rl_data d2d = as_data(&in, s_ref, NULL, shift);
        plx_rl_estimate(&d2d, &pr, &R.tmpl, &R.w, true, &R2);
        iter_table(&R2, kap);
        double c, ch, rr;
        plx_img w2g = gauged(&R2.w, &v, &R.support, p.core, NULL);
        plx_rl_compare(&w2g, &wg, p.core, 4.0f, &c, &ch, &rr);
        plx_img_free(&w2g);
        say("  w after refinement vs before: corr %.4f (high-pass %.4f), rms difference %.3f px; sharpness %.3f -> %.3f\n",
            c, ch, kap * rr, R.sharpness, R2.sharpness);
        if (!plx_arg_flag(a, "no-controls")) {
            uint8_t *side = PLX_NEW(uint8_t, in.n);
            plx_rng rng;
            plx_rng_seed(&rng, 0x5911);
            for (int i = 0; i < in.n; i++) side[i] = (uint8_t)plx_rng_below(&rng, 2);
            plx_img wa, wb;
            int na, nb;
            agreement g = split_half(&in, &p, &model.tmpl, side, s_ref, shift, &v, &R.support, &wa, &wb, &na, &nb);
            say("  random split-half with the refined s (same halves, full schedule): corr %.4f, high-pass %.4f, rms diff "
                "%.3f px (before refinement: %.4f, %.4f, %.3f)\n", g.corr, g.corr_hp, kap * g.rms, ag[0].corr,
                ag[0].corr_hp, kap * ag[0].rms);
            plx_img_free(&wa), plx_img_free(&wb);
            free(side);
        }
        snprintf(path, sizeof path, "%s/s_refined.plxt", out);
        if (plx_tensor_save(path, PLX_F32, 1, (int64_t[]){in.n}, s_ref) != PLX_OK) plx_die("cannot write %s", path);
        free(ds), free(shift);
    }
    const plx_rl_result *F = s_ref ? &R2 : &R; /* the final model */
    double fc[3];
    plx_img wF = gauged(&F->w, &v, &F->support, p.core, fc);
    if (s_ref)
        say("  final gauge: alpha %+.3f px, gamma %+.5f, c %+.4f\n", kap * fc[0], kap * fc[1], fc[2]);

    /* ---- shape and sharpness ---- */
    plx_img zb = combine(&v, kap, NULL, 0, p.core, false), za = combine(&v, kap, &wF, kap, p.core, false);
    plx_img prof = shape_report(&zb, &za, &F->mean_gray, p.core);
    say("\n[5] SHARPNESS (same definitions as the flow summary; before = the population rendered at w = 0)\n");
    say("  mean face sharpness %.3f -> %.3f; core feature variance %.2f -> %.2f (var/2nd moment %.4f -> %.4f)\n",
        R.iter[0].sharpness, F->sharpness, R.iter[0].core_var, F->core_var, R.iter[0].core_ratio, F->core_ratio);

    /* ---- outputs ---- */
    plx_img vhr = combine(&v, 1, &wF, 1, p.core, true), zhr = combine(&v, kap, &wF, kap, p.core, true);
    save_map(out, "w.plxt", &F->w);
    save_map(out, "vhr.plxt", &vhr);
    save_map(out, "zbar_hr.plxt", &zhr);
    save_map(out, "support.plxt", &F->support);
    save_map(out, "zscore.plxt", &z);
    save_map(out, "mean_face.plxt", &F->mean_gray);
    plx_fl_model mm = model;
    mm.tmpl = F->tmpl;
    snprintf(path, sizeof path, "%s/model", out);
    if (plx_fl_model_save(&mm, path) != PLX_OK) plx_die("cannot write %s", path);

    /* ---- pictures ---- */
    {
        plx_img zbm = combine(&v, kap, NULL, 0, p.core, true), wk = combine(&wF, kap, NULL, 0, p.core, true),
                sp = combine(&F->support, 1, NULL, 0, p.core, false);
        float lo, hi, m = 0;
        plx_viz_auto_range(zhr.px, (size_t)W * H, 0.0f, 1.0f, &lo, &hi);
        for (int k = 0; k < W * H; k++)
            if (isfinite(wk.px[k])) m = fmaxf(m, fabsf(wk.px[k]));
        float slo, shi;
        plx_viz_auto_range(sp.px, (size_t)W * H, 0.0f, 0.99f, &slo, &shi);
        plx_img t[5] = {cmap_tile(&zbm, lo, hi, PLX_CMAP_SEQ), cmap_tile(&zhr, lo, hi, PLX_CMAP_SEQ),
                        cmap_tile(&wk, -m, m, PLX_CMAP_DIV), cmap_tile(&sp, 0, shi, PLX_CMAP_SEQ),
                        plx_viz_upscale(&F->mean_gray, 2, PLX_UP_BILINEAR)};
        char l0[64], l2[64];
        snprintf(l0, sizeof l0, "before: kappa v [%.0f, %.0f] px", lo, hi);
        snprintf(l2, sizeof l2, "kappa w, +-%.1f px", m);
        const char *lab[5] = {l0, "after: kappa (v + w)", l2, "support G*D", "mean face (after)"};
        plx_img tt[5];
        for (int k = 0; k < 5; k++) tt[k] = titled(&t[k], lab[k]);
        plx_img im = stack(tt, 5, 5);
        save_png(diag, "1_depth_map.png", &im);
        for (int k = 0; k < 5; k++) plx_img_free(&t[k]), plx_img_free(&tt[k]);
        plx_img_free(&im), plx_img_free(&zbm), plx_img_free(&wk), plx_img_free(&sp);
    }
    save_png(diag, "2_profiles.png", &prof);
    {
        plx_img zbm = combine(&v, kap, NULL, 0, p.core, true);
        plx_img b = render_set(&zbm, &in.face0), c = render_set(&zhr, &F->mean_gray);
        plx_img tb = titled(&b, "BEFORE: the emerge dome kappa v (lattice, bilinear), flow mean face"),
                tc = titled(&c, "AFTER: kappa (v + w), refined mean face");
        save_png(diag, "3_render_before.png", &tb);
        save_png(diag, "3_render_after.png", &tc);
        plx_img_free(&b), plx_img_free(&c), plx_img_free(&tb), plx_img_free(&tc), plx_img_free(&zbm);
    }
    if (wperm.w) {
        float m = 0;
        for (int k = 0; k < W * H; k++)
            if (plx_rl_in_core(p.core, k % W, k / W)) m = fmaxf(m, (float)fabs(kap * wg.px[k]));
        const char *lab[8] = {"random half A", "random half B", "identity half A", "identity half B",
                              "high-pass A", "high-pass B", "high-pass id A", "high-pass id B"};
        plx_img t[8];
        int nt = 0;
        for (int k = 0; k < 4; k++) {
            if (!wsp[k].w) continue;
            plx_img wk = combine(&wsp[k], kap, NULL, 0, p.core, true), tk = cmap_tile(&wk, -m, m, PLX_CMAP_DIV);
            t[nt++] = titled(&tk, lab[k]);
            plx_img_free(&wk), plx_img_free(&tk);
        }
        int nmaps = nt;
        for (int k = 0; k < nmaps; k++) { /* high-pass rows: w minus its 4-px normalised blur, via compare's definition */
            plx_img wk = combine(&wsp[k], kap, NULL, 0, p.core, false), bl = plx_img_clone(&wk), msk = plx_img_new(W, H, 1);
            for (int q = 0; q < W * H; q++) {
                bool ic = plx_rl_in_core(p.core, q % W, q / W);
                msk.px[q] = ic, bl.px[q] = ic ? bl.px[q] : 0;
            }
            plx_blur(&bl, &bl, 4.0f), plx_blur(&msk, &msk, 4.0f);
            for (int q = 0; q < W * H; q++) wk.px[q] = plx_rl_in_core(p.core, q % W, q / W) ? wk.px[q] - bl.px[q] / msk.px[q] : NAN;
            plx_img tk = cmap_tile(&wk, -0.5f * m, 0.5f * m, PLX_CMAP_DIV);
            t[nt++] = titled(&tk, lab[4 + k]);
            plx_img_free(&wk), plx_img_free(&bl), plx_img_free(&msk), plx_img_free(&tk);
        }
        plx_img im = stack(t, nt, nmaps);
        save_png(diag, "4_split_half.png", &im);
        for (int k = 0; k < nt; k++) plx_img_free(&t[k]);
        plx_img_free(&im);
        plx_img a0 = combine(&wg, kap, NULL, 0, p.core, true), a1 = combine(&wperm, kap, NULL, 0, p.core, true),
                a2 = combine(&z, 1, NULL, 0, p.core, true), a3 = combine(&zperm, 1, NULL, 0, p.core, true);
        plx_img u4[4] = {cmap_tile(&a0, -m, m, PLX_CMAP_DIV), cmap_tile(&a1, -m, m, PLX_CMAP_DIV),
                         cmap_tile(&a2, -8, 8, PLX_CMAP_DIV), cmap_tile(&a3, -8, 8, PLX_CMAP_DIV)};
        const char *l4[4] = {"kappa w (true s)", "kappa w (s permuted), same scale", "z-score (true s), +-8",
                             "z-score (s permuted), +-8"};
        plx_img t4[4];
        for (int k = 0; k < 4; k++) t4[k] = titled(&u4[k], l4[k]);
        plx_img im4 = stack(t4, 4, 4);
        save_png(diag, "5_permutation.png", &im4);
        for (int k = 0; k < 4; k++) plx_img_free(&u4[k]), plx_img_free(&t4[k]);
        plx_img_free(&im4), plx_img_free(&a0), plx_img_free(&a1), plx_img_free(&a2), plx_img_free(&a3);
    }
    {
        plx_img f0 = plx_viz_upscale(&in.face0, 2, PLX_UP_BILINEAR), f1 = plx_viz_upscale(&F->mean_gray, 2, PLX_UP_BILINEAR);
        char a0[64], a1[64];
        snprintf(a0, sizeof a0, "before (flow mean face), sharpness %.2f", plx_cg_sharpness(&in.face0));
        snprintf(a1, sizeof a1, "after, sharpness %.2f", F->sharpness);
        plx_img t[2] = {titled(&f0, a0), titled(&f1, a1)};
        plx_img im = stack(t, 2, 2);
        save_png(diag, "6_mean_face.png", &im);
        plx_img_free(&t[0]), plx_img_free(&t[1]), plx_img_free(&f0), plx_img_free(&f1), plx_img_free(&im);
    }
    for (int k = 0; k < 4; k++) plx_img_free(&wsp[k]);
    plx_img_free(&wperm), plx_img_free(&zperm), plx_img_free(&z), plx_img_free(&v), plx_img_free(&wg), plx_img_free(&wF);
    plx_img_free(&zb), plx_img_free(&za), plx_img_free(&prof), plx_img_free(&vhr), plx_img_free(&zhr);
    plx_rl_result_free(&R), plx_rl_result_free(&R2);
    free(s_ref);
}

int main(int argc, char **argv) {
    plx_args a = {argc, argv};
    const char *out = plx_arg_str(&a, "out", "runs/main/relief");
    char diag[4096], path[4096];
    snprintf(diag, sizeof diag, "%s/diag", out);
    if (plx_mkdir_p(diag) != PLX_OK) plx_die("cannot create %s", diag);
    plx_fl_model model;
    plx_tensor gray;
    uint8_t *own_px;
    inputs in = {0};
    load_inputs(&a, &in, &model, &gray, &own_px);
    plx_rl_params p = plx_rl_default(&model.p.cg);
    if (plx_arg_str(&a, "sched", NULL)) parse_sched(plx_arg_str(&a, "sched", NULL), &p);
    p.lambda = (float)plx_arg_num(&a, "lambda", p.lambda);
    p.eta = (float)plx_arg_num(&a, "eta", p.eta);
    snprintf(path, sizeof path, "%s/summary.txt", out);
    g_sum = fopen(path, "w");
    if (!g_sum) plx_die("cannot write %s", path);
    int nu = 0;
    for (int i = 0; i < in.n; i++) nu += in.use[i];
    char sch[256];
    sched_str(&p, sch, sizeof sch);
    double t0 = plx_now();
    say("plx_relief: %d photos (%d trusted: emerge robust weight >= 0.5), %d threads; kappa %.4f px\n", in.n, nu,
        plx_num_threads(), in.kappa);
    say("schedule (level:sigma px:passes) %s, lambda %.3g (x mean G*D over the core), eta %.2f; features as the flow "
        "model (K=%d, huber %.2f sqrt(K)); core ellipse (%.0f, %.0f, %.0f)\n", sch, p.lambda, p.eta, p.cg.feat.k,
        p.cg.huber, p.core.ax, p.core.ay, p.core.yc);
    if (!plx_arg_flag(&a, "no-synth")) run_synth(&a, &p, &in, diag);
    if (!plx_arg_flag(&a, "synth-only")) run_estimate(&a, &in, &model, &p, out, diag);
    say("\noutputs in %s, diagnostics in %s; total %.0f s\n", out, diag, plx_now() - t0);
    fclose(g_sum);
    free_inputs(&in);
    free(own_px);
    plx_tensor_free(&gray);
    plx_fl_model_free(&model);
    return 0;
}
