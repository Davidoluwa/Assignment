/* Finding the face in an arbitrary frame with what the system already learned: the population
 * template of orientation features, slid over position x scale x roll, then similarity
 * alignment of the best hypotheses. And the fixed crop window. See fuse.h.
 *
 * WHY CORRELATION, NOT THE ALIGNMENT COST. The alignment cost Σ ω (F - μ)² / (σ² + ε) is the
 * right objective once a face is known to be there, but as a detector it prefers blank wall:
 * contrast-normalised features of a flat region are ~0, which costs Σ ω μ² / (σ² + ε) ≈ 3.4 per
 * sample, while a real face fitted by congealing costs ≈ K = 8 (measured on a synthetic clip:
 * all five best hypotheses were in the smooth background). The score used instead is the
 * weighted normalised cross-correlation of the feature channels with the template mean,
 * centred per channel, with the same weights ω / (σ² + ε): invariant to feature energy, so an
 * empty or uniformly textured region scores ~0 and a face scores high. */
#include "fuse/fuse.h"

#include "feat/feat.h"
#include "ops/ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define DEG (M_PI / 180.0)

/* ---- the window ------------------------------------------------------------------------------ */
plx_fu_window plx_fu_window_for(plx_sim T) {
    double s = plx_sim_scale(T), cx, cy;
    plx_sim_apply(T, PLX_CANON_CX, PLX_CANON_CY, &cx, &cy);
    return (plx_fu_window){.s = s, .x0 = cx - s * PLX_CANON_IMG_CX, .y0 = cy - s * PLX_CANON_IMG_CY};
}

plx_img plx_fu_window_extract(const plx_img *frame, plx_fu_window w) {
    plx_img src = plx_img_clone(frame);
    if (w.s > 1) plx_blur(&src, &src, (float)(0.5 * sqrt(w.s * w.s - 1)));
    plx_img out = plx_img_new(PLX_CANON_IMG_SIDE, PLX_CANON_IMG_SIDE, frame->c);
    float A[6] = {(float)w.s, 0, (float)w.x0, 0, (float)w.s, (float)w.y0};
    plx_warp_affine(&src, A, PLX_INTERP_BILINEAR, &out, NULL);
    plx_img_free(&src);
    return out;
}

plx_sim plx_fu_sim_to_crop(plx_fu_window w, plx_sim T) {
    plx_sim C = {T.a / w.s, T.b / w.s, 0, 0};
    C.tx = (PLX_CANON_IMG_CX + T.tx - w.x0) / w.s - PLX_CANON_IMG_CX;
    C.ty = (PLX_CANON_IMG_CY + T.ty - w.y0) / w.s - PLX_CANON_IMG_CY;
    return C;
}

plx_sim plx_fu_sim_to_frame(plx_fu_window w, plx_sim C) {
    plx_sim T = {C.a * w.s, C.b * w.s, 0, 0};
    T.tx = w.x0 + w.s * (PLX_CANON_IMG_CX + C.tx) - PLX_CANON_IMG_CX;
    T.ty = w.y0 + w.s * (PLX_CANON_IMG_CY + C.ty) - PLX_CANON_IMG_CY;
    return T;
}

/* ---- locator ------------------------------------------------------------------------------- */
plx_fu_locate_opts plx_fu_locate_defaults(void) {
    return (plx_fu_locate_opts){.scale_min = 0, .scale_max = 0, .scale_step = pow(2.0, 0.25), .rot_max_deg = 10,
                                .level = 2, .n_refine = 5};
}

void plx_fu_locator_init(plx_fu_locator *L, const plx_fl_model *m) {
    memset(L, 0, sizeof *L);
    L->m = m;
    L->p = m->p.cg;
    /* congealing's own coarse-to-fine schedule, shortened (the search already got within a few px) */
    plx_cg_stage st[] = {{2, 1.0f, 10, 0.1f}, {2, 0.5f, 10, 0}, {1, 0, 10, 0}, {0, 0, 10, 0}, {0, 0, 10, 0}};
    L->n_stages = (int)(sizeof st / sizeof st[0]);
    for (int s = 0; s < L->n_stages; s++) {
        L->stages[s] = st[s];
        L->refs[s] = plx_cg_ref_build(&m->tmpl, &L->p, st[s].level, st[s].blur);
    }
}

void plx_fu_locator_free(plx_fu_locator *L) {
    for (int s = 0; s < L->n_stages; s++) plx_cg_ref_free(&L->refs[s]);
    memset(L, 0, sizeof *L);
}

/* Template samples of the search level: positions, window, weights, channel-centred mean. */
typedef struct {
    int     n, K, tw, th;
    int    *sx, *sy;
    float  *om, *wt, *mu; /* [n], [n K] weight ω / (σ² + ε), [n K] μ minus its weighted channel mean */
    double  om_sum, mu2;  /* Σ ω; Σ ω wt μ̃² */
} tmpl_samples;

static void samples_build(const plx_fu_locator *L, int lev, tmpl_samples *ts) {
    const plx_cg_template *t = &L->m->tmpl;
    int K = t->K, tw = t->mean[lev].w, th = t->mean[lev].h;
    size_t tplane = (size_t)tw * th;
    float *om = plx_cg_level_window(tw, th, lev, &L->p);
    double eps = plx_cg_var_floor(&t->var[lev], lev, &L->p);
    memset(ts, 0, sizeof *ts);
    ts->K = K, ts->tw = tw, ts->th = th;
    for (size_t q = 0; q < tplane; q++) ts->n += om[q] > 0;
    ts->sx = PLX_NEW(int, ts->n), ts->sy = PLX_NEW(int, ts->n), ts->om = PLX_NEW(float, ts->n);
    ts->wt = PLX_NEW(float, (size_t)ts->n * K), ts->mu = PLX_NEW(float, (size_t)ts->n * K);
    double *cw = PLX_NEW(double, K), *cm = PLX_NEW(double, K);
    int n = 0;
    for (int y = 0; y < th; y++)
        for (int x = 0; x < tw; x++) {
            size_t q = (size_t)y * tw + x;
            if (om[q] <= 0) continue;
            ts->sx[n] = x, ts->sy[n] = y, ts->om[n] = om[q], ts->om_sum += om[q];
            for (int k = 0; k < K; k++) {
                double w = 1.0 / (t->var[lev].px[k * tplane + q] + eps), m = t->mean[lev].px[k * tplane + q];
                ts->wt[(size_t)n * K + k] = (float)w;
                ts->mu[(size_t)n * K + k] = (float)m;
                cw[k] += om[q] * w, cm[k] += om[q] * w * m;
            }
            n++;
        }
    for (int i = 0; i < n; i++)
        for (int k = 0; k < K; k++) {
            float *m = &ts->mu[(size_t)i * K + k];
            *m -= (float)(cm[k] / cw[k]);
            ts->mu2 += ts->om[i] * ts->wt[(size_t)i * K + k] * (double)*m * *m;
        }
    free(om), free(cw), free(cm);
}

static void samples_free(tmpl_samples *ts) {
    free(ts->sx), free(ts->sy), free(ts->om), free(ts->wt), free(ts->mu);
}

/* Weighted NCC of features F (any size, K channels) with the template placed at (ox, oy); -2
 * when the placement is mostly outside the valid area V (nullable = all valid). */
static double ncc_at(const tmpl_samples *ts, const plx_img *F, const plx_img *V, int ox, int oy) {
    int W = F->w, K = ts->K;
    size_t plane = (size_t)W * F->h;
    double sv = 0, fm[64] = {0}, fw[64] = {0}, f2 = 0, fmu = 0;
    PLX_ASSERT(K <= 64);
    for (int i = 0; i < ts->n; i++) {
        size_t o = (size_t)(oy + ts->sy[i]) * W + (ox + ts->sx[i]);
        if (V && V->px[o] <= 0.5f) continue;
        sv += ts->om[i];
        const float *wt = ts->wt + (size_t)i * K, *mu = ts->mu + (size_t)i * K;
        for (int k = 0; k < K; k++) {
            double w = ts->om[i] * wt[k], f = F->px[k * plane + o];
            fm[k] += w * f, fw[k] += w, f2 += w * f * f, fmu += w * f * mu[k];
        }
    }
    if (sv < 0.8 * ts->om_sum) return -2;
    for (int k = 0; k < K; k++) f2 -= fm[k] * fm[k] / fw[k]; /* per-channel centring */
    return f2 > 0 ? fmu / sqrt(f2 * ts->mu2) : 0;
}

typedef struct {
    const tmpl_samples *ts;
    const plx_img *F, *V;
    float *score;
    int nx, ny;
} slide_ctx;

static void slide_row(int64_t oy, int tid, void *pc) {
    (void)tid;
    slide_ctx *c = pc;
    for (int ox = 0; ox < c->nx; ox++) c->score[oy * c->nx + ox] = (float)ncc_at(c->ts, c->F, c->V, ox, (int)oy);
}

typedef struct { double score, s, rho, gx, gy; plx_sim T; } hyp;

static int cmp_hyp(const void *a, const void *b) {
    double x = ((const hyp *)a)->score, y = ((const hyp *)b)->score;
    return (x < y) - (x > y); /* descending */
}

/* NCC of the face at similarity T (Ω -> img) at the search level. */
static double score_at(const plx_fu_locator *L, const tmpl_samples *ts, int lev, const plx_img *img, plx_sim T) {
    plx_cg_view v;
    plx_cg_render(img, T, &L->p, &v);
    double s = ncc_at(ts, &v.feat.lev[lev], &v.vpyr.lev[lev], 0, 0);
    plx_cg_view_free(&v);
    return s;
}

plx_fu_located plx_fu_locate(const plx_fu_locator *L, const plx_img *gray, const plx_fu_locate_opts *o) {
    PLX_ASSERT(gray->c == 1);
    const plx_cg_params *p = &L->p;
    int lev = o->level;
    PLX_ASSERT(lev >= 0 && lev < L->m->tmpl.levels);
    tmpl_samples ts;
    samples_build(L, lev, &ts);
    double smin = o->scale_min > 0 ? o->scale_min : 0.5;
    double smax = o->scale_max > 0 ? o->scale_max : fmax(smin, PLX_MIN(gray->w, gray->h) / 150.0);
    int nrot = o->rot_max_deg > 0 ? 3 : 1;
    hyp *H = NULL;
    int nh = 0;
    double lsc = ldexp(1.0, lev);
    for (double s = smin; s <= smax * 1.0001; s *= o->scale_step) {
        plx_img src = plx_img_clone(gray);
        if (s > 1) plx_blur(&src, &src, (float)(0.5 * sqrt(s * s - 1)));
        for (int ir = 0; ir < nrot; ir++) {
            double rho = (ir - (nrot - 1) / 2) * o->rot_max_deg * DEG, c = cos(rho), sn = sin(rho);
            /* G(q) = frame(s R (q + qmin)), q over the bounding box of the frame in G coordinates:
             * a face of scale s and roll rho is upright at unit scale in G */
            double qmin[2] = {INFINITY, INFINITY}, qmax[2] = {-INFINITY, -INFINITY};
            for (int k = 0; k < 4; k++) {
                double px = k & 1 ? gray->w - 1 : 0, py = k & 2 ? gray->h - 1 : 0;
                double qx = (c * px + sn * py) / s, qy = (-sn * px + c * py) / s;
                qmin[0] = fmin(qmin[0], qx), qmin[1] = fmin(qmin[1], qy);
                qmax[0] = fmax(qmax[0], qx), qmax[1] = fmax(qmax[1], qy);
            }
            int gw = (int)ceil(qmax[0] - qmin[0]) + 1, gh = (int)ceil(qmax[1] - qmin[1]) + 1;
            if (gw < (ts.tw + 2) * lsc || gh < (ts.th + 2) * lsc) continue;
            float A[6] = {(float)(s * c), (float)(-s * sn), (float)(s * (c * qmin[0] - sn * qmin[1])),
                          (float)(s * sn), (float)(s * c), (float)(s * (sn * qmin[0] + c * qmin[1]))};
            plx_img G = plx_img_new(gw, gh, 1), Gv = plx_img_new(gw, gh, 1);
            plx_warp_affine(&src, A, PLX_INTERP_BILINEAR, &G, &Gv);
            plx_img feat = plx_orient_channels(&G, &p->feat);
            plx_pyramid fp = plx_pyramid_build(&feat, lev + 1, 1), vp = plx_pyramid_build(&Gv, lev + 1, 1);
            plx_img_free(&G), plx_img_free(&Gv), plx_img_free(&feat);
            slide_ctx sc = {.ts = &ts, .F = &fp.lev[lev], .V = &vp.lev[lev]};
            sc.nx = fp.n > lev ? fp.lev[lev].w - ts.tw + 1 : 0, sc.ny = fp.n > lev ? fp.lev[lev].h - ts.th + 1 : 0;
            if (sc.nx > 0 && sc.ny > 0) {
                sc.score = PLX_NEW(float, (size_t)sc.nx * sc.ny);
                plx_parallel_for(sc.ny, 1, slide_row, &sc);
                for (int y = 0; y < sc.ny; y++)
                    for (int x = 0; x < sc.nx; x++) {
                        float v = sc.score[y * sc.nx + x];
                        bool mx = v > -1;
                        for (int dy = -1; dy <= 1 && mx; dy++)
                            for (int dx = -1; dx <= 1 && mx; dx++) {
                                int xx = x + dx, yy = y + dy;
                                if ((dx || dy) && xx >= 0 && yy >= 0 && xx < sc.nx && yy < sc.ny &&
                                    sc.score[yy * sc.nx + xx] > v)
                                    mx = false;
                            }
                        if (!mx) continue;
                        /* template level pixel q is canonical point 2^L (q + 0.5) - 0.5 - margin and
                         * lands on G level-0 point 2^L x + that + margin, i.e. T(p) = s R (p + d) */
                        double cx = PLX_CANON_CX + p->margin + lsc * x + qmin[0];
                        double cy = PLX_CANON_CY + p->margin + lsc * y + qmin[1];
                        hyp h = {.score = v, .s = s, .rho = rho};
                        h.T = plx_sim_make(s, rho, s * (c * cx - sn * cy) - PLX_CANON_IMG_CX,
                                           s * (sn * cx + c * cy) - PLX_CANON_IMG_CY);
                        plx_sim_apply(h.T, PLX_CANON_CX, PLX_CANON_CY, &h.gx, &h.gy);
                        H = plx_realloc(H, sizeof(hyp) * (nh + 1));
                        H[nh++] = h;
                    }
                free(sc.score);
            }
            plx_pyramid_free(&fp), plx_pyramid_free(&vp);
        }
        plx_img_free(&src);
    }

    plx_fu_located res = {.score = -2, .runner_score = -2, .T = plx_sim_make(1, 0, 0, 0)};
    if (nh == 0) {
        samples_free(&ts);
        return res;
    }
    /* non-maximum suppression across scales and rolls: a hypothesis within a third of a face
     * (~100 px at unit scale) of a better one is the same face */
    qsort(H, nh, sizeof(hyp), cmp_hyp);
    hyp *keep = PLX_NEW(hyp, o->n_refine);
    int nk = 0;
    for (int i = 0; i < nh && nk < o->n_refine; i++) {
        bool dup = false;
        for (int j = 0; j < nk && !dup; j++)
            dup = hypot(H[i].gx - keep[j].gx, H[i].gy - keep[j].gy) < 0.33 * 100 * fmin(H[i].s, keep[j].s);
        if (!dup) keep[nk++] = H[i];
    }
    free(H);

    /* refinement: crop around the hypothesis, similarity alignment there (inside congealing's
     * bounds), twice so the second crop is centred on the refined face; then re-scored. An
     * alignment that runs off (it can shrink a face onto its own nose: 2 of 60 pasted LFW faces)
     * scores worse than where it started, and the unrefined hypothesis is kept instead. */
    double *refined = PLX_NEW(double, nk);
    for (int i = 0; i < nk; i++) {
        plx_sim T = keep[i].T;
        plx_cg_fit fit = {0};
        double s0 = 0;
        for (int pass = 0; pass < 2; pass++) {
            plx_fu_window w = plx_fu_window_for(T);
            plx_img crop = plx_fu_window_extract(gray, w);
            plx_sim C0 = plx_fu_sim_to_crop(w, T);
            if (pass == 0) s0 = score_at(L, &ts, lev, &crop, C0);
            plx_sim C = plx_cg_align_image(&crop, C0, L->refs, L->stages, L->n_stages, p, &fit, NULL);
            T = plx_fu_sim_to_frame(w, C);
            if (pass == 1) refined[i] = score_at(L, &ts, lev, &crop, C);
            plx_img_free(&crop);
        }
        if (refined[i] < s0) T = keep[i].T, refined[i] = s0;
        keep[i].T = T;
        if (refined[i] > res.score) res.score = refined[i], res.T = T, res.search_score = keep[i].score, res.cost = fit.cost;
    }
    /* the runner-up must be another place, not the winner found twice */
    double gx, gy;
    plx_sim_apply(res.T, PLX_CANON_CX, PLX_CANON_CY, &gx, &gy);
    for (int i = 0; i < nk; i++) {
        double hx, hy;
        plx_sim_apply(keep[i].T, PLX_CANON_CX, PLX_CANON_CY, &hx, &hy);
        if (hypot(hx - gx, hy - gy) >= 0.33 * 100 * plx_sim_scale(res.T)) res.runner_score = fmax(res.runner_score, refined[i]);
    }
    res.n_hyp = nk;
    free(keep), free(refined);
    samples_free(&ts);
    return res;
}
