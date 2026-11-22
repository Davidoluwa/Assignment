/* Tracking a clip: population flow per frame (warm-started), then personal re-tracking against
 * the person's own template. See fuse.h.
 *
 * WHY THE PERSONAL FIT NEVER RE-RENDERS. flow.h (WHY TWO KINDS OF STAGE): orientation channels
 * respond to the strain of an 8-px hat very differently from the Gauss-Newton model, so
 * re-rendering every round through a fine field diverges. It does here too: re-rendering each
 * personal round made the frame-mean field change grow 3.6 -> 19 -> 119 px in three rounds on a
 * synthetic clip. So each frame is rendered ONCE, through its population warp, and everything
 * personal is an increment a_f SAMPLED on top of that render, exactly as the population's own
 * 8-px stages refine: the personal template is the mean of the sampled views (what the solver
 * compares with it), and the full warp is W_f(p) = T_f(p + compose(a_f, u_pop,f)(p)). */
#include "fuse/fuse.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

enum { MAX_DOF = 2 * PLX_LAT_MAX_NODES, MAX_STAGES = 2 };

plx_fu_track_opts plx_fu_track_defaults(void) {
    return (plx_fu_track_opts){.rounds = 3, .step = 8, .lambda = 3, .iters = 10, .coarse = true};
}

/* ---- sampled views ------------------------------------------------------------------------------ */
/* dst (src's size) = src sampled at q + a(p(q)) / 2^level, p(q) the canonical point of level pixel q
 * (bilinear, clamped into the image: the same sampling the solver does). */
static void sample_level(const plx_img *src, int level, int margin, const plx_lattice *L, const float *a, plx_img *dst) {
    int w = src->w, h = src->h;
    size_t plane = (size_t)w * h;
    double inv = ldexp(1.0, -level);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            double v[2];
            plx_lat_eval(L, a, plx_pyr_coord((float)x, level, 0) - margin, plx_pyr_coord((float)y, level, 0) - margin, v);
            float sx = (float)PLX_CLAMP(x + v[0] * inv, 0.0, (double)(w - 1));
            float sy = (float)PLX_CLAMP(y + v[1] * inv, 0.0, (double)(h - 1));
            for (int k = 0; k < src->c; k++)
                dst->px[k * plane + (size_t)y * w + x] = plx_sample1(src, k, PLX_INTERP_BILINEAR, sx, sy);
        }
}

/* ---- the clip ---------------------------------------------------------------------------------- */
typedef struct {
    const plx_img       *grays;
    const plx_fl_ref    *refs;
    int                  n_refs;
    const plx_fl_params *p;
    plx_fu_track        *t;
    plx_cg_view         *views;  /* [F] rendered through the population warps, kept */
    float               *ub;     /* [F][dof] population fields on the personal lattice */
    float               *a;      /* [F][dof] personal increments */
    double              *s1[PLX_CG_MAX_LEVELS], *s2[PLX_CG_MAX_LEVELS]; /* [threads][level size] template sums */
    double              *gsum;   /* [threads][Ω+] gray sums */
    size_t               lsize[PLX_CG_MAX_LEVELS], gsize;
    plx_fl_nodestat     *ns;     /* [F] final pass */
    int                  mode;   /* 0 render, 1 accumulate the template, 2 personal fit, 3 final costs */
} pass_ctx;

static void pass_one(int64_t f, int tid, void *pc) {
    pass_ctx *c = pc;
    plx_fu_track *t = c->t;
    int dof = plx_lat_dof(&t->lat), margin = c->p->cg.margin;
    float *ub = c->ub + (size_t)f * dof, *a = c->a + (size_t)f * dof;
    plx_cg_view *v = &c->views[f];
    if (c->mode == 0) {
        plx_fl_render(&c->grays[f], t->T_pop[f], &t->lat, ub, &c->p->cg, v);
    } else if (c->mode == 1) {
        for (int l = 0; l < v->feat.n; l++) {
            const plx_img *src = &v->feat.lev[l];
            plx_img s = plx_img_new(src->w, src->h, src->c);
            sample_level(src, l, margin, &t->lat, a, &s);
            double *s1 = c->s1[l] + (size_t)tid * c->lsize[l], *s2 = c->s2[l] + (size_t)tid * c->lsize[l];
            for (size_t k = 0; k < c->lsize[l]; k++) s1[k] += s.px[k], s2[k] += (double)s.px[k] * s.px[k];
            plx_img_free(&s);
        }
        plx_img g = plx_img_new(v->gray.w, v->gray.h, 1);
        sample_level(&v->gray, 0, margin, &t->lat, a, &g);
        double *gs = c->gsum + (size_t)tid * c->gsize;
        for (size_t k = 0; k < c->gsize; k++) gs[k] += g.px[k];
        plx_img_free(&g);
    } else if (c->mode == 2) {
        for (int s = 0; s < c->n_refs; s++) plx_fl_solve(&c->refs[s], v, c->p, ub, a, NULL);
    } else {
        /* costs of the population warps (a = 0) and of the personal ones, one evaluation each */
        plx_fl_ref one = c->refs[0];
        one.st.iters = 1;
        float z[MAX_DOF] = {0}, w[MAX_DOF];
        memcpy(w, a, sizeof(float) * dof);
        t->cost_pp[f] = plx_fl_solve(&one, v, c->p, ub, z, NULL).cost0;
        t->cost[f] = plx_fl_solve(&one, v, c->p, ub, w, &c->ns[f]).cost0;
    }
}

static void build_template(pass_ctx *c, int F, plx_img *gray_out) {
    const plx_cg_params *p = &c->p->cg;
    plx_fu_track *t = c->t;
    int nt = plx_num_threads();
    for (int l = 0; l < p->levels; l++) {
        memset(c->s1[l], 0, sizeof(double) * c->lsize[l] * nt);
        memset(c->s2[l], 0, sizeof(double) * c->lsize[l] * nt);
    }
    memset(c->gsum, 0, sizeof(double) * c->gsize * nt);
    c->mode = 1;
    plx_parallel_for(F, 1, pass_one, c);
    plx_cg_template_free(&t->tmpl);
    t->tmpl.K = p->feat.k, t->tmpl.margin = p->margin, t->tmpl.levels = p->levels, t->tmpl.count = F;
    for (int l = 0; l < p->levels; l++) {
        const plx_img *f0 = &c->views[0].feat.lev[l];
        t->tmpl.mean[l] = plx_img_new(f0->w, f0->h, f0->c);
        t->tmpl.var[l] = plx_img_new(f0->w, f0->h, f0->c);
        for (size_t k = 0; k < c->lsize[l]; k++) {
            double s1 = 0, s2 = 0;
            for (int th = 0; th < nt; th++) s1 += c->s1[l][th * c->lsize[l] + k], s2 += c->s2[l][th * c->lsize[l] + k];
            double m = s1 / F;
            t->tmpl.mean[l].px[k] = (float)m;
            t->tmpl.var[l].px[k] = (float)fmax(s2 / F - m * m, 0);
        }
    }
    plx_img ext = plx_img_new(c->views[0].gray.w, c->views[0].gray.h, 1);
    for (size_t k = 0; k < c->gsize; k++) {
        double s = 0;
        for (int th = 0; th < nt; th++) s += c->gsum[th * c->gsize + k];
        ext.px[k] = (float)(s / F);
    }
    plx_img_free(gray_out);
    *gray_out = plx_canon_crop(&ext, p->margin);
    plx_img_free(&ext);
}

void plx_fu_track_clip(const plx_fl_fitter *pop, const plx_img *grays, int F, plx_sim T0, const plx_fu_track_opts *o,
                       plx_fu_track *t) {
    PLX_ASSERT(F >= 1 && (o->step == 4 || o->step == 8) && o->rounds <= 8);
    const plx_fl_model *m = pop->m;
    plx_lattice L8 = plx_lattice_make(PLX_FL_STEP);
    int d8 = plx_lat_dof(&L8);
    memset(t, 0, sizeof *t);
    t->F = F;
    t->lat = plx_lattice_make(o->step);
    int dof = plx_lat_dof(&t->lat), N = plx_lat_nodes(&t->lat), nt = plx_num_threads();
    t->T_pop = PLX_NEW(plx_sim, F);
    t->u_pop = PLX_NEW(float, (size_t)F * d8);
    t->T = PLX_NEW(plx_sim, F);
    t->u = PLX_NEW(float, (size_t)F * dof);
    t->cost_pop = PLX_NEW(double, F);
    t->cost_pp = PLX_NEW(double, F);
    t->cost = PLX_NEW(double, F);
    t->node_cost = PLX_NEW(float, (size_t)F * N);
    t->node_data = PLX_NEW(float, N);

    /* population pass, warm-started from the previous frame's similarity */
    plx_sim Tw = T0;
    for (int f = 0; f < F; f++) {
        plx_fl_fit fit = plx_fl_fit_image(pop, &grays[f], Tw, &t->T_pop[f], t->u_pop + (size_t)f * d8);
        t->cost_pop[f] = fit.cost;
        Tw = t->T_pop[f];
    }

    /* personal stages: (level 1), level 0, both on the personal lattice; damping and prox as
     * the population's sampled stages, the bending weight the personal one */
    plx_fl_params p = m->p;
    plx_fl_stage st[MAX_STAGES];
    int ns = 0;
    if (o->coarse) st[ns++] = (plx_fl_stage){o->step, 1, 0, o->lambda, 0.3f, o->iters, 0};
    st[ns++] = (plx_fl_stage){o->step, 0, 0, o->lambda, 0.3f, o->iters, 0};

    pass_ctx c = {.grays = grays, .p = &p, .t = t, .views = PLX_NEW(plx_cg_view, F),
                  .ub = PLX_NEW(float, (size_t)F * dof), .a = PLX_NEW(float, (size_t)F * dof)};
    for (int f = 0; f < F; f++) plx_lat_resample(&L8, t->u_pop + (size_t)f * d8, &t->lat, c.ub + (size_t)f * dof);
    c.mode = 0;
    plx_parallel_for(F, 1, pass_one, &c);
    for (int l = 0; l < p.cg.levels; l++) {
        const plx_img *f0 = &c.views[0].feat.lev[l];
        c.lsize[l] = (size_t)f0->w * f0->h * f0->c;
        c.s1[l] = PLX_NEW(double, c.lsize[l] * nt);
        c.s2[l] = PLX_NEW(double, c.lsize[l] * nt);
    }
    c.gsize = (size_t)c.views[0].gray.w * c.views[0].gray.h;
    c.gsum = PLX_NEW(double, c.gsize * nt);
    build_template(&c, F, &t->pop_gray);

    plx_fl_ref refs[MAX_STAGES];
    for (int r = 0; r < o->rounds; r++) {
        for (int s = 0; s < ns; s++) refs[s] = plx_fl_ref_build(&t->tmpl, &p, &st[s]);
        c.refs = refs, c.n_refs = ns, c.mode = 2;
        plx_parallel_for(F, 1, pass_one, &c);
        for (int s = 0; s < ns; s++) plx_fl_ref_free(&refs[s]);
        /* gauge: the frame-mean increment is a common drift of template and fits */
        double *mean = PLX_NEW(double, dof), rms = 0;
        for (int f = 0; f < F; f++)
            for (int k = 0; k < dof; k++) mean[k] += c.a[(size_t)f * dof + k] / F;
        int nc = 0;
        for (int n = 0; n < N; n++) {
            double x = plx_lat_x(&t->lat, n % t->lat.nx), y = plx_lat_y(&t->lat, n / t->lat.nx);
            if (x >= PLX_LAT_CORE_X0 && x <= PLX_LAT_CORE_X1 && y >= PLX_LAT_CORE_Y0 && y <= PLX_LAT_CORE_Y1)
                rms += mean[n] * mean[n] + mean[N + n] * mean[N + n], nc++;
        }
        for (int f = 0; f < F; f++)
            for (int k = 0; k < dof; k++) c.a[(size_t)f * dof + k] -= (float)mean[k];
        free(mean);
        t->drift[r] = sqrt(rms / PLX_MAX(nc, 1));
        build_template(&c, F, &t->tmpl_gray);
    }
    if (o->rounds == 0) t->tmpl_gray = plx_img_clone(&t->pop_gray);

    /* evaluation against the final personal template (level 0 stage) */
    refs[0] = plx_fl_ref_build(&t->tmpl, &p, &st[ns - 1]);
    plx_fl_data_fraction(&refs[0], t->node_data);
    plx_fl_nodestat *nst = PLX_NEW(plx_fl_nodestat, F);
    for (int f = 0; f < F; f++) nst[f] = (plx_fl_nodestat){PLX_NEW(double, N), PLX_NEW(double, N)};
    c.refs = refs, c.n_refs = 1, c.mode = 3, c.ns = nst;
    plx_parallel_for(F, 1, pass_one, &c);
    plx_fl_ref_free(&refs[0]);
    for (int f = 0; f < F; f++) {
        for (int n = 0; n < N; n++)
            t->node_cost[(size_t)f * N + n] = nst[f].mass[n] > 0 ? (float)(nst[f].cost[n] / nst[f].mass[n]) : NAN;
        free(nst[f].cost), free(nst[f].mass);
        /* the full personal warp */
        t->T[f] = t->T_pop[f];
        plx_lat_compose(&t->lat, c.a + (size_t)f * dof, c.ub + (size_t)f * dof, t->u + (size_t)f * dof);
        plx_cg_view_free(&c.views[f]);
    }
    free(nst);
    for (int l = 0; l < p.cg.levels; l++) free(c.s1[l]), free(c.s2[l]);
    free(c.gsum), free(c.views), free(c.ub), free(c.a);
}

void plx_fu_track_free(plx_fu_track *t) {
    free(t->T_pop), free(t->u_pop), free(t->T), free(t->u), free(t->cost_pop), free(t->cost_pp), free(t->cost);
    free(t->node_cost), free(t->node_data);
    plx_img_free(&t->tmpl_gray), plx_img_free(&t->pop_gray);
    plx_cg_template_free(&t->tmpl);
    memset(t, 0, sizeof *t);
}

void plx_fu_track_positions(const plx_fu_track *t, bool pop, double *x) {
    plx_lattice L8 = plx_lattice_make(PLX_FL_STEP);
    int N = plx_lat_nodes(&t->lat), dof = plx_lat_dof(&t->lat);
    for (int f = 0; f < t->F; f++)
        for (int n = 0; n < N; n++) {
            double px = plx_lat_x(&t->lat, n % t->lat.nx), py = plx_lat_y(&t->lat, n / t->lat.nx), v[2], ix, iy;
            plx_sim T;
            if (pop) {
                plx_lat_eval(&L8, t->u_pop + (size_t)f * plx_lat_dof(&L8), px, py, v);
                T = t->T_pop[f];
            } else {
                const float *u = t->u + (size_t)f * dof;
                v[0] = u[n], v[1] = u[N + n];
                T = t->T[f];
            }
            plx_sim_apply(T, px + v[0], py + v[1], &ix, &iy);
            x[((size_t)f * 2) * N + n] = ix;
            x[((size_t)f * 2 + 1) * N + n] = iy;
        }
}
