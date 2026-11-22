/* Dense flow: the smooth displacement field u_i from the canonical template to every photo, on
 * top of congealing's similarity (docs/DESIGN.md §4). Full warp
 *     W_i(p) = T_i(p + u_i(p)),   u_i a lattice field (align/lattice.h), canonical pixels.
 *
 * A round is congealing's loop with a richer warp. Per image the state is (T, ub, a): the photo
 * is RENDERED through ub, the increment a is SAMPLED on top of it, and the field is
 * u = a + ub o (id + a) (plx_lat_compose(a, ub)). A re-rendering stage first merges a into ub.
 *   1. RENDER: C_i(p) = I_i(T_i(p + ub_i(p))) on Ω+ (bilinear), orientation channels and pyramid
 *      (congeal.h's view); F_i(p) = that view sampled at p + a_i(p). F_i (and the gray image
 *      rendered through the full warp) is accumulated into the next template, symmetrised as in
 *      congealing.
 *   2. SOLVE an increment against the current template by inverse-compositional Gauss-Newton.
 *      Template-side field d with F_i(p) ≈ μ(p + d(p)) ≈ μ(p) + ∇μ(p)·d(p); residual
 *      r(p, k) = F_i(p)_k - μ_k(p); Jacobian J = ∇μ_k(p) ⊗ B(p) from TEMPLATE gradients, so
 *          H = Σ_{p,k} w(p,k) JᵀJ + (λ + λp) R + (δ + λ0) I
 *      is the same for every image: built and Cholesky-factored once per round and stage.
 *      The bending penalty acts on the TOTAL field, which becomes ≈ u - d, so minimising
 *          Σ w (r - J d)² + λ (u - d)ᵀ R (u - d) + λ0 |u - d|²
 *                         + λp (e - d)ᵀ R (e - d) + δ |e - d|²
 *      (e = a - a0, the increment so far this round; see PROXIMAL) gives
 *          H d = g + λ R u + λ0 u + λp R e + δ e,   g = Σ w Jᵀ ψ(r)
 *      (ψ: per-pixel Huber clipping of the residual vector; the shared H is not reweighted).
 *      If F(p) ≈ μ(p + d(p)) then F(p - d(p)) ≈ μ(p): the image-side increment is Δ = -d,
 *      composed at the nodes as  a(p_n) <- Δ(p_n) + a(p_n + Δ(p_n))  (plx_lat_compose).
 *      Inner iterations do not re-render: they sample at p + a(p), exactly as congealing
 *      samples F_i(S(p)).
 *   3. GAUGES after the round: the similarity part of u_i moves into T_i (ub is pulled back
 *      with it, which leaves the render warp and a unchanged); the population mean field,
 *      symmetrised (x odd, y even), is subtracted from every u_i (through a) so the template
 *      stays the mean shape; congealing's mean log-scale / mean ty gauge on T.
 *
 * WHY TWO KINDS OF STAGE. Re-rendering is what makes the result deformation-equivariant (the
 * photo's features are recomputed in the canonical frame at the current warp, so at the fixed
 * point a photo and a smoothly deformed copy of it give the same residual). But the orientation
 * channels respond to a displacement under one 8-px hat very differently from ∇F·B: the hat's
 * strain (ε/8) rotates and rescales the local gradients about as much as the shift moves them
 * (measured relative model error 0.8-2.7 for an 8-px hat; 0.35 for a 32-px hat; 0.14 for a
 * translation; the same without contrast normalisation). So round-to-round re-rendering is
 * well-behaved on the 32- and 16-px lattices and diverges on the 8-px one (on LFW: rms|u| grew
 * 4 -> 16 -> 67 px over three 8-px rounds). SAMPLED stages (rerender = 0) keep the last render
 * and refine only by sampling, where the Gauss-Newton model is exactly self-consistent; the
 * price is that the fine increment's own strain is not frontalised (small: it is the 8-px
 * detail on top of the 16-px solution).
 *
 * PROXIMAL TERMS. λp (bending) and δ (identity) penalise the increment of the current round
 * (e = a - a0). They vanish at the fixed point, so they change the speed, not the answer, and
 * they do not bias the affine part. δ is needed on real faces: from u = 0 the first Gauss-Newton
 * step against a blurred mean template is a 10-25 px smooth field (bending leaves smooth and
 * affine modes almost free, and identity differences push them), which then oscillates; δ = 1
 * bounds the per-round move. λp damps the high-frequency modes whose model is worst.
 *
 * WEIGHTS. w(p, k) = ω(p) / (σ²_k(p) + ε) (INVVAR, congealing's) or ω(p) (PLAIN). ω is
 * congealing's raised-cosine window on Ω. Samples are further scaled by v (fraction of the
 * sample's support that came from inside the photo) and the Huber factor, in g only.
 *
 * REGULARISATION SCALE. λ, λp, δ and λ0 are given relative to the data: the stiffness used is
 *     λ_eff = λ · tr(H_data) / tr(R)  (λp likewise),   λ0_eff = λ0 · tr(H_data) / dof  (δ likewise),
 * so one λ means the same balance at every pyramid level, lattice and weighting.
 *
 * SYMMETRY. Template, window, weights, lattice and R are σ-symmetric and the gauges commute
 * with σ, so the procedure is mirror-equivariant: fitting the flipped photo from twin(T) gives
 * twin(T') and the lattice twin of u' (up to float rounding). */
#ifndef PLX_ALIGN_FLOW_H
#define PLX_ALIGN_FLOW_H

#include "align/canon.h"
#include "align/congeal.h"
#include "align/lattice.h"
#include "la/la.h"

#include <stdbool.h>
#include <stdint.h>

#define PLX_FL_MAX_STAGES 32
#define PLX_FL_STEP 8 /* lattice of the output field: 17 x 21 nodes */

typedef enum { PLX_FL_W_INVVAR = 0, PLX_FL_W_PLAIN = 1 } plx_fl_weighting;

typedef struct {
    plx_cg_params    cg;        /* features, Ω+ margin, pyramid levels, interp, eps_frac, huber, ramp */
    plx_fl_weighting weighting;
    float            lambda0;   /* relative Tikhonov weight on the total field (0 = none) */
    float            damp;      /* relative Tikhonov weight on the round's increment (see PROXIMAL) */
    float            tol;       /* inner loop stops once max |Δ| over core nodes < tol px */
} plx_fl_params;
plx_fl_params plx_fl_default(void);

/* One round's solver setting. */
typedef struct {
    int   step;    /* lattice spacing, canonical px (8, 16, 32) */
    int   level;   /* pyramid level of the solve */
    float blur;    /* extra Gaussian blur (level px) of image features and template */
    float lambda;  /* relative bending weight on the total field */
    float prox;    /* relative bending weight on the round's increment (damping, see PROXIMAL) */
    int   iters;   /* inner Gauss-Newton iterations (no re-render) */
    int   rerender; /* 1: render through the whole field this round; 0: keep rendering through the
                     * field as of the last re-rendering round and refine by sampling (SAMPLED) */
} plx_fl_stage;

/* ---- rendering ------------------------------------------------------------------------ */
/* View of `gray` through W(p) = T(p + u(p)), u on lattice L (clamped outside it). */
void plx_fl_render(const plx_img *gray, plx_sim T, const plx_lattice *L, const float *u, const plx_cg_params *p,
                   plx_cg_view *v);
/* out (Ω+ with `margin`, gray's channel count) = gray through W(p) = T(p + u(p)), u on lattice
 * L (clamped outside it); valid (optional, 1 channel) as in plx_warp_map. */
void plx_fl_render_gray(const plx_img *gray, plx_sim T, const plx_lattice *L, const float *u, int margin,
                        plx_interp interp, plx_img *out, plx_img *valid);
/* The Ω (128 x 160) gray image through W, bilinear: the "frontalised" crop. */
plx_img plx_fl_warp_crop(const plx_img *gray, plx_sim T, const plx_lattice *L, const float *u);

/* ---- reference: the template prepared for one stage (shared by all images, read-only) ---- */
typedef struct {
    plx_fl_stage st;
    plx_lattice  lat;
    int          K, w, h, n;         /* channels, level Ω+ size, sample count */
    double       eps, omega_sum;     /* variance floor; Σ ω */
    double       lambda_eff, prox_eff, lambda0_eff, damp_eff;
    float       *qx, *qy;            /* [n] level-L Ω+ pixel of each sample */
    plx_lat_taps *taps;              /* [n] lattice hat functions at the sample */
    float       *omega;              /* [n] */
    float       *mu, *inv;           /* [n*K] template mean, 1 / (σ² + ε) */
    float       *g;                  /* [n*K*2] w_k ∇μ_k in full-resolution units */
    double      *hdata;              /* [dof] diagonal of the data Hessian */
    plx_mat      R;                  /* bending matrix (dof x dof) */
    plx_mat      chol;               /* Cholesky factor of H */
} plx_fl_ref;
plx_fl_ref plx_fl_ref_build(const plx_cg_template *t, const plx_fl_params *p, const plx_fl_stage *st);
void       plx_fl_ref_free(plx_fl_ref *r);

/* Per-node sums of the robust cost, for a population reliability map (optional in solve):
 * cost[n] += Σ_p B_n(p) ω v e²,  mass[n] += Σ_p B_n(p) ω v at the first iteration. */
typedef struct { double *cost, *mass; } plx_fl_nodestat;

typedef struct {
    double cost0;     /* Σ ω v e² / Σ ω v at the start (e² = Σ_k r²/(σ²+ε), ≈ K when typical) */
    double cost;      /* the same at the last evaluated field */
    double inlier;    /* Σ ω v h / Σ ω v at the last evaluation */
    double valid;     /* Σ ω v / Σ ω */
    double step;      /* max |Δ| over core nodes of the last increment, px */
    int    iters;
    bool   converged; /* last increment < tol before the cap */
} plx_fl_fit;
/* Inner iterations against r for view v rendered with field u (on r->lat), starting from the
 * image-side increment a (sampled on top of v: the total field is plx_lat_compose(a, u)),
 * which is updated in place; the proximal terms anchor at a's value on entry. ns (optional)
 * receives per-node cost sums at the start. */
plx_fl_fit plx_fl_solve(const plx_fl_ref *r, const plx_cg_view *v, const plx_fl_params *p, const float *u,
                        float *a, const plx_fl_nodestat *ns);

/* Per-node reliability of a stage's solution: the share of the node's stiffness that comes
 * from data, hdata / (hdata + λ_eff R_nn + λ0_eff), averaged over the two components, then
 * symmetrised. In [0, 1]; ~0 where the field is only the regulariser's extrapolation. */
void plx_fl_data_fraction(const plx_fl_ref *r, float *w);

/* ---- the frozen model and single-image fitting ------------------------------------------ */
typedef struct {
    plx_fl_params   p;
    int             n_stages;
    plx_fl_stage    stages[PLX_FL_MAX_STAGES]; /* the population schedule; last step must be 8 */
    plx_cg_template tmpl;                      /* final template, all pyramid levels */
} plx_fl_model;
/* Directory layout: model.txt (parameters and schedule, key value lines) and
 * mean_L<l>.plxt / var_L<l>.plxt f32 [K, h, w] per pyramid level (Ω+, margin p.cg.margin). */
int  plx_fl_model_save(const plx_fl_model *m, const char *dir);
int  plx_fl_model_load(const char *dir, plx_fl_model *m);
void plx_fl_model_free(plx_fl_model *m);

typedef struct {
    const plx_fl_model *m;
    plx_fl_ref         *refs; /* one per stage, built once */
} plx_fl_fitter;
void plx_fl_fitter_init(plx_fl_fitter *f, const plx_fl_model *m);
void plx_fl_fitter_free(plx_fl_fitter *f);
/* Fits one gray image (any size; LFW crops are 250 x 250) from an initial similarity with
 * u = 0, through the model's whole schedule (render, solve, similarity gauge per stage, refining
 * the lattice as the schedule does). Out: T, u on the 8-px lattice (2 * 357 floats) and the
 * last stage's fit. Thread-safe (the fitter is read-only). */
plx_fl_fit plx_fl_fit_image(const plx_fl_fitter *f, const plx_img *gray, plx_sim T0, plx_sim *T, float *u);

/* ---- population ------------------------------------------------------------------------- */
typedef struct {
    double  seconds;
    double  core_var, core_ratio; /* plx_cg_core_variance of the template this round produced */
    double  mean_cost;            /* mean fit cost0 over the population (align rounds) */
    double  median_step;          /* median over images of the max node update of the round, px */
    double  mean_rms_u;           /* RMS |u| over core nodes after the gauges, population mean */
    int     n_unconverged;
    plx_img mean_gray;            /* Ω mean canonical gray face of this round's population */
} plx_fl_round;

typedef struct {
    float   cost, inlier, valid;  /* evaluation solve against the final template, last stage */
    float   step;                 /* px a fresh solve would still move the field (max over core nodes) */
    float   min_jdet;             /* plx_lat_min_jdet of the final field */
    float   max_u;                /* plx_lat_max_abs of the final field */
    uint8_t converged;            /* last alignment round converged within its cap */
} plx_fl_quality;

/* Round 0 accumulates at (T, u = 0); rounds 1..n_stages accumulate and align with sched[r-1];
 * round n_stages + 1 accumulates the final template; then an evaluation pass fills q and node
 * cost statistics (ns, optional, on the 8-px lattice). trusted (optional, [n]): only these images
 * enter the template and the mean-shape gauge (all are fitted). T [n] in/out; u [n * 714] out
 * (8-px lattice); rounds [n_stages + 2] (caller frees mean_gray); q [n]. */
void plx_fl_congeal(const plx_cg_source *src, const plx_fl_params *p, const plx_fl_stage *sched, int n_stages,
                    const uint8_t *trusted, plx_sim *T, float *u, plx_cg_template *final_template,
                    plx_fl_round *rounds, plx_fl_quality *q, const plx_fl_nodestat *ns);

#endif
