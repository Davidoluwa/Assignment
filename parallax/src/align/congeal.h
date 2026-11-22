/* Similarity congealing: joint alignment of a population of face crops to a template that is
 * learnt from them (docs/DESIGN.md §4), with no labels.
 *
 * Unknowns: one image similarity T_i per image (align/canon.h) and the template. A round is one
 * pass over the population; per image it
 *   1. RE-RENDERS: warps the gray image into Ω+ through T_i and computes orientation channels
 *      there (features are never warped, so scale and roll are always handled exactly);
 *   2. accumulates those features into the statistics of the NEXT template;
 *   3. solves a small canonical similarity S against the CURRENT template by inverse-
 *      compositional Gauss-Newton at one pyramid level, and sets T_i <- T_i o S.
 * After the pass the new template is symmetrised (mean and second moment averaged with their
 * mirror images, equivalent to adding every image's mirror twin) and the global gauge is fixed
 * (mean log scale = 0, mean ty = 0).
 *
 * GAUSS-NEWTON. Incremental warp S_δ(p) = c0 + [1+δ1 -δ2; δ2 1+δ1](p - c0) + (δ3, δ4), in the
 * coordinates of the solve level. Residual r(p, k) = F_k(S(p)) - μ_k(p); steepest-descent
 * images J(p, k) = ∇μ_k(p) · dS_δ/dδ from TEMPLATE gradients (so the per-pixel normal matrices
 * are precomputed once per round and shared by all images). Step Δ = (Σ w JᵀJ)⁻¹ Σ w Jᵀ r,
 * update S <- S o S_Δ⁻¹. (Note the sign: with r defined as image minus template and the
 * inverse update, the step is +H⁻¹Jᵀr.) Weights
 *     w(p, k) = ω(p) · v(p) · h(p) / (σ²_k(p) + ε)
 * with σ² the population variance of the feature, ε = eps_frac · median σ² over the window,
 * ω a raised-cosine window that is 0 on the Ω border, v the fraction of the sample's support
 * that came from inside the source photo (0 outside Ω+), and h the Huber IRLS weight of the
 * pixel's normalised residual norm e(p) = sqrt(Σ_k r²/(σ²_k + ε)) at threshold huber·sqrt(K).
 *
 * TEMPLATE LAYOUT. Template mean / variance are kept on Ω+ (margin p->margin) at every pyramid
 * level of Ω+; level 0 is (128 + 2m) x (160 + 2m). Level-L pixel q relates to level 0 by the
 * ops.h pyramid registration q_0 = 2^L (q_L + 0.5) - 0.5, and Ω+ pixel q_0 is canonical point
 * q_0 - margin. */
#ifndef PLX_ALIGN_CONGEAL_H
#define PLX_ALIGN_CONGEAL_H

#include "align/canon.h"
#include "feat/feat.h"
#include "ops/ops.h"

#include <stdbool.h>
#include <stdint.h>

#define PLX_CG_MAX_LEVELS 4

/* Face-core box in canonical pixels (the raw LFW mean's face core, docs/DESIGN.md §2), inclusive.
 * σ-symmetric: 127 - 24 = 103 lies inside, 127 - 104 = 23 just outside, so treat it as
 * [23.5, 103.5] when symmetry matters (the flow lattice's core nodes do). */
#define PLX_CG_CORE_X0 24
#define PLX_CG_CORE_X1 104
#define PLX_CG_CORE_Y0 21
#define PLX_CG_CORE_Y1 136

typedef struct {
    plx_orient_params feat;
    int        margin;     /* Ω+ margin, full-resolution pixels */
    int        levels;     /* pyramid levels kept for the template (1..PLX_CG_MAX_LEVELS) */
    plx_interp interp;     /* gray warp into Ω+ */
    float      eps_frac;   /* ε = eps_frac * median feature variance in the window */
    float      huber;      /* Huber threshold on e(p), in units of sqrt(K) */
    float      ramp;       /* width of the window's raised-cosine ramp, full-res pixels */
    float      tol;        /* convergence: GN step displacement below tol level pixels */
    double     scale_min, scale_max, rot_max /* radians */, shift_max /* image px, |tx|, |ty| */;
} plx_cg_params;
plx_cg_params plx_cg_default(void);

/* One alignment round's solver setting. */
typedef struct {
    int   level;  /* pyramid level the solve runs at (0 = full resolution) */
    float blur;   /* extra Gaussian blur, level pixels, of image features AND template */
    int   iters;  /* Gauss-Newton iteration cap */
    float prior;  /* damping of S towards identity, relative to the Hessian diagonal (0 = none) */
} plx_cg_stage;

/* ---- rendering one image into the canonical frame ------------------------------------ */
typedef struct {
    plx_img     gray, valid; /* Ω+, full resolution; valid = 1 where the photo was sampled */
    plx_pyramid feat, vpyr;  /* orientation-channel and validity pyramids of Ω+ */
} plx_cg_view;
void plx_cg_render(const plx_img *gray, plx_sim T, const plx_cg_params *p, plx_cg_view *v);
/* Builds the pyramids of a view whose gray and valid Ω+ images (full resolution, margin
 * p->margin) the caller has already rendered by some other warp. */
void plx_cg_view_finish(const plx_cg_params *p, plx_cg_view *v);
void plx_cg_view_free(plx_cg_view *v);

/* ---- template -------------------------------------------------------------------------- */
typedef struct {
    int     K, margin, levels;
    int64_t count;                       /* images accumulated (mirror twins implied) */
    plx_img mean[PLX_CG_MAX_LEVELS];     /* K channels on Ω+ at level L, σ-symmetric */
    plx_img var[PLX_CG_MAX_LEVELS];      /* population variance, same layout */
} plx_cg_template;
void plx_cg_template_free(plx_cg_template *t);

/* Population sums for the next template (and the mean canonical gray face). Use one per thread
 * and merge. */
typedef struct {
    int     K, margin, levels;
    int64_t n;
    double *s1[PLX_CG_MAX_LEVELS], *s2[PLX_CG_MAX_LEVELS];
    double *gray; /* Ω+ level 0 */
} plx_cg_accum;
void plx_cg_accum_init(plx_cg_accum *a, const plx_cg_params *p);
void plx_cg_accum_add(plx_cg_accum *a, const plx_cg_view *v);
void plx_cg_accum_merge(plx_cg_accum *dst, const plx_cg_accum *src);
/* Mean, symmetrised mean and second moment, variance. mean_gray (optional): the Ω (not Ω+)
 * mean of the canonical gray images, NOT symmetrised (its asymmetry is a diagnostic). */
void plx_cg_accum_finish(const plx_cg_accum *a, const plx_cg_params *p, plx_cg_template *t, plx_img *mean_gray);
void plx_cg_accum_free(plx_cg_accum *a);

/* Face-core statistics of a template at level 0: total variance Σσ² over the core box
 * (canonical x in [24, 104], y in [21, 136], all channels) and that variance as a fraction of
 * the total second moment Σ(σ² + μ²) there. */
void plx_cg_core_variance(const plx_cg_template *t, double *total, double *ratio);

/* ---- solver building blocks shared with the flow stage --------------------------------- */
/* The raised-cosine window ω at canonical point (x, y): 0 on and beyond the Ω border, 1 once
 * `ramp` pixels inside; σ-symmetric. */
double plx_cg_window(double x, double y, double ramp);
/* ω evaluated at the centre of every pixel of the level-`level` Ω+ grid (w x h); caller frees. */
float *plx_cg_level_window(int w, int h, int level, const plx_cg_params *p);
/* The variance floor ε = eps_frac * median of var (a level-`level` Ω+ image, all channels) over
 * the pixels where ω > 0.5. */
double plx_cg_var_floor(const plx_img *var, int level, const plx_cg_params *p);
/* Common canonical-side correction G (T_i <- T_i o G for all i) so that mean log scale = 0 and
 * mean ty = 0. */
void plx_cg_gauge_fix(plx_sim *T, int n);
/* Mean squared gradient (Scharr) of an Ω gray face over the face-core box, x 1e4: the sharpness
 * of a population mean face, independent of the features the alignment optimises. */
double plx_cg_sharpness(const plx_img *face);

/* ---- per-image solve ------------------------------------------------------------------- */
/* The template prepared for one stage: blurred mean, gradients, weights, and the precomputed
 * per-sample quantities of the GN normal equations. Read-only once built (shared by threads). */
typedef struct {
    int    level, K, w, h, n;  /* solve level, channels, level Ω+ size, sample count */
    float  blur;
    double cx, cy;             /* c0 in level-L Ω+ pixel coordinates */
    double eps, omega_sum;     /* variance floor; Σ ω over samples */
    double rmax;               /* largest |p - c0| over samples (level px) */
    float *X, *Y;              /* [n] sample positions relative to c0 */
    float *omega;              /* [n] window */
    float *mu, *inv;           /* [n*K] template mean, 1 / (σ² + ε) */
    float *g;                  /* [n*K*4] w_k J_k (without h, v) */
    float *hp;                 /* [n*10] Σ_k w_k J_kᵀJ_k, upper triangle row-major */
} plx_cg_ref;
plx_cg_ref plx_cg_ref_build(const plx_cg_template *t, const plx_cg_params *p, int level, float blur);
void       plx_cg_ref_free(plx_cg_ref *r);

typedef struct {
    double cost0;     /* Σ ω v e² / Σ ω v at the start S */
    double cost;      /* the same at the last evaluated S (≈ K for a typical fit) */
    double inlier;    /* Σ ω v h / Σ ω v: Huber weight mass */
    double valid;     /* Σ ω v / Σ ω: window fraction backed by real photo pixels */
    double step;      /* displacement of the last GN step, full-res px (max over the window) */
    int    iters;
    bool   converged; /* last step < tol before the cap */
} plx_cg_fit;
/* Solves for the canonical similarity S (full-resolution units; in: start, usually identity;
 * out: result) aligning view v to the reference: F(S(p)) ~ μ(p). */
plx_cg_fit plx_cg_solve(const plx_cg_ref *r, const plx_cg_view *v, const plx_cg_params *p,
                        const plx_cg_stage *st, plx_sim *S);

/* Max displacement |S(p) - p| over the face-core box corners, canonical px. */
double plx_cg_csim_disp(plx_sim S);
/* Clamp T into the parameter bounds; true if anything was clipped. */
bool   plx_cg_clamp(plx_sim *T, const plx_cg_params *p);

/* One image through a schedule against fixed per-stage references (refs[s] built for
 * stages[s]): re-render, solve, compose, clamp, per stage. Returns the final T; `last` gets the
 * last stage's fit and *bound whether any stage clipped (both optional). */
plx_sim plx_cg_align_image(const plx_img *gray, plx_sim T0, const plx_cg_ref *refs, const plx_cg_stage *stages,
                           int n_stages, const plx_cg_params *p, plx_cg_fit *last, bool *bound);

/* ---- population ------------------------------------------------------------------------ */
typedef struct { const uint8_t *px; int n, w, h; } plx_cg_source; /* u8 gray [n][h][w], /255 */
plx_img plx_cg_source_image(const plx_cg_source *s, int i);

typedef struct {
    double  seconds;
    double  core_var, core_ratio; /* plx_cg_core_variance of the template this round produced */
    double  mean_cost;            /* mean fit cost over the population (align rounds) */
    double  median_disp;          /* median plx_cg_csim_disp of this round's updates */
    int     n_unconverged, n_bound;
    plx_img mean_gray;            /* Ω mean canonical gray face of this round's population */
} plx_cg_round;

/* Per-image quality from the final evaluation against the final template (level 0). */
typedef struct {
    float   cost, inlier, valid;
    float   disp;      /* displacement (px) a fresh solve vs the final template would still apply */
    uint8_t converged; /* last alignment round converged within its cap */
    uint8_t bound;     /* clipped by a bound in any round */
} plx_cg_quality;

/* Round 0 accumulates at T (pass identities for a fresh start); rounds 1..n_stages accumulate and
 * align with sched[r-1]; round n_stages+1 accumulates the final template at the final T; then an
 * evaluation pass fills q. T [n] in/out. rounds [n_stages + 2] (caller frees mean_gray), q [n]. */
void plx_cg_congeal(const plx_cg_source *src, const plx_cg_params *p, const plx_cg_stage *sched, int n_stages,
                    plx_sim *T, plx_cg_template *final_template, plx_cg_round *rounds, plx_cg_quality *q);

#endif
