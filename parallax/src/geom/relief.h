/* Depth relief: the population-shared, full-resolution refinement of the mean depth pattern.
 *
 * WHY. emerge (geom.h) reads depth from the per-photo lattice flows, and a per-photo flow must be
 * stiff (one photo is weak evidence and would otherwise chase beards and glasses), so the depth
 * pattern v it yields is a smooth dome: the nose sliding against the cheeks is below its
 * resolution. But the depth pattern is SHARED by the population, and every photo's yaw parallax
 * s_i is now known, so the fine depth at a canonical pixel is a regression coefficient across
 * 13,000 photos, which needs almost no spatial regularisation.
 *
 * MODEL. One dense layer on top of every photo's warp (align/flow.h, plx_fl_render_relief):
 *     W_i(p) = T_i( p + u_i(p) + s_i · w(p) · x̂ ),
 * w a scalar map on Ω (128 x 160), mirror-even, in the units of v, so the refined pattern is
 * v_hr = v + w (v bilinear from the lattice, exactly as the flow evaluates it) and metric depth is
 * κ v_hr. T_i, u_i, s_i are held fixed.
 *
 * ONE ITERATION (one pass over the used photos):
 *   1. render C_i = I_i o W_i on Ω+ with the current w, orientation channels F_i; accumulate the
 *      next template (mean μ, variance σ², mirror-symmetrised exactly as congealing does);
 *   2. residual against the current template r_ik = F_ik - μ_k;
 *   3. Lucas-Kanade with the population as the window: we want Δw with C_i(p + s_i Δw x̂) ≈ μ(p).
 *      With b_ik = s_i ∂x μ_k (see THE GRADIENT below) and ω_k = 1/(σ²_k + ε) (times congealing's
 *      window and the validity of the sample)
 *          N(p) = Σ_i Σ_k ω_k b_ik ψ(r_ik),   D(p) = Σ_i Σ_k ω_k b_ik²,
 *      ψ the Huber clipping of the pixel's normalised residual vector (flow.h's rule);
 *   4. mirror twins. The twin of photo i is the flipped photo with s -> -s; its features are
 *      F'_k(p) = F_π(k)(σp) (π the orientation-channel permutation) and the template is symmetric,
 *      μ_k(p) = μ_π(k)(σp), so ∂x μ_k(p) = -∂x μ_π(k)(σp), b'_k(p) = (-s)(-∂x μ_π(k)(σp)) =
 *      b_π(k)(σp), r'_k(p) = r_π(k)(σp), ω_k(p) = ω_π(k)(σp); the Huber factor depends on the sum
 *      over k only. Summing over k absorbs π: N'(p) = N(σp), D'(p) = D(σp) (the same holds with the
 *      photo's own gradient). So N + N∘σ and D + D∘σ are used: w is exactly even, and every
 *      appearance effect that is even in yaw cancels in N;
 *   5. normalised convolution (the LK window, a few px wide) with the penalty on the TOTAL layer,
 *      min Σ (r + b Δw)² + λ (w + Δw)²:
 *          Δw = -(G_σ * N + λ w) / ((G_σ * D) + λ),     w <- w + η Δw,
 *      λ = lambda · mean of G_σ * D over the core. At the fixed point w = shrunk regression
 *      coefficient, so w falls to zero where nothing is measured. (λ on the increment alone,
 *      -G*N / (G*D + λ), only slows the iteration: its fixed point is G*N = 0 for any λ.)
 *   6. gauge: the depth part of w is w minus its projection on span{1, Y0, v}, least squares over
 *      the core with weights G_σ * D (plx_rl_gauge): 1 and Y0 are v's plane gauge (a translation /
 *      shear of each photo proportional to its yaw), v keeps <v_hr, v> = <v, v>, so κ and the yaw
 *      angles stay valid. It is NOT imposed on the warp layer inside the loop: with every s_i held
 *      fixed the data determine the product s_i (v + w) completely, including its components
 *      along 1, Y0 and v, and projecting them out after every step leaves a misfit the next pass
 *      asks for again (measured: steps never shrank; on LFW the gauge removed ~0.2 px of constant
 *      and 3% of v per pass, forever). The scale CAN trade only when s moves (refinement), and
 *      that is where the scale gauge is applied (plx_relief: the common rescaling of the s steps
 *      is removed). The warp layer w therefore renders exactly what was fitted; depth outputs use
 *      the gauged w and report the removed coefficients;
 *   7. coarse to fine: each stage sets the feature pyramid level and the window σ.
 * Each pass also accumulates Q(p) = Σ_i (G_σ * m_i)², m_i = n_i + n_i∘σ the photo's (symmetrised)
 * contribution to N: sqrt(Q) / (G*D + λ) is the sampling standard error of a regression step
 * (photos as the independent units), the denominator of the z-score map.
 *
 * THE GRADIENT. Write photo i's features as F_i(p) = A_i(p + s_i δ(p)), A_i = shared face +
 * identity, μ ≈ E[A]. Then r_i ≈ (A_i - μ) + s_i δ ∂x A_i, and with the photo's OWN gradient,
 * b = s ∂x F_i, the windowed expectation is (integrating the second-order term by parts)
 *     E[N] ≈ δ Σ s² ∂x F_i ∂x μ,    while   D = Σ s² (∂x F_i)²,
 * so the own-gradient step is attenuated by (shared gradient energy) / (total gradient energy):
 * identity texture inflates D and adds noise to N, but carries no information about δ, which is
 * uncorrelated with identity. On faces that share is small and the iteration crawls (synthetic
 * face textures: 8 passes recovered corr 0.78, high-pass 0.36). The full-Newton normaliser
 * Σ s² ∂x F_i ∂x μ is calibrated but not positive and diverged. Since E[∂x F_i] = ∂x μ, the
 * template gradient is the noise-free instrument: b = s ∂x μ in both N and D (population
 * inverse compositional). Measured on the same synthetic data: corr 0.96, high-pass 0.76, half
 * the permutation noise; on 3000 LFW photos split-half corr 0.986 (high-pass 0.979) vs 0.947
 * (0.920). The worry that the template's edge is blurred by the misalignment being measured is
 * real but self-correcting: the template is re-accumulated every pass and sharpens as w improves.
 *
 * Yaw refinement (optional, after w is stable): one Gauss-Newton step per photo on its own s
 * against the sharper model, Δs_i = -Σ ω (v_hr ∂x μ) ψ(r_i) / Σ ω (v_hr ∂x μ)² (the same
 * instrument). The refined warp is T(p + u + Δs v + s' w): the change along v is a lattice field,
 * carried by plx_rl_data.s_shift. */
#ifndef PLX_GEOM_RELIEF_H
#define PLX_GEOM_RELIEF_H

#include "align/congeal.h"
#include "align/lattice.h"
#include "geom/geom.h"

#define PLX_RL_MAX_STAGES 8
#define PLX_RL_MAX_ITERS 64
#define PLX_RL_LAT_NODES (17 * 21) /* the flow's 8-px lattice */

typedef struct {
    int   level; /* feature pyramid level of the regression */
    float sigma; /* LK window, full-resolution px */
    int   iters; /* passes */
} plx_rl_stage;

typedef struct {
    plx_cg_params  cg;     /* features, Ω+ margin, levels, eps_frac, huber, ramp: the flow model's */
    int            n_stages;
    plx_rl_stage   stages[PLX_RL_MAX_STAGES];
    float          lambda; /* relative regularisation (step 5) */
    float          prior_sigma; /* > 0: shrink towards w's support-weighted blur at this σ (px), not 0 */
    float          eta;    /* damping of each update, <= 1 */
    plx_core_shape core;   /* gauge and statistics region (geom.h's ellipse) */
} plx_rl_params;
/* From the flow model's parameters: schedule 1:4px x 2, 0:2px x 3, 0:1.5px x 3, lambda 0.1,
 * eta 0.8, the default core ellipse. Chosen by measurement (apps/plx_relief.c's summary has the
 * sweep): on synthetic known relief λ 0.03 is noisier (permutation rms +50%) and λ 0.3 biased
 * (corr 0.92 vs 0.96); η = 1 oscillates; on 3000 LFW photos a final window of 1.5 px keeps
 * lowering the population feature variance against 2 px at split-half high-pass agreement 0.979
 * vs 0.981, while 1 px had not converged in two passes. */
plx_rl_params plx_rl_default(const plx_cg_params *cg);

typedef struct {
    const plx_cg_source *src;     /* gray photos */
    const plx_sim       *T;       /* [n] */
    const float         *u;       /* [n * 714] flow fields, 8-px lattice */
    const float         *s;       /* [n] yaw parallax: the regressor and the amplitude of w, px */
    const float         *s_shift; /* [n] nullable: u_i + s_shift_i · v (x component), see refinement */
    const uint8_t       *use;     /* [n] nullable: photos in the template and the regression */
    const float         *v_lat;   /* [357] the depth pattern v at the lattice nodes (unit-free) */
} plx_rl_data;

typedef struct {
    int    stage, level;
    float  sigma;
    double seconds;
    double core_var, core_ratio, sharpness; /* of the template accumulated in this pass (w before the update) */
    double cost, inlier;                    /* mean e² = Σ_k r²/(σ²+ε) and Huber weight over the samples */
    double lambda_abs;                      /* λ in units of D */
    double step_rms, step_max, w_rms;       /* η Δw and w over the core, units of v */
    double gauge[3];                        /* (α, γ, c): w's projection on 1, Y0, v (step 6) */
} plx_rl_iter;

typedef struct {
    plx_img         w;          /* Ω, units of v: the warp layer (ungauged; plx_rl_gauge gives its depth part) */
    plx_img         support;    /* G * (D + D∘σ) of the last iteration, upsampled to Ω */
    plx_img         se;         /* sqrt(Q) / (G*D + λ) of the last iteration, Ω */
    plx_cg_template tmpl;       /* the last accumulated template */
    plx_img         mean_gray;  /* its Ω mean gray face (not symmetrised) */
    double          core_var, core_ratio, sharpness; /* of that template / face */
    int             n_iter;
    plx_rl_iter     iter[PLX_RL_MAX_ITERS];
} plx_rl_result;
void plx_rl_result_free(plx_rl_result *r);

/* Runs the schedule. init (nullable): the template to regress against in the first pass (the
 * flow model's: it was accumulated with the same T, u at w = 0); NULL first runs an
 * accumulation-only pass. w0 (nullable): starting map (zero otherwise). final_accumulate: one
 * more accumulation-only pass with the final w, so tmpl / mean_gray describe the result. */
void plx_rl_estimate(const plx_rl_data *d, const plx_rl_params *p, const plx_cg_template *init, const plx_img *w0,
                     bool final_accumulate, plx_rl_result *out);

/* Yaw refinement: the undamped Gauss-Newton step ds[i] of every used photo (0 for the others)
 * against template t (accumulated at w), at pyramid level `level`. */
void plx_rl_refine_yaw(const plx_rl_data *d, const plx_rl_params *p, const plx_cg_template *t, const plx_img *w,
                       int level, float *ds);

/* ---- helpers shared by the app and the tests ------------------------------------------------- */
/* v on Ω (128 x 160) from the lattice nodes, bilinear as the flow evaluates fields. */
plx_img plx_rl_v_dense(const float *v_lat);
/* Canonical pixel (x, y) inside the core ellipse. */
bool plx_rl_in_core(plx_core_shape cs, int x, int y);
/* Step 6: removes span{1, Y0, v} from w (all pixels) by least squares over the core with weights
 * wt (Ω, nullable = 1); coef (nullable) receives (α, γ, c). */
void plx_rl_gauge(plx_img *w, const plx_img *v, const plx_img *wt, plx_core_shape cs, double coef[3]);
/* Agreement of two Ω maps over the core: Pearson correlation, the same after high-pass filtering
 * (each minus its normalised Gaussian blur at hp_sigma inside the core), and the RMS of a - b.
 * Any output nullable. */
void plx_rl_compare(const plx_img *a, const plx_img *b, plx_core_shape cs, float hp_sigma, double *corr,
                    double *corr_hp, double *rms);

/* ---- synthetic population with known relief (acceptance test 1) ------------------------------
 * One rigid surface z_true = plx_synth_depth (half-ellipsoid dome + nose + eye-socket dents + brow,
 * lips, chin), rendered by the z-buffer renderer at random yaw θ_i with a different texture per
 * photo, unit similarity, so that photo pixel (X0 cosθ + z sinθ + 124.5, Y0 + 128.5) shows the
 * texture at canonical p. The algorithm gets the smooth dome only: v_lat = the half-ellipsoid at the
 * lattice nodes, s_i = sin θ_i (κ = 1: depth in px), u_i = ((cosθ - 1) X0 + s_i v, 0) at the nodes
 * (foreshortening exact), so the true layer is w_true = z_true - v (bilinear). */
typedef struct {
    int             n;
    double          yaw_sd_deg, yaw_clip_deg;
    double          nose_px;    /* plx_synth_depth's nose amplitude */
    double          s_noise;    /* noise on the s_i handed to the estimator, fraction of sd(s) */
    double          u_noise_px; /* smooth lattice error on the u_i handed over (not in the photos) */
    double          tex_indiv;  /* random textures: per-photo noise relative to the shared pattern */
    const plx_img  *tex;        /* nullable: Ω+ textures (margin PLX_CANON_MARGIN), photo i uses tex[i % n_tex] */
    int             n_tex;
    uint64_t        seed;
} plx_rl_synth_opts;
plx_rl_synth_opts plx_rl_synth_defaults(void); /* n 600, yaw sd 12 clip 35, nose 20, no noise, tex_indiv 1 */

typedef struct {
    int           n;
    uint8_t      *px;       /* [n][250][250] */
    plx_cg_source src;
    plx_sim      *T;
    float        *u;        /* [n * 714] */
    float        *s, *s_true;
    float         v_lat[PLX_RL_LAT_NODES];
    plx_img       w_true;   /* Ω, before any gauge */
} plx_rl_synth;
void plx_rl_synth_make(const plx_rl_synth_opts *o, plx_rl_synth *out);
void plx_rl_synth_free(plx_rl_synth *s);

#endif
