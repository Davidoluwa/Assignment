/* Mirror-parallax geometry: depth, pose, metric scale and shape bases emerging from flow fields.
 *
 * Input is a population of lattice displacement fields u_i (DESIGN.md §2, §5): f32 [N, 2, rows, cols],
 * component x/y, canonical pixels, already gauged by the flow stage (per-image similarity removed;
 * mirror-symmetrised population mean subtracted). Everything here is unsupervised: identity labels
 * are never used for fitting.
 *
 * NOTATION. ⟨f, g⟩ = Σ_n ω_n f_n g_n over all lattice nodes, ω = node_weight × core mask
 * (mirror-symmetric). Mirror partner of (row, col) is (row, cols - 1 - col). Fields are split per
 * node into E = even(u_x), O = odd(u_x), V = even(u_y), Q = odd(u_y).
 *
 * COORDINATES. Even fields are stored as y_k = sqrt(m_k ω_k) f(node_k) over the core nodes of the
 * right half-lattice (col >= axis), m = 2 off the axis and 1 on it; odd fields likewise over the
 * off-axis core nodes. Then ⟨f, g⟩ over the full lattice is the plain dot product y_f · y_g, so
 * every weighted least-squares / SVD / covariance below is an ordinary Euclidean one.
 *
 * UNITS. v is dimensionless (unit ω-RMS over the core). s_i, t_i are pixels (yaw / pitch parallax
 * amplitude). κ is pixels: z̄ = κ v, sin θ_i = s_i / κ, sin φ_i cos θ_i = t_i / κ. The depth basis B
 * is in units of v (personal depth ẑ_i = κ (v + B ĉ_i)). Pitch is relative to the population-mean
 * pitch (the flow stage's mean subtraction defines it; see emerge.c). */
#ifndef PLX_GEOM_H
#define PLX_GEOM_H

#include "core/core.h"

/* ---- lattice ------------------------------------------------------------------------------ */
typedef struct {
    int    cols, rows; /* cols odd: the centre column lies on the mirror axis */
    double spacing;    /* canonical px between nodes */
    double x0, y0;     /* canonical position of node (row 0, col 0) */
    double cx, cy;     /* centre c0: X0 = x - cx, Y0 = y - cy; the lattice must be mirror-symmetric about cx */
} plx_glat;

plx_glat plx_glat_flow(void); /* the flow stage's: 17 x 21, 8 px, origin (-0.5, -0.5), c0 (63.5, 79.5) */
static inline int plx_glat_n(const plx_glat *L) { return L->cols * L->rows; }
double plx_glat_X0(const plx_glat *L, int node);
double plx_glat_Y0(const plx_glat *L, int node);
int    plx_glat_mirror(const plx_glat *L, int node);

/* Core region: the ellipse (X0 / ax)² + ((Y0 - yc) / ay)² <= 1 (inner face; outer nodes sit on hair,
 * background and the occluding contour). Mirror-symmetric by construction. */
typedef struct { double ax, ay, yc; } plx_core_shape;
plx_core_shape plx_core_default(void); /* ax 40, ay 50, yc 6 (canonical eyes ~ Y0 -16, chin ~ +56) */
void plx_core_mask(const plx_glat *L, plx_core_shape c, uint8_t *mask); /* mask[P] */
/* Face-centre region used for the convexity sign: |X0| <= 0.3 ax and |Y0 - yc| <= 0.4 ay. */
bool plx_core_centre(const plx_glat *L, plx_core_shape c, int node);

/* Parity split of one field (full lattice, P doubles per component). Any output may be NULL. */
void plx_parity_split(const plx_glat *L, const double *ux, const double *uy, double *E, double *O, double *V,
                      double *Q);

/* ---- weighted coordinate space -------------------------------------------------------------- */
typedef struct {
    plx_glat lat;
    int     P;
    double *omega;    /* [P] node_weight × core */
    double  wsum;     /* Σ ω over the full lattice */
    int     ne, no;   /* even / odd coordinate counts */
    int    *enode;    /* [ne] full-lattice node of each even coordinate (col >= axis) */
    int    *onode;    /* [no] (col > axis) */
    double *esw, *osw;/* sqrt(m ω) per coordinate */
    double *e1, *eY;  /* [ne] coordinates of the even fields 1 and Y0 */
    double *oX;       /* [no] coordinates of the odd field X0 */
} plx_gspace;

/* node_weight nullable (= 1). Nodes with ω = 0 are dropped from the coordinates. */
void plx_gspace_init(plx_gspace *sp, const plx_glat *L, const float *node_weight, const uint8_t *core);
void plx_gspace_free(plx_gspace *sp);
/* Coordinates of the even / odd part of a full-lattice field. */
void plx_gspace_even(const plx_gspace *sp, const double *f, double *y);
void plx_gspace_odd(const plx_gspace *sp, const double *f, double *y);
/* Back to node values on the full lattice (mirror-extended); nodes outside the coordinates get fill. */
void plx_gspace_even_field(const plx_gspace *sp, const double *y, double fill, double *f);
void plx_gspace_odd_field(const plx_gspace *sp, const double *y, double fill, double *f);
/* Gauge projection: remove span{1, Y0} from even coordinates y by ω-weighted least squares, in
 * place. coef (nullable) receives (α, γ) of the removed plane α + γ Y0. Idempotent. */
void plx_gspace_remove_plane(const plx_gspace *sp, double *y, double coef[2]);

/* ---- the estimator (DESIGN.md §5) ------------------------------------------------------------ */
typedef struct {
    int      robust_iters;  /* reweighting rounds of the rank-1 fit (default 8) */
    double   huber_k;       /* residual-norm cut as a multiple of the median (default 2.5) */
    int      kz_max;        /* depth-variation modes kept at most (default 8) */
    int      ks_max;        /* in-plane shape modes kept at most (default 24) */
    int      boot;          /* bootstrap resamples for the κ interval (default 200) */
    int      n_scramble;    /* scrambled-v controls for the rigid-body check (default 200) */
    uint64_t seed;
    double   sigma0_cond;   /* Σ0 eigenvalue floor = λ_max / sigma0_cond (default 1e3) */
    bool     no_pitch_lin;  /* drop the linear-in-t term of the κ fit; for ablation */
    bool     plain_pitch;   /* plain projection for t instead of the GLS estimate (step 4b); for ablation */
    bool     plain_angles;  /* sin θ = s/κ, sin φ = t/κ as specified, without the gauge scale ρ; ablation */
    bool     tilt;          /* step 8: estimate the depth tilt c1 and add it to z̄ (zbar only) */
    const float *prior_w;   /* [N] nullable: flow-quality weights in [0, 1] (0 = untrusted) */
    const double *gauge_w;  /* [P] nullable: the flow stage's similarity-gauge point weights; they set
                             * fx (step 5). NULL = ω */
} plx_emerge_opts;
plx_emerge_opts plx_emerge_defaults(void);

enum { PLX_GEOM_NSV = 12, PLX_GEOM_NCTRL = 5 };
/* Per-image flags. */
enum {
    PLX_GF_PRIOR = 1,     /* flow quality weight below 0.5 */
    PLX_GF_OUTLIER = 2,   /* rank-1 robust weight below 0.5 */
    PLX_GF_YAWVIOL = 4,   /* ρ |s| / κ >= 1: no real yaw explains it (model violation) */
    PLX_GF_PITCHVIOL = 8, /* likewise for pitch */
    PLX_GF_NOKAPPA = 16   /* excluded from the κ fit (bad anisotropy, |s| or |t| near κ, or low weight) */
};

typedef struct {
    plx_gspace sp;
    int n;
    /* step 3: yaw mode */
    double *v;         /* [P] node values; core nodes from the singular vector, others by regression on s */
    double *yv;        /* [ne] coordinates of v (|yv|² = wsum) */
    double  sv2[PLX_GEOM_NSV]; /* leading eigenvalues of the weighted E⊥ scatter (= σ_k²) */
    double  sv_frac1, sv_ratio12, sv_total;
    double *s, *t;     /* [n] px: s = ⟨E⊥, v⟩/⟨v, v⟩; t by GLS (step 4b, emerge.c) */
    double *t_raw;     /* [n] px: the plain projection -⟨V⊥, v⟩/⟨v, v⟩ */
    double  fx;        /* x share of the similarity gauge's scale, Σg X0² / Σg (X0² + Y0²); -1 = plain */
    double *w;         /* [n] robust weight used throughout, in [0, 1] */
    uint8_t *flags;    /* [n] */
    /* step 4: rigid-body check */
    double  rigid_frac;             /* fraction of V⊥ energy along v */
    double  rigid_ctrl[PLX_GEOM_NCTRL]; /* same for E⊥ singular vectors 2..6 */
    double  rigid_scr_mean, rigid_scr_sd, rigid_z, rigid_z_ctrl;
    /* step 5: metric scale */
    double *beta, *gamma, *logA; /* [n] (logA NaN where undefined) */
    double  kappa, kappa_lo, kappa_hi, kappa_yaw, kappa_pitch, kappa_c, kappa_d, kappa_scale;
    int     kappa_n;
    /* step 6: personal depth */
    int     kz, nr;     /* modes kept; dimension of the complement of span{1, Y0, v} */
    double *zeig;       /* [nr] eigenvalues of Σ_B, descending */
    double  zeig_floor; /* |most negative eigenvalue|: the estimation-noise level of Σ_B */
    double *Uc;         /* [ne][nr] orthonormal complement basis (row-major) */
    double *Br;         /* [nr][kz] B in complement coordinates */
    double *S0;         /* [nr][nr] Σ0, regularised */
    double *Sr;         /* [nr][nr] Σ_rest: positive part of Σ_B beyond the kept modes */
    double *Bnode;      /* [kz][P] node values of the modes, units of v (NaN outside the core) */
    double *chat, *Lam; /* [n][kz], [n][kz][kz] */
    double  s0_trace, sb_trace;
    /* step 7: in-plane shape */
    int     ks, nsh;    /* kept, total (no + ne) */
    double *sheig;      /* [nsh] eigenvalues (variance), descending */
    double *shape;      /* [ks][2][P] node fields of 1-σ modes, px (NaN outside the core) */
    double *scode;      /* [n][ks] whitened codes */
    /* step 8 */
    double  tilt_c1, tilt_se; /* depth Y-slope (px / px) and its standard error; NaN when not run */
} plx_geom_model;

/* u: f32 [n, 2, rows, cols]. node_weight nullable. Returns PLX_OK, or PLX_EINVAL (too few usable
 * images or a degenerate fit). */
int  plx_emerge(const plx_glat *L, const float *u, int n, const float *node_weight, const uint8_t *core,
                plx_core_shape cs, const plx_emerge_opts *o, plx_geom_model *m);
void plx_geom_model_free(plx_geom_model *m);

/* Pose angles (radians) from the parallax amplitudes: sin θ = ρ s/κ, sin φ cos θ = ρ t/κ with the
 * gauge's foreshortened scale ρ = fx cos θ + (1 - fx) cos φ (see emerge.c, step 5). Returns a bit
 * mask: 1 when no real yaw explains s, 2 likewise for pitch (that angle is then clipped near ±90°).
 * fx < 0 selects the uncorrected relation sin θ = s/κ, sin φ = t/κ (the model stores fx = -1 when
 * fitted with plain_angles). */
int plx_geom_angles(double fx, double kappa, double s, double t, double *yaw, double *pitch);

/* Per-image measurement shared by emerge and later single-image fitting: plane-removed even parts
 * yE, yV (ne), odd part yO (no) of field i, and the slopes β (O on X0) and γ (V on Y0). */
void plx_geom_measure(const plx_gspace *sp, const float *u_i, double *yE, double *yV, double *yO, double *beta,
                      double *gamma);
/* Depth code of one image from its plane-removed E coordinates and yaw amplitude s:
 * Λ = I + s² Bᵀ N⁻¹ B, ĉ = Λ⁻¹ s Bᵀ N⁻¹ R, R = E⊥ − s v, N = Σ0 + s² Σ_rest. c (kz) and Lam (kz × kz)
 * outputs. */
void plx_geom_depth_code(const plx_geom_model *m, const double *yE, double s, double *c, double *Lam);
/* Same-vs-different log-likelihood ratio of two depth codes under c ~ N(0, I): same person = one
 * shared c, different = independent c. Exactly 0 when either image is frontal (Λ = I). */
double plx_geom_llr(int kz, const double *c1, const double *L1, const double *c2, const double *L2);

/* ---- synthetic ground truth --------------------------------------------------------------- */
typedef struct {
    int      n;              /* photos */
    int      per_person;     /* photos per person (>= 1; identity shared, pose and nuisance independent) */
    double   yaw_sd_deg, yaw_clip_deg, pitch_sd_deg, pitch_mean_deg, pitch_clip_deg, roll_sd_deg;
    double   scale_sd, trans_sd_px;
    double   noise_px;       /* i.i.d. node noise */
    double   noise_yaw;      /* node noise grows as noise_px (1 + noise_yaw |sin θ|) */
    double   nuis_px;        /* pose-independent low-rank nuisance, ω-RMS per image */
    int      nuis_rank;
    double   a_px, b_px, g_px; /* identity RMS: widths (odd), heights (even), depth (even, modes 1..; mode 0 = nose size ±15%) */
    int      ka, kb, kg;
    double   ar_sd;          /* per-person aspect: a += w X0, b += h Y0, w, h ~ N(0, ar_sd) */
    double   nose_px;        /* nose protrusion above the cheeks */
    double   depth_tilt;     /* adds depth_tilt · Y0 to the true mean depth */
    bool     twins;          /* append the mirror twin of every photo (n doubles) */
    uint64_t seed;
} plx_synth_opts;
plx_synth_opts plx_synth_defaults(void);

typedef struct {
    int     n, P, kg;
    float  *u;           /* [n, 2, rows, cols], after both flow-stage gauges */
    double *zbar;        /* [P] true mean depth, px */
    double *G;           /* [kg][P] identity depth basis, px per unit coefficient */
    double *gc;          /* [n][kg] true identity depth coefficients */
    double *yaw, *pitch, *roll; /* [n] radians */
    int    *person;      /* [n] */
} plx_synth;

/* Generates on lattice L. core (with node_weight, nullable) sets the ω used to normalise the random
 * identity fields; gauge[P] is the similarity gauge's point weights (plx_gauge_box). */
void plx_synth_make(const plx_glat *L, const float *node_weight, const uint8_t *core, const double *gauge,
                    const plx_synth_opts *o, plx_synth *out);
void plx_synth_free(plx_synth *s);
/* The true mean depth surface (px) at centred coordinates; used by the generator and the tests. */
double plx_synth_depth(double X0, double Y0, double nose_px);

/* Recovery of a fit against the synthetic truth (all angles in degrees). */
enum { PLX_SYNTH_NBIN = 4 };
typedef struct {
    double v_corr;          /* ω-correlation of v with the plane-gauged true z̄ */
    double kappa_true;      /* ω-RMS of the plane-gauged true z̄ (px) */
    double kappa_ratio;     /* κ / kappa_true */
    double yaw_slope, yaw_corr, yaw_rmse;      /* recovered vs true yaw (slope of estimate on truth) */
    double pitch_slope, pitch_corr, pitch_rmse;/* recovered vs true pitch minus its population mean */
    double angle_cos[8];    /* cosines of the principal angles span(B̂) vs true depth variation */
    int    nangle;
    double bin_edge[PLX_SYNTH_NBIN + 1];   /* |yaw| bins (deg) */
    double code_r2[PLX_SYNTH_NBIN];        /* 1 - |B̂ĉ - d|² / |d|² per |yaw| bin, d = true variation */
    double calib_z2;        /* mean of (ĉ - c)ᵀ Λ (ĉ - c) / kz over images with |θ| > 10° (1 if calibrated) */
    double calib_sd[8];     /* per-component std of standardised errors (ĉ - c)_k / sqrt(Λ⁻¹_kk), |θ| > 10° */
} plx_synth_report;
void plx_synth_score(const plx_synth *s, const plx_geom_model *m, plx_synth_report *r);

/* The flow stage's two gauges, exactly as that stage applies them (exposed for the generator and
 * the tests): (1) per image, the similarity S minimising Σ g |S(p) - (p + u)|² over the gauge
 * points (weights g[P]) is removed, u <- S⁻¹(p + u) - p; (2) the mirror-symmetrised population
 * mean is subtracted. plx_gauge_box sets g = 1 on the nodes inside a canonical box (inclusive), 0
 * elsewhere: the flow stage uses x in [23.5, 103.5], y in [23.5, 135.5]. */
void plx_gauge_similarity(const plx_glat *L, const double *g, float *u_i);
void plx_gauge_mean(const plx_glat *L, float *u, int n);
void plx_gauge_box(const plx_glat *L, double x0, double x1, double y0, double y1, double *g);

#endif
