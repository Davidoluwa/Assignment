/* Multi-frame fusion: many views of ONE person, one coherent 3D shape (docs/DESIGN.md §8).
 *
 * PIPELINE (each step is a function below; apps/plx_fuse.c strings them together):
 *   clip    frames from a .plxv container or a directory of numbered images;
 *   locate  where the face is in an arbitrary frame (no detector exists: the congealing template
 *           is slid over position x scale x roll, the best few hypotheses are refined with the
 *           similarity alignment) -> ONE fixed 250 x 250 window for the whole clip, laid out like
 *           an LFW crop, so every frame shares one pixel coordinate system (static camera,
 *           moving head) and the unmodified population model applies;
 *   track   per frame: population flow, warm-started from the previous frame; then personal
 *           re-tracking: the person's own template (mean of their warped frames) replaces the
 *           population template and every frame is re-fitted with a much softer field;
 *   factor  the tracked node positions W (2F x P) = M S + t (weak perspective, rank 3):
 *           weighted ALS, metric upgrade, explicit mirror and bas-relief handling, canonical
 *           gauge, non-rigid residue, parametric-bootstrap uncertainty;
 *   fuse    visibility-weighted texture in Ω, and the output record (record.c).
 *
 * CONVENTIONS are those of DESIGN.md §2: crop pixels x right / y down; 3D X right, Y down, Z
 * towards the viewer; R = Rz(roll) Rx(pitch) Ry(yaw) (render/render.h); image position of a 3D
 * point v in frame f is  x_f = s_f [R_f v]_xy + t_f. The canonical gauge puts the shape on the
 * canonical layout: node n sits at (X0 + a, Y0 + b, z) with (X0, Y0) its centred canonical
 * position, depth in canonical pixels, nose towards the camera (z > 0 at the face centre).
 *
 * GAUGES AND AMBIGUITIES (reported, never hidden):
 *   - similarity: the factorised shape is only defined up to rotation, translation and scale; it
 *     is put on (X0, Y0, zp) (zp = the population mean depth at the nodes) by weighted Procrustes;
 *   - mirror: orthographic projection cannot tell a depth map from its reflection; the solution
 *     whose depth correlates positively with zp is kept, both correlations are reported;
 *   - bas-relief: the metric upgrade pins the depth scale only through second-order
 *     (foreshortening, cos θ) effects, so small or single-axis motion leaves depth scale and
 *     rotation amplitude nearly interchangeable. What matters is how well the scale is pinned,
 *     and that is measured directly: a parametric bootstrap of the metric solution gives the
 *     relative sd of its depth scale. The scale is then fused with the population prior (the
 *     scale that makes the projection of the person's depth onto zp equal zp's own) by
 *     precision, in log space, exactly as DESIGN §5.5 fuses everything else; the record is
 *     flagged degenerate when the prior carries more weight than the motion. The spread of the
 *     rotation axes and the conditioning (gap) of the metric constraint system are reported as
 *     diagnostics. (A first version flagged degeneracy from a small second rotation axis; on
 *     synthetic clips a pure ±30° yaw sweep pins the scale to 3 % with a bootstrap sd of 0.3 %,
 *     so a single axis is not degenerate in itself: the amplitude is what counts.) */
#ifndef PLX_FUSE_H
#define PLX_FUSE_H

#include "align/canon.h"
#include "align/congeal.h"
#include "align/flow.h"
#include "align/lattice.h"
#include "core/core.h"
#include "img/image.h"

#include <stdbool.h>
#include <stdint.h>

/* ---- clip container ---------------------------------------------------------------------
 * .plxv (little-endian): magic "PLXV", u32 version = 1, u32 frame count, then per frame:
 * f64 timestamp (ms), u32 byte length, that many bytes of an encoded image. The browser capture
 * page writes JPEG; anything plx_img_decode sniffs (JPEG, binary PNM) is accepted, which is how
 * synthetic clips are stored (the system has no JPEG encoder).
 * A DIRECTORY is read as a clip too: every regular file whose name contains a decimal number and
 * ends in .jpg/.jpeg/.ppm/.pgm, ordered by the LAST number in the name; timestamps are
 * index * 1000 / 30 ms (no timing information exists). */
typedef struct {
    int       n;
    double   *t_ms; /* [n] */
    uint8_t **data; /* [n] encoded frames */
    size_t   *len;  /* [n] */
} plx_fu_clip;

int  plx_fu_clip_read(const char *path, plx_fu_clip *c);  /* .plxv file or directory; PLX_EIO / PLX_EFORMAT */
int  plx_fu_clip_write(const char *path, const plx_fu_clip *c);
void plx_fu_clip_append(plx_fu_clip *c, double t_ms, const uint8_t *data, size_t len); /* copies data */
void plx_fu_clip_free(plx_fu_clip *c);
/* Frame i decoded to [0, 1] (1 or 3 channels as stored). */
int  plx_fu_clip_decode(const plx_fu_clip *c, int i, plx_img *out);

/* ---- crop window ---------------------------------------------------------------------------
 * The fixed window of a clip: crop pixel q maps to frame point (x0 + s qx, y0 + s qy). Upright
 * (the camera is upright; head roll is tracked, not cropped away). */
typedef struct { double s, x0, y0; } plx_fu_window;
/* Window that puts the face of image similarity T (Ω -> frame) where the population puts an
 * average face in a 250 x 250 LFW crop (T_crop(c0) = (124.5, 128.5), scale |T| -> 1). */
plx_fu_window plx_fu_window_for(plx_sim T);
/* 250 x 250 crop of frame (any channels) through the window, bilinear; when s > 1 the frame is
 * first blurred by 0.5 sqrt(s² - 1) so the shrink does not alias. */
plx_img plx_fu_window_extract(const plx_img *frame, plx_fu_window w);
/* Similarities between the two pixel systems: crop(T) = window⁻¹ o T and its inverse. */
plx_sim plx_fu_sim_to_crop(plx_fu_window w, plx_sim T_frame);
plx_sim plx_fu_sim_to_frame(plx_fu_window w, plx_sim T_crop);

/* ---- localisation ---------------------------------------------------------------------- */
typedef struct {
    double scale_min, scale_max; /* face scale relative to an LFW crop; <= 0: 0.5 .. min(w, h) / 150 */
    double scale_step;           /* geometric step of the scale grid (default 2^(1/4)) */
    double rot_max_deg;          /* rolls searched: -rot_max, 0, +rot_max (default 10) */
    int    level;                /* pyramid level of the brute-force search (default 2) */
    int    n_refine;             /* hypotheses refined by similarity alignment (default 5) */
} plx_fu_locate_opts;
plx_fu_locate_opts plx_fu_locate_defaults(void);

/* Scores are weighted normalised cross-correlations of the frame's orientation channels with
 * the template mean (locate.c explains why not the alignment cost); 1 = perfect, ~0 = no face. */
typedef struct {
    plx_sim T;           /* Ω -> frame, after refinement */
    double  score;       /* refined score of the winner */
    double  search_score;/* its brute-force score */
    double  runner_score;/* refined score of the best hypothesis elsewhere in the frame (-2 if none) */
    double  cost;        /* the winner's similarity-alignment cost (congealing's, ≈ K typical) */
    int     n_hyp;       /* hypotheses refined */
} plx_fu_located;

typedef struct {
    const plx_fl_model *m;
    plx_cg_params       p;
    int                 n_stages;
    plx_cg_stage        stages[8];
    plx_cg_ref          refs[8];
} plx_fu_locator;
void plx_fu_locator_init(plx_fu_locator *L, const plx_fl_model *m);
void plx_fu_locator_free(plx_fu_locator *L);
/* gray: 1-channel frame of any size. Thread-safe (the locator is read-only). */
plx_fu_located plx_fu_locate(const plx_fu_locator *L, const plx_img *gray, const plx_fu_locate_opts *o);

/* ---- tracking ----------------------------------------------------------------------------
 * Every frame f gets a full warp W_f(p) = T_f(p + u_f(p)) from Ω into the crop. Population pass:
 * plx_fl_fit_image from the previous frame's similarity. Personal pass: every frame is rendered
 * ONCE through its population warp, and the personal increment is sampled on top of that render
 * (track.c says why re-rendering is not an option); then, `rounds` times: (1) personal template =
 * per-pixel mean (and variance) of the frames' sampled orientation features at every pyramid
 * level, weighted by sample validity (and visibility, see plx_fu_track_retrack), NOT
 * symmetrised (a person is not exactly symmetric and their own asymmetries are good texture);
 * (2) every frame's increment is re-fitted against it, warm-started, on lattice `step` with
 * bending weight `lambda` (relative, flow.h; the population needs λ = 100 because it matches
 * across identities, frames of one face do not) and plain window weights; (3) gauge: the
 * frame-mean increment is removed (template and fits could otherwise drift together; the
 * personal frame stays anchored to the population fits' average). */
typedef struct {
    int   rounds;       /* personal template <-> re-fit iterations (default 4) */
    int   step;         /* personal lattice: 8 or 4 (default 8; 4 was not better on synthetic clips) */
    float lambda;       /* relative bending weight of the personal fit (default 0.5) */
    int   iters;        /* Gauss-Newton iterations per personal stage (default 10) */
    bool  coarse;       /* also run a level-1 stage first (default true) */
} plx_fu_track_opts;
plx_fu_track_opts plx_fu_track_defaults(void);

typedef struct {
    int          F;
    plx_lattice  lat;          /* personal lattice (8 or 4 px) */
    plx_sim     *T_pop;        /* [F] population fit (8-px field u_pop) */
    float       *u_pop;        /* [F][2 * 357] */
    plx_sim     *T;            /* [F] personal fit */
    float       *u;            /* [F][dof(lat)] */
    double      *cost_pop;     /* [F] population fit cost against the population template */
    /* the next three are in units of the personal template's own across-frame variance, so a
     * well-tracked clip sits near 1, and the population warps' value shows what re-tracking bought */
    double      *cost_pp;      /* [F] population warps measured against the FINAL personal template */
    double      *cost;         /* [F] personal warps against the final personal template */
    float       *node_cost;    /* [F][nodes] per-node normalised residual of the final personal fit */
    float       *node_data;    /* [nodes] data share of the personal stiffness (flow.h data fraction) */
    double       drift[8];     /* per round: RMS of the removed frame-mean change, canonical px */
    plx_img      tmpl_gray;    /* Ω personal mean gray face (final round) */
    plx_img      pop_gray;     /* Ω mean gray face through the population fits */
    plx_cg_template tmpl;      /* final personal feature template */
} plx_fu_track;

/* grays: F crops (1 channel, 250 x 250). T0: population-frame similarity of frame 0 (from
 * localisation). */
void plx_fu_track_clip(const plx_fl_fitter *pop, const plx_img *grays, int F, plx_sim T0, const plx_fu_track_opts *o,
                       plx_fu_track *out);
/* Re-runs the personal pass (from the population fits already in t) with per-frame canonical
 * visibility vis [F][128 * 160] in [0, 1] (plx_fu_visibility): a sample's weight in the template
 * and in the frame's fit is multiplied by it, so a frame contributes nothing where that part of
 * the surface faces away or is hidden (the far flank of the nose at a turn). */
void plx_fu_track_retrack(const plx_fl_fitter *pop, const plx_img *grays, const float *vis, const plx_fu_track_opts *o,
                          plx_fu_track *t);
void plx_fu_track_free(plx_fu_track *t);
/* x [F][2][nodes]: image (crop) positions W_f(p_n) of the personal lattice nodes; pop = true uses
 * the population fits (8-px nodes when lat is 8 px; refined exactly onto a 4-px lattice). */
void plx_fu_track_positions(const plx_fu_track *t, bool pop, double *x);

/* ---- factorisation ---------------------------------------------------------------------- */
typedef struct {
    int           F, P;
    int           nx, ny;    /* nodes form an nx x ny grid, row-major (P = nx ny): normals, visibility */
    int           img_w, img_h; /* crop size the positions live in (visibility raster) */
    const double *x;         /* [F][2][P] image positions */
    const double *w;         /* [F][P] base weights >= 0 (0 = missing) */
    const double *X0, *Y0;   /* [P] centred canonical node positions */
    const double *zp;        /* [P] population mean depth at the nodes: sign, gauge, prior scale */
    const double *gw;        /* [P] weights of the canonical gauge and the prior projection (nullable: 1) */
} plx_fu_tracks;

typedef struct {
    int      als_iters;     /* default 300 */
    double   als_tol;       /* relative SSE change (default 1e-10) */
    int      rounds;        /* robust / visibility re-weighting rounds (default 4; 1 = none) */
    double   robust_k;      /* Cauchy weight 1 / (1 + (e / (k σ))²) per entry (default 3) */
    double   nonrigid_k;    /* per-node 1 / (1 + (rms / (k median))²) (default 2; <= 0 off) */
    bool     visibility;    /* weight entries by surface visibility from round 2 on (default true) */
    double   prior_scale_sd;/* relative sd of the population prior on a person's depth scale (default 0.15) */
    int      prior_scale;   /* depth scale: -1 metric fused with the prior by precision (default), 0 metric
                             * only, 1 prior only */
    int      boot;          /* parametric bootstrap resamples for the uncertainty (default 40; 0 = none;
                             * the automatic depth-scale fusion always runs at least 16 of the metric
                             * solution to know its precision) */
    bool     boot_frames;   /* with F >= 8, also resample whole frames and keep the larger spread
                             * (default true; factor.c says why) */
    uint64_t seed;
} plx_fu_factor_opts;
plx_fu_factor_opts plx_fu_factor_defaults(void);

typedef struct {
    int     F, P;
    double *S;          /* [3][P] canonical gauge (X0 + a, Y0 + b, z); NaN for unused nodes */
    double *sd;         /* [3][P] bootstrap standard deviation of each coordinate, px (NaN unused) */
    double *R;          /* [F][9] rotation of each frame (canonical gauge) */
    double *scale, *tx, *ty, *yaw, *pitch, *roll; /* [F]; angles in radians */
    double *M;          /* [F][2][3] scaled projection rows (canonical gauge), t = (tx, ty) */
    double *w;          /* [F][P] final weights (base x robust x non-rigid x visibility) */
    double *vis;        /* [F][P] visibility of the final shape in each frame, [0, 1] */
    double *res;        /* [F][P] reprojection residual |x - M S - t| px (NaN where base weight 0) */
    double *node_rms;   /* [P] weighted RMS residual over frames (non-rigid residue), px */
    double *frame_rms;  /* [F] weighted RMS residual over nodes, px */
    uint8_t *used;      /* [P] node estimated */
    double  rms;        /* weighted reprojection RMS, px */
    double  sigma;      /* robust per-coordinate track noise σ, px */
    double  gap;        /* metric constraint system: σ5 / σ1 (row-normalised; ~0 = degenerate) */
    double  spread_deg[3]; /* principal angular spreads of the frame rotations about their mean, deg */
    double  axis[3][3]; /* the corresponding axes (rows), canonical gauge */
    double  L_min_eig;  /* smallest eigenvalue of L / largest (< 0: L projected to PSD) */
    double  scale_sd_metric; /* bootstrap relative sd of the METRIC solution's depth scale (NaN: not run) */
    double  prior_weight;/* weight of the prior in the fused depth scale, sd_m² / (sd_m² + sd_prior²) */
    bool    degenerate; /* bas-relief: the prior outweighs the motion (prior_weight > 0.5) */
    double  depth_ratio;/* ⟨z, zp⟩ / ⟨zp, zp⟩ (centred, gw-weighted) of the metric solution */
    double  prior_scale;/* factor applied to z by the fusion: ratio^(-prior_weight) (1 = metric) */
    double  corr;       /* depth correlation with zp of the kept solution ... */
    double  corr_mirror;/* ... and of its mirror (depth-reflected) twin, rejected. Which of the two
                         * the metric upgrade happens to return is arbitrary (the handedness of the
                         * affine factors is), so only the margin between these is meaningful */
    double  gauge_rms;  /* Procrustes residual to (X0, Y0, zp), px */
    double  scale_sd;   /* bootstrap relative sd of the final depth scale (NaN without bootstrap) */
    int     n_boot;
} plx_fu_shape;

int  plx_fu_factorise(const plx_fu_tracks *tr, const plx_fu_factor_opts *o, plx_fu_shape *out);
void plx_fu_shape_free(plx_fu_shape *s);

/* Building blocks (exposed for the tests). */
/* Weighted rank-3 + translation fit x ≈ M S + t by alternating least squares from the SVD of the
 * centred matrix. x [F][2][P], w [F][P]; out M [F][2][3], t [F][2], S [3][P]. Returns iterations. */
int    plx_fu_als(int F, int P, const double *x, const double *w, int iters, double tol, double *M, double *t,
                  double *S);
/* Metric upgrade: Q (3x3, row-major) with L = Q Qᵀ making every frame's two rows of M Q
 * orthogonal with equal norm (linear least squares in the 6 entries of L, smallest singular
 * vector; L projected to PSD). frame_w (nullable) weights frames. gap = σ5/σ1, min_eig = smallest
 * / largest eigenvalue of L before the projection. Returns PLX_OK or PLX_EINVAL (F < 3). */
int    plx_fu_metric(int F, const double *M, const double *frame_w, double Q[9], double *gap, double *min_eig);
/* Weighted similarity Procrustes (Umeyama): B ≈ s R A + t over n points (A, B [3][n]); R proper
 * (det +1). Returns the weighted RMS residual. */
double plx_fu_procrustes(int n, const double *A, const double *B, const double *w, double *s, double R[9],
                         double t[3]);
/* Yaw, pitch, roll of R = Rz(roll) Rx(pitch) Ry(yaw) (the inverse of plx_rot_ypr). */
void   plx_fu_angles(const double R[9], double *yaw, double *pitch, double *roll);
/* Visibility of grid surface points (S [3][nx * ny], row-major grid, valid nullable) seen at
 * x = s [R v]_xy + t in a w x h image: z-buffered (occluded -> 0) times the facing factor
 * clamp(n_z / 0.5, 0, 1) of the grid normal. */
void   plx_fu_grid_visibility(const double *S, const uint8_t *valid, int nx, int ny, const double R[9], double s,
                              double tx, double ty, int w, int h, double *vis);

/* ---- fused texture and the record -------------------------------------------------------
 * RECORD (a directory; what the recognition back-end consumes as "a face enrolled from video"):
 *   record.txt        key value lines: version, frames, lattice step, window, quality numbers
 *                     (rms, sigma, gap, spreads, degenerate, flipped, corr, track residuals, ...)
 *   shape.plxt        f32 [7, 21, 17]  a, b, z, sd_x, sd_y, sd_z, weight on the 8-px lattice
 *                     (canonical px; NaN where not estimated; weight = Σ_f final weight / F)
 *   shape_lat.plxt    f32 [7, ny, nx]  the same on the personal lattice (equal to shape.plxt at 8 px)
 *   posterior.plxt    f32 [2, 21, 17]  z fused with the population prior (DESIGN §5.5: precisions
 *                     add, per node), and its sd
 *   depth.plxt        f32 [160, 128]   dense depth (bicubic over the personal lattice), NaN outside
 *                     the estimated region
 *   texture.plxt      f32 [C, 160, 128] fused texture in Ω (C = 3 for colour clips, else 1)
 *   texture_w.plxt    f32 [160, 128]   Σ visibility weights behind each texture pixel
 *   pose.plxt         f32 [F, 9]       t_ms, yaw, pitch, roll (rad), scale, tx, ty (crop px),
 *                     frame rms (px), frame weight
 *   tracks.plxt       f32 [F, 3, ny, nx] x, y (crop px) and final weight of every node
 *   template.plxt     f32 [160, 128]   personal mean gray face */
typedef struct {
    int           F, C;
    plx_lattice   lat;
    float        *shape;    /* [7][P8] on the 8-px lattice */
    float        *shape_lat;/* [7][P] on lat */
    float        *post;     /* [2][P8] */
    plx_img       depth, texture, texture_w, tmpl;
    float        *pose;     /* [F][9] */
    float        *tracks;   /* [F][3][P] */
    char         *info;     /* record.txt contents (key value lines) */
} plx_fu_record;

/* Texture fusion: every frame (C channels, crop pixels) warped into Ω through its personal fit
 * and averaged with weights = per-pixel visibility of the dense shape at that frame's pose x the
 * frame's weight; each frame's gain is first matched to the visibility-weighted median frame
 * (lighting drifts during a clip). dense_S [3][128 * 160] (canonical gauge). Out: texture
 * (C x 128 x 160) and the weight sum. */
void plx_fu_fuse_texture(const plx_img *frames, int F, const plx_fu_track *tr, const plx_fu_shape *sh,
                         const double *dense_S, plx_img *texture, plx_img *weight);
/* Per-frame visibility of every canonical pixel (vis [F][128 * 160]) of the dense surface at the
 * shape's poses, in a w x h crop (plx_fu_grid_visibility on the dense grid). */
void plx_fu_visibility(const plx_fu_shape *sh, const double *dense_S, int w, int h, float *vis);
/* Dense (128 x 160) canonical-gauge surface from a lattice shape (Catmull-Rom over the nodes,
 * NaN-filled harmonically, NaN outside the estimated region when mask is set). S_lat [3][P]. */
void plx_fu_dense_surface(const plx_lattice *L, const double *S_lat, const uint8_t *used, bool mask, double *dense);

int  plx_fu_record_save(const char *dir, const plx_fu_record *r);
int  plx_fu_record_load(const char *dir, plx_fu_record *r);
void plx_fu_record_free(plx_fu_record *r);

/* ---- synthetic clips with ground truth ----------------------------------------------------
 * A head is a height field z(X0, Y0) over the canonical frame (centred coordinates), textured
 * with a frontalised face (Ω, 128 x 160, 1 or 3 channels) and rendered by the z-buffer renderer
 * through x = s [R v]_xy + t, R = Rz Rx Ry, over a static background. */
typedef struct {
    double dome_depth, dome_ax, dome_ay;  /* half-ellipsoid: depth, semi-axes (px) */
    double nose, nose_w, nose_len;        /* nose protrusion, half-width (sd px), ridge length */
    double eye_depth, eye_x, eye_y;       /* eye-socket dents */
    double brow, lips, chin;              /* brow ridge, lip and chin relief */
} plx_fu_head;
plx_fu_head plx_fu_head_preset(int which);  /* 0: average (geom synth's face), 1: long nose, flat
                                             * face, deep eyes, 2: short wide nose, round face */
double plx_fu_head_depth(const plx_fu_head *h, double X0, double Y0); /* NaN outside the head */

/* Radians; s = scale, (x, y) = frame position of the head's origin (X0 = Y0 = Z = 0). */
typedef struct { double yaw, pitch, roll, s, x, y; } plx_fu_pose;

typedef struct {
    int    F;
    double yaw_deg, pitch_deg, roll_deg; /* amplitudes: yaw sweep ±, nod ±, roll ± */
    double scale_drift, trans_drift;     /* relative scale ±, translation ± px */
    double yaw_cycles, pitch_cycles;     /* periods over the clip (default 1 and 2) */
    double noise, gain;                  /* sensor noise sd, brightness change ± (fraction) */
    double shade;                        /* Lambertian shading mix in [0, 1] (0: albedo only) */
    int    frame_w, frame_h;             /* frame size (250 x 250: already an LFW-like crop) */
    double face_scale, face_x, face_y;   /* placement in larger frames: scale, image of c0 */
    bool   clutter;                      /* cluttered background (else smooth gradient) */
    uint64_t seed;
} plx_fu_synth_opts;
plx_fu_synth_opts plx_fu_synth_defaults(void);
/* Frame f's true pose (deterministic in f / F). */
plx_fu_pose plx_fu_synth_pose(const plx_fu_synth_opts *o, int f);
/* Renders frame f: tex (Ω, 1 or 3 channels), background (frame size, same channels; NULL = made
 * from the seed). Output has tex's channel count. */
plx_img plx_fu_synth_frame(const plx_fu_head *h, const plx_img *tex, const plx_img *background,
                           const plx_fu_synth_opts *o, int f);
/* A background image of the frame size: smooth gradient, or clutter (random rectangles, discs,
 * lines and gratings) when o->clutter. */
plx_img plx_fu_synth_background(const plx_fu_synth_opts *o, int channels);
/* Projection of the head point at canonical centred (X0, Y0) with pose P (frame pixels). */
void plx_fu_synth_project(const plx_fu_head *h, const plx_fu_synth_opts *o, plx_fu_pose P, double X0, double Y0,
                          double *x, double *y);

#endif
