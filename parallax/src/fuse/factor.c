/* Rigid factorisation of tracked node positions into one shape and per-frame poses
 * (docs/DESIGN.md §8; fuse.h has the model, gauges and ambiguities).
 *
 * MODEL. x_f(n) = M_f S_n + t_f + ε, M_f the two scaled rotation rows of frame f. Translation is
 * an unknown of the fit rather than removed by centring first: with missing or down-weighted
 * entries the per-frame centroid of the VISIBLE nodes is not the projection of the shape's
 * centroid, and centring by it would bias the shape. With complete, equally weighted data the
 * two are identical.
 *
 * ESTIMATE (one call of estimate()): ALS -> metric upgrade -> mirror choice + Procrustes to the
 * canonical layout -> rotations, axis spreads -> bas-relief decision (prior-scaled depth). The
 * robust loop in plx_fu_factorise re-weights entries (Cauchy on the residual), nodes (non-rigid
 * residue) and, once a shape exists, visibility (z-buffer + facing), and re-estimates.
 *
 * UNCERTAINTY. Parametric bootstrap: synthetic tracks = fitted model + Gaussian noise with the
 * estimated per-node noise level divided by the entry's weight; each resample goes through the
 * same estimate() with the mirror and bas-relief decisions of the main fit held fixed (they are
 * discrete choices of the model, not noise). The spread of the resampled canonical shapes is the
 * per-coordinate sd: it contains the per-node triangulation error (∝ 1 / sqrt(Σ_f sin² of the
 * view angle) -- "precision adds"), the depth-scale uncertainty of the metric upgrade and the
 * gauge, everything the motion geometry implies. It does not contain systematic tracking bias. */
#include "fuse/fuse.h"

#include "la/la.h"
#include "render/render.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define DEG (M_PI / 180.0)

plx_fu_factor_opts plx_fu_factor_defaults(void) {
    return (plx_fu_factor_opts){.als_iters = 300, .als_tol = 1e-10, .rounds = 4, .robust_k = 3, .nonrigid_k = 2,
                                .visibility = true, .degen_spread_deg = 3, .degen_gap = 1e-3, .prior_scale = -1,
                                .boot = 40, .seed = 0xf05e};
}

/* ---- small dense helpers -------------------------------------------------------------------- */
/* A x = b for n <= 6 by Gaussian elimination with partial pivoting, in place (x in b). A tiny
 * ridge rescues exactly singular systems (a node seen in one frame only). */
static void solve_small(int n, double *A, double *b) {
    double tr = 0;
    for (int i = 0; i < n; i++) tr += fabs(A[i * n + i]);
    for (int i = 0; i < n; i++) A[i * n + i] += 1e-12 * (tr > 0 ? tr : 1) / n;
    for (int c = 0; c < n; c++) {
        int p = c;
        for (int r = c + 1; r < n; r++)
            if (fabs(A[r * n + c]) > fabs(A[p * n + c])) p = r;
        if (p != c) {
            for (int k = 0; k < n; k++) {
                double t = A[c * n + k];
                A[c * n + k] = A[p * n + k];
                A[p * n + k] = t;
            }
            double t = b[c];
            b[c] = b[p];
            b[p] = t;
        }
        double d = A[c * n + c];
        if (d == 0) continue;
        for (int r = c + 1; r < n; r++) {
            double f = A[r * n + c] / d;
            if (f == 0) continue;
            for (int k = c; k < n; k++) A[r * n + k] -= f * A[c * n + k];
            b[r] -= f * b[c];
        }
    }
    for (int c = n - 1; c >= 0; c--) {
        double s = b[c];
        for (int k = c + 1; k < n; k++) s -= A[c * n + k] * b[k];
        b[c] = A[c * n + c] != 0 ? s / A[c * n + c] : 0;
    }
}

static void mat3_mul(const double *A, const double *B, double *C) {
    double T[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) T[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
    memcpy(C, T, sizeof T);
}

static void cross3(const double *a, const double *b, double *c) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}

/* ---- ALS ---------------------------------------------------------------------------------- */
static double als_sse(int F, int P, const double *x, const double *w, const double *M, const double *t,
                      const double *S) {
    double e = 0;
    for (int f = 0; f < F; f++)
        for (int n = 0; n < P; n++) {
            double wf = w[(size_t)f * P + n];
            if (wf <= 0) continue;
            for (int r = 0; r < 2; r++) {
                const double *m = M + (f * 2 + r) * 3;
                double d = x[((size_t)f * 2 + r) * P + n] - t[f * 2 + r] -
                           (m[0] * S[n] + m[1] * S[P + n] + m[2] * S[2 * P + n]);
                e += wf * d * d;
            }
        }
    return e;
}

int plx_fu_als(int F, int P, const double *x, const double *w, int iters, double tol, double *M, double *t,
               double *S) {
    PLX_ASSERT(F >= 2 && P >= 3);
    /* initialisation: weighted per-frame centroids, missing entries at the centroid, SVD */
    plx_mat A = plx_mat_new(2 * F, P);
    for (int f = 0; f < F; f++)
        for (int r = 0; r < 2; r++) {
            double sw = 0, sx = 0;
            for (int n = 0; n < P; n++) {
                double wf = w[(size_t)f * P + n];
                sw += wf;
                sx += wf * x[((size_t)f * 2 + r) * P + n];
            }
            double c = sw > 0 ? sx / sw : 0;
            t[f * 2 + r] = c;
            for (int n = 0; n < P; n++)
                if (w[(size_t)f * P + n] > 0) PLX_M(&A, f * 2 + r, n) = x[((size_t)f * 2 + r) * P + n] - c;
        }
    int k = PLX_MIN(2 * F, P);
    plx_mat U = plx_mat_new(2 * F, k), V = plx_mat_new(P, k);
    double *sv = PLX_NEW(double, k);
    if (plx_svd(&A, &U, sv, &V) != PLX_OK) plx_die("fuse: SVD of the track matrix failed");
    for (int i = 0; i < 2 * F; i++)
        for (int j = 0; j < 3; j++) M[i * 3 + j] = j < k ? PLX_M(&U, i, j) * sqrt(sv[j]) : 0;
    for (int j = 0; j < 3; j++)
        for (int n = 0; n < P; n++) S[j * P + n] = j < k ? sqrt(sv[j]) * PLX_M(&V, n, j) : 0;
    plx_mat_free(&A), plx_mat_free(&U), plx_mat_free(&V), free(sv);

    double prev = als_sse(F, P, x, w, M, t, S);
    int it = 0;
    for (; it < iters && prev > 0; it++) {
        /* S step: per node, 3 unknowns */
        for (int n = 0; n < P; n++) {
            double H[9] = {0}, b[3] = {0};
            for (int f = 0; f < F; f++) {
                double wf = w[(size_t)f * P + n];
                if (wf <= 0) continue;
                for (int r = 0; r < 2; r++) {
                    const double *m = M + (f * 2 + r) * 3;
                    double y = x[((size_t)f * 2 + r) * P + n] - t[f * 2 + r];
                    for (int i = 0; i < 3; i++) {
                        b[i] += wf * m[i] * y;
                        for (int j = 0; j < 3; j++) H[i * 3 + j] += wf * m[i] * m[j];
                    }
                }
            }
            solve_small(3, H, b);
            for (int i = 0; i < 3; i++) S[i * P + n] = b[i];
        }
        /* M step: per frame, rows share the normal matrix of [S; 1] */
        for (int f = 0; f < F; f++) {
            double H[16] = {0}, b[2][4] = {{0}};
            for (int n = 0; n < P; n++) {
                double wf = w[(size_t)f * P + n];
                if (wf <= 0) continue;
                double s4[4] = {S[n], S[P + n], S[2 * P + n], 1};
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) H[i * 4 + j] += wf * s4[i] * s4[j];
                    for (int r = 0; r < 2; r++) b[r][i] += wf * s4[i] * x[((size_t)f * 2 + r) * P + n];
                }
            }
            for (int r = 0; r < 2; r++) {
                double Hc[16];
                memcpy(Hc, H, sizeof H);
                solve_small(4, Hc, b[r]);
                for (int i = 0; i < 3; i++) M[(f * 2 + r) * 3 + i] = b[r][i];
                t[f * 2 + r] = b[r][3];
            }
        }
        double e = als_sse(F, P, x, w, M, t, S);
        bool done = prev - e <= tol * prev;
        prev = e;
        if (done) {
            it++;
            break;
        }
    }
    return it;
}

/* ---- metric upgrade ------------------------------------------------------------------------- */
/* Coefficients of aᵀ L b in l = (L11, L12, L13, L22, L23, L33). */
static void lrow(const double *a, const double *b, double *r) {
    r[0] = a[0] * b[0];
    r[1] = a[0] * b[1] + a[1] * b[0];
    r[2] = a[0] * b[2] + a[2] * b[0];
    r[3] = a[1] * b[1];
    r[4] = a[1] * b[2] + a[2] * b[1];
    r[5] = a[2] * b[2];
}

static double qform(const double *L, const double *a, const double *b) {
    double s = 0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) s += a[i] * L[i * 3 + j] * b[j];
    return s;
}

int plx_fu_metric(int F, const double *M, const double *frame_w, double Q[9], double *gap, double *min_eig) {
    if (F < 3) return PLX_EINVAL;
    plx_mat A = plx_mat_new(2 * F, 6);
    for (int f = 0; f < F; f++) {
        const double *m1 = M + f * 6, *m2 = m1 + 3;
        /* each frame's constraints are normalised by its squared scale (weak perspective: frames
         * at different distances count alike), then weighted */
        double nrm = plx_vec_dot(m1, m1, 3) + plx_vec_dot(m2, m2, 3);
        double wf = (frame_w ? sqrt(fmax(frame_w[f], 0)) : 1) / (nrm > 0 ? nrm : 1);
        double r11[6], r22[6], r12[6];
        lrow(m1, m1, r11), lrow(m2, m2, r22), lrow(m1, m2, r12);
        for (int k = 0; k < 6; k++) {
            PLX_M(&A, 2 * f, k) = wf * (r11[k] - r22[k]);
            PLX_M(&A, 2 * f + 1, k) = wf * 2 * r12[k];
        }
    }
    plx_mat V = plx_mat_new(6, 6);
    double s[6];
    int st = plx_svd(&A, NULL, s, &V);
    plx_mat_free(&A);
    if (st != PLX_OK) {
        plx_mat_free(&V);
        return st;
    }
    *gap = s[0] > 0 ? s[4] / s[0] : 0;
    double l[6];
    for (int k = 0; k < 6; k++) l[k] = PLX_M(&V, k, 5);
    plx_mat_free(&V);
    double L[9] = {l[0], l[1], l[2], l[1], l[3], l[4], l[2], l[4], l[5]};
    /* overall scale: the mean frame scale is kept (the canonical gauge fixes it later anyway) */
    double num = 0, den = 0;
    for (int f = 0; f < F; f++) {
        const double *m1 = M + f * 6, *m2 = m1 + 3;
        num += qform(L, m1, m1) + qform(L, m2, m2);
        den += plx_vec_dot(m1, m1, 3) + plx_vec_dot(m2, m2, 3);
    }
    if (num == 0) return PLX_EINVAL;
    for (int k = 0; k < 9; k++) L[k] *= den / num;
    plx_mat Lm = plx_mat_from(3, 3, L), E = plx_mat_new(3, 3);
    double ev[3];
    st = plx_eigh(&Lm, ev, &E);
    plx_mat_free(&Lm);
    if (st != PLX_OK || ev[0] <= 0) {
        plx_mat_free(&E);
        return st != PLX_OK ? st : PLX_EINVAL;
    }
    *min_eig = ev[2] / ev[0];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) Q[i * 3 + j] = PLX_M(&E, i, j) * sqrt(fmax(ev[j], 1e-6 * ev[0]));
    plx_mat_free(&E);
    return PLX_OK;
}

/* ---- Procrustes, angles --------------------------------------------------------------------- */
double plx_fu_procrustes(int n, const double *A, const double *B, const double *w, double *s, double R[9],
                         double t[3]) {
    double W = 0, ma[3] = {0}, mb[3] = {0};
    for (int i = 0; i < n; i++) {
        double wi = w ? w[i] : 1;
        if (!(wi > 0)) continue;
        W += wi;
        for (int k = 0; k < 3; k++) ma[k] += wi * A[k * n + i], mb[k] += wi * B[k * n + i];
    }
    PLX_ASSERT(W > 0);
    for (int k = 0; k < 3; k++) ma[k] /= W, mb[k] /= W;
    double C[9] = {0}, va = 0;
    for (int i = 0; i < n; i++) {
        double wi = w ? w[i] : 1;
        if (!(wi > 0)) continue;
        for (int r = 0; r < 3; r++) {
            double br = B[r * n + i] - mb[r];
            va += wi * (A[r * n + i] - ma[r]) * (A[r * n + i] - ma[r]);
            for (int c = 0; c < 3; c++) C[r * 3 + c] += wi * br * (A[c * n + i] - ma[c]);
        }
    }
    double U[9], sv[3], V[9];
    plx_svd3(C, U, sv, V);
    /* R = U diag(1, 1, d) Vᵀ with d = det(U Vᵀ) */
    double Vt[9], UVt[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) Vt[i * 3 + j] = V[j * 3 + i];
    mat3_mul(U, Vt, UVt);
    double det = UVt[0] * (UVt[4] * UVt[8] - UVt[5] * UVt[7]) - UVt[1] * (UVt[3] * UVt[8] - UVt[5] * UVt[6]) +
                 UVt[2] * (UVt[3] * UVt[7] - UVt[4] * UVt[6]);
    double d = det < 0 ? -1 : 1, Ud[9];
    memcpy(Ud, U, sizeof Ud);
    for (int i = 0; i < 3; i++) Ud[i * 3 + 2] *= d;
    mat3_mul(Ud, Vt, R);
    *s = va > 0 ? (sv[0] + sv[1] + d * sv[2]) / va : 1;
    for (int r = 0; r < 3; r++) t[r] = mb[r] - *s * (R[r * 3] * ma[0] + R[r * 3 + 1] * ma[1] + R[r * 3 + 2] * ma[2]);
    double e = 0;
    for (int i = 0; i < n; i++) {
        double wi = w ? w[i] : 1;
        if (!(wi > 0)) continue;
        for (int r = 0; r < 3; r++) {
            double p = *s * (R[r * 3] * A[i] + R[r * 3 + 1] * A[n + i] + R[r * 3 + 2] * A[2 * n + i]) + t[r];
            e += wi * (B[r * n + i] - p) * (B[r * n + i] - p);
        }
    }
    return sqrt(e / W);
}

void plx_fu_angles(const double R[9], double *yaw, double *pitch, double *roll) {
    /* R = Rz Rx Ry: row 2 = (-cφ sθ, sφ, cφ cθ); (R01, R11) = cφ (-sin ρ, cos ρ) */
    *pitch = asin(PLX_CLAMP(R[7], -1.0, 1.0));
    *yaw = atan2(-R[6], R[8]);
    *roll = atan2(-R[1], R[4]);
}

/* Rotation and scale of one frame from its two projection rows. */
static void frame_rotation(const double *m, double R[9], double *s) {
    double n1 = plx_vec_norm(m, 3), n2 = plx_vec_norm(m + 3, 3), r[9];
    *s = 0.5 * (n1 + n2);
    for (int k = 0; k < 3; k++) r[k] = m[k] / n1, r[3 + k] = m[3 + k] / n2;
    cross3(r, r + 3, r + 6);
    plx_nearest_rotation3(r, R);
}

/* Axis-angle vector of a rotation. */
static void rot_log(const double R[9], double w[3]) {
    double c = PLX_CLAMP(0.5 * (R[0] + R[4] + R[8] - 1), -1.0, 1.0), th = acos(c), sn = sin(th);
    double k = sn > 1e-9 ? th / (2 * sn) : 0.5;
    w[0] = k * (R[7] - R[5]);
    w[1] = k * (R[2] - R[6]);
    w[2] = k * (R[3] - R[1]);
}

/* ---- visibility -------------------------------------------------------------------------- */
void plx_fu_grid_visibility(const double *S, const uint8_t *valid, int nx, int ny, const double R[9], double s,
                            double tx, double ty, int w, int h, double *vis) {
    int P = nx * ny, nt = 0;
    plx_mesh m = plx_mesh_new(P, 2 * (nx - 1) * (ny - 1), 0);
    for (int i = 0; i < P; i++)
        for (int k = 0; k < 3; k++) m.v[i * 3 + k] = (float)(isfinite(S[k * P + i]) ? S[k * P + i] : 0);
    bool *ok = PLX_NEW(bool, P);
    for (int i = 0; i < P; i++) ok[i] = (!valid || valid[i]) && isfinite(S[i]) && isfinite(S[P + i]) && isfinite(S[2 * P + i]);
    for (int y = 0; y + 1 < ny; y++)
        for (int x = 0; x + 1 < nx; x++) {
            int a = y * nx + x, b = a + 1, c = a + nx, d = c + 1;
            if (!(ok[a] && ok[b] && ok[c] && ok[d])) continue;
            int *t = m.tri + 3 * nt;
            t[0] = a, t[1] = b, t[2] = c, t[3] = b, t[4] = d, t[5] = c;
            nt += 2;
        }
    m.nt = nt;
    plx_camera cam = {.s = s, .t = {tx, ty, 0}};
    memcpy(cam.R, R, sizeof cam.R);
    /* no culling: a height field is an open surface, and where its back is seen (the far flank
     * of the nose at a turn) that back is exactly what hides the cheek behind it */
    plx_raster_opts ro = {.w = w, .h = h, .cull = PLX_CULL_NONE, .bg = 0};
    plx_raster ras;
    plx_render(&m, &cam, &ro, &ras);
    for (int i = 0; i < P; i++) {
        vis[i] = 0;
        if (!ok[i]) continue;
        /* facing: grid normal from central (one-sided at the border) differences, camera frame */
        int gx = i % nx, gy = i / nx;
        int xa = gx > 0 && ok[i - 1] ? i - 1 : i, xb = gx + 1 < nx && ok[i + 1] ? i + 1 : i;
        int ya = gy > 0 && ok[i - nx] ? i - nx : i, yb = gy + 1 < ny && ok[i + nx] ? i + nx : i;
        if (xa == xb || ya == yb) continue;
        double u[3], v[3], nrm[3], nc[3];
        for (int k = 0; k < 3; k++) u[k] = S[k * P + xb] - S[k * P + xa], v[k] = S[k * P + yb] - S[k * P + ya];
        cross3(u, v, nrm);
        double nl = plx_vec_norm(nrm, 3);
        if (!(nl > 0)) continue;
        for (int r = 0; r < 3; r++) nc[r] = (R[r * 3] * nrm[0] + R[r * 3 + 1] * nrm[1] + R[r * 3 + 2] * nrm[2]) / nl;
        double facing = PLX_CLAMP(nc[2] / 0.5, 0.0, 1.0);
        if (facing <= 0) continue;
        /* occlusion: the z-buffer at the nearest pixel centre, which is up to half a pixel away
         * on each axis; the slack covers the surface's own depth slope over that offset */
        double p[3];
        plx_camera_project(&cam, m.v + 3 * i, p);
        int px = (int)lrint(p[0]), py = (int)lrint(p[1]);
        if (px < 0 || py < 0 || px >= w || py >= h) continue;
        float zb = PLX_PX(&ras.depth, px, py, 0);
        double tol = 2.0 * s + (fabs(nc[0]) + fabs(nc[1])) / fmax(nc[2], 0.1);
        if (isfinite(zb) && p[2] < zb - tol) continue;
        vis[i] = facing;
    }
    plx_raster_free(&ras);
    plx_mesh_free(&m);
    free(ok);
}

/* ---- one estimate --------------------------------------------------------------------------- */
typedef struct {
    double *M, *t, *S;      /* canonical gauge: [F][2][3], [F][2], [3][P] */
    double *R, *scale;      /* [F][9], [F] */
    double  gap, min_eig, spread[3], axis[3][3], ratio, prior_scale, corr, corr_mirror, gauge_rms;
    bool    flipped, degenerate;
} est;

static void est_alloc(est *e, int F, int P) {
    memset(e, 0, sizeof *e);
    e->M = PLX_NEW(double, (size_t)F * 6);
    e->t = PLX_NEW(double, (size_t)F * 2);
    e->S = PLX_NEW(double, (size_t)3 * P);
    e->R = PLX_NEW(double, (size_t)F * 9);
    e->scale = PLX_NEW(double, F);
}
static void est_free(est *e) {
    free(e->M), free(e->t), free(e->S), free(e->R), free(e->scale);
}

/* Weighted centred correlation and regression ratio ⟨a, b⟩ / ⟨b, b⟩ of a on b. */
static void corr_ratio(int P, const double *a, const double *b, const double *w, double *corr, double *ratio) {
    double W = 0, ma = 0, mb = 0;
    for (int n = 0; n < P; n++)
        if (w[n] > 0) W += w[n], ma += w[n] * a[n], mb += w[n] * b[n];
    ma /= W, mb /= W;
    double ab = 0, aa = 0, bb = 0;
    for (int n = 0; n < P; n++)
        if (w[n] > 0) {
            double x = a[n] - ma, y = b[n] - mb;
            ab += w[n] * x * y, aa += w[n] * x * x, bb += w[n] * y * y;
        }
    if (corr) *corr = aa > 0 && bb > 0 ? ab / sqrt(aa * bb) : 0;
    if (ratio) *ratio = bb > 0 ? ab / bb : 0;
}

/* Per-frame LS of (M_f, t_f) for a fixed shape. */
static void solve_motion(int F, int P, const double *x, const double *w, const double *S, double *M, double *t) {
    for (int f = 0; f < F; f++) {
        double H[16] = {0}, b[2][4] = {{0}};
        for (int n = 0; n < P; n++) {
            double wf = w[(size_t)f * P + n];
            if (wf <= 0) continue;
            double s4[4] = {S[n], S[P + n], S[2 * P + n], 1};
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) H[i * 4 + j] += wf * s4[i] * s4[j];
                for (int r = 0; r < 2; r++) b[r][i] += wf * s4[i] * x[((size_t)f * 2 + r) * P + n];
            }
        }
        for (int r = 0; r < 2; r++) {
            double Hc[16];
            memcpy(Hc, H, sizeof H);
            solve_small(4, Hc, b[r]);
            for (int i = 0; i < 3; i++) M[(f * 2 + r) * 3 + i] = b[r][i];
            t[f * 2 + r] = b[r][3];
        }
    }
}

typedef struct {
    int F, P;
    const double *x, *w;   /* tracks and the current effective weights */
    const double *X0, *Y0, *zp;
    const double *gw;      /* gauge weights (zero for unused nodes) */
    const plx_fu_factor_opts *o;
    int force_flip;        /* -1 decide, else 0/1 */
    int force_degen;       /* -1 decide, else 0/1 */
} est_in;

static int estimate(const est_in *in, est *e) {
    int F = in->F, P = in->P;
    double *Ma = PLX_NEW(double, (size_t)F * 6), *ta = PLX_NEW(double, (size_t)F * 2), *Sa = PLX_NEW(double, (size_t)3 * P);
    plx_fu_als(F, P, in->x, in->w, in->o->als_iters, in->o->als_tol, Ma, ta, Sa);
    double *fw = PLX_NEW(double, F);
    for (int f = 0; f < F; f++)
        for (int n = 0; n < P; n++) fw[f] += in->w[(size_t)f * P + n];
    double Q[9];
    int st = plx_fu_metric(F, Ma, fw, Q, &e->gap, &e->min_eig);
    free(fw);
    if (st != PLX_OK) {
        free(Ma), free(ta), free(Sa);
        return st;
    }
    /* M <- M Q, S <- Q⁻¹ S */
    double Qi[9];
    {
        plx_mat Qm = plx_mat_from(3, 3, Q), Qinv = plx_mat_new(3, 3);
        if (plx_inverse(&Qm, &Qinv) != PLX_OK) {
            plx_mat_free(&Qm), plx_mat_free(&Qinv);
            free(Ma), free(ta), free(Sa);
            return PLX_EINVAL;
        }
        memcpy(Qi, Qinv.a, sizeof Qi);
        plx_mat_free(&Qm), plx_mat_free(&Qinv);
    }
    for (int r = 0; r < 2 * F; r++) {
        double m[3];
        for (int j = 0; j < 3; j++) m[j] = Ma[r * 3] * Q[j] + Ma[r * 3 + 1] * Q[3 + j] + Ma[r * 3 + 2] * Q[6 + j];
        memcpy(Ma + r * 3, m, sizeof m);
    }
    for (int n = 0; n < P; n++) {
        double v[3] = {Sa[n], Sa[P + n], Sa[2 * P + n]};
        for (int i = 0; i < 3; i++) Sa[i * P + n] = Qi[i * 3] * v[0] + Qi[i * 3 + 1] * v[1] + Qi[i * 3 + 2] * v[2];
    }

    /* mirror choice + canonical gauge: Procrustes of S and of its depth reflection onto (X0, Y0, zp) */
    double *B = PLX_NEW(double, (size_t)3 * P), *Sm = PLX_NEW(double, (size_t)3 * P), *Sc[2];
    for (int n = 0; n < P; n++) B[n] = in->X0[n], B[P + n] = in->Y0[n], B[2 * P + n] = in->zp[n];
    double gs[2], gR[2][9], gt[2][3], grms[2], corr[2];
    for (int c = 0; c < 2; c++) {
        Sc[c] = PLX_NEW(double, (size_t)3 * P);
        for (int n = 0; n < P; n++)
            Sm[n] = Sa[n], Sm[P + n] = Sa[P + n], Sm[2 * P + n] = c ? -Sa[2 * P + n] : Sa[2 * P + n];
        grms[c] = plx_fu_procrustes(P, Sm, B, in->gw, &gs[c], gR[c], gt[c]);
        for (int n = 0; n < P; n++)
            for (int r = 0; r < 3; r++)
                Sc[c][r * P + n] = gs[c] * (gR[c][r * 3] * Sm[n] + gR[c][r * 3 + 1] * Sm[P + n] + gR[c][r * 3 + 2] * Sm[2 * P + n]) + gt[c][r];
        corr_ratio(P, Sc[c] + 2 * P, in->zp, in->gw, &corr[c], NULL);
    }
    int c = in->force_flip >= 0 ? in->force_flip : corr[1] > corr[0];
    e->flipped = c == 1;
    e->corr = corr[c], e->corr_mirror = corr[1 - c], e->gauge_rms = grms[c];
    memcpy(e->S, Sc[c], sizeof(double) * 3 * P);
    /* motion in the gauge: S_c = s R F S_a + tg  =>  M_c = M_a F Rᵀ / s, t_c = t_a - M_c tg */
    for (int f = 0; f < F; f++)
        for (int r = 0; r < 2; r++) {
            const double *ma = Ma + (f * 2 + r) * 3;
            double mf[3] = {ma[0], ma[1], c ? -ma[2] : ma[2]}, *mc = e->M + (f * 2 + r) * 3;
            for (int j = 0; j < 3; j++) mc[j] = (mf[0] * gR[c][j * 3] + mf[1] * gR[c][j * 3 + 1] + mf[2] * gR[c][j * 3 + 2]) / gs[c];
            e->t[f * 2 + r] = ta[f * 2 + r] - (mc[0] * gt[c][0] + mc[1] * gt[c][1] + mc[2] * gt[c][2]);
        }
    for (int k = 0; k < 2; k++) free(Sc[k]);
    free(B), free(Sm), free(Ma), free(ta), free(Sa);

    /* rotations and the spread of their axes about the mean rotation */
    double Rsum[9] = {0}, Rbar[9];
    for (int f = 0; f < F; f++) {
        frame_rotation(e->M + f * 6, e->R + f * 9, &e->scale[f]);
        for (int k = 0; k < 9; k++) Rsum[k] += e->R[f * 9 + k];
    }
    plx_nearest_rotation3(Rsum, Rbar);
    plx_mat C = plx_mat_new(3, 3), E = plx_mat_new(3, 3);
    for (int f = 0; f < F; f++) {
        double Rt[9], D[9], wv[3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) Rt[i * 3 + j] = Rbar[j * 3 + i];
        mat3_mul(e->R + f * 9, Rt, D);
        rot_log(D, wv);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) PLX_M(&C, i, j) += wv[i] * wv[j] / F;
    }
    double ev[3];
    plx_eigh(&C, ev, &E);
    for (int k = 0; k < 3; k++) {
        e->spread[k] = sqrt(fmax(ev[k], 0)) / DEG;
        for (int j = 0; j < 3; j++) e->axis[k][j] = PLX_M(&E, j, k);
    }
    plx_mat_free(&C), plx_mat_free(&E);

    /* bas-relief: depth scale from the population pattern when the motion cannot fix it */
    corr_ratio(P, e->S + 2 * P, in->zp, in->gw, NULL, &e->ratio);
    bool degen = in->o->prior_scale == 1 ||
                 (in->o->prior_scale < 0 && (e->spread[1] < in->o->degen_spread_deg || e->gap < in->o->degen_gap));
    if (in->force_degen >= 0) degen = in->force_degen == 1;
    e->degenerate = degen;
    e->prior_scale = 1;
    if (degen && e->ratio > 0.05) {
        double W = 0, mz = 0;
        for (int n = 0; n < P; n++)
            if (in->gw[n] > 0) W += in->gw[n], mz += in->gw[n] * e->S[2 * P + n];
        mz /= W;
        e->prior_scale = 1 / e->ratio;
        for (int n = 0; n < P; n++) e->S[2 * P + n] = mz + (e->S[2 * P + n] - mz) * e->prior_scale;
        solve_motion(F, P, in->x, in->w, e->S, e->M, e->t);
        for (int f = 0; f < F; f++) frame_rotation(e->M + f * 6, e->R + f * 9, &e->scale[f]);
    }
    return PLX_OK;
}

/* ---- the robust loop and the bootstrap -------------------------------------------------------- */
static double residual(const double *x, int F, int P, const est *e, int f, int n) {
    double d2 = 0;
    for (int r = 0; r < 2; r++) {
        const double *m = e->M + (f * 2 + r) * 3;
        double d = x[((size_t)f * 2 + r) * P + n] - e->t[f * 2 + r] -
                   (m[0] * e->S[n] + m[1] * e->S[P + n] + m[2] * e->S[2 * P + n]);
        d2 += d * d;
    }
    return sqrt(d2);
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *v, int n) {
    if (n <= 0) return 0;
    qsort(v, n, sizeof(double), cmp_double);
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

typedef struct {
    const est_in *in;
    const est    *main;
    const double *noise_sd; /* [F][P] per-coordinate noise sd of each entry */
    double       *Sb;       /* [B][3][P] */
    uint8_t      *ok;       /* [B] */
    uint64_t      seed;
} boot_ctx;

static void boot_one(int64_t b, int tid, void *pc) {
    boot_ctx *c = pc;
    const est_in *in = c->in;
    int F = in->F, P = in->P;
    plx_rng rng;
    plx_rng_seed(&rng, c->seed + 0x9e3779b97f4a7c15ull * (uint64_t)(b + 1));
    double *xb = PLX_NEW(double, (size_t)F * 2 * P);
    for (int f = 0; f < F; f++)
        for (int n = 0; n < P; n++) {
            if (in->w[(size_t)f * P + n] <= 0) continue;
            for (int r = 0; r < 2; r++) {
                const double *m = c->main->M + (f * 2 + r) * 3;
                double mu = c->main->t[f * 2 + r] + m[0] * c->main->S[n] + m[1] * c->main->S[P + n] +
                            m[2] * c->main->S[2 * P + n];
                xb[((size_t)f * 2 + r) * P + n] = mu + c->noise_sd[(size_t)f * P + n] * plx_rng_normal(&rng);
            }
        }
    est_in ib = *in;
    ib.x = xb;
    ib.force_flip = c->main->flipped;
    ib.force_degen = c->main->degenerate;
    est e;
    est_alloc(&e, F, P);
    c->ok[b] = estimate(&ib, &e) == PLX_OK;
    if (c->ok[b]) memcpy(c->Sb + (size_t)b * 3 * P, e.S, sizeof(double) * 3 * P);
    est_free(&e);
    free(xb);
}

int plx_fu_factorise(const plx_fu_tracks *tr, const plx_fu_factor_opts *o, plx_fu_shape *out) {
    int F = tr->F, P = tr->P;
    memset(out, 0, sizeof *out);
    if (F < 3) return PLX_EINVAL;
    bool grid = tr->nx > 0 && tr->ny > 0 && tr->nx * tr->ny == P;
    uint8_t *used = PLX_NEW(uint8_t, P);
    double *gw = PLX_NEW(double, P);
    int n_used = 0;
    for (int n = 0; n < P; n++) {
        int nf = 0;
        for (int f = 0; f < F; f++) nf += tr->w[(size_t)f * P + n] > 0;
        used[n] = nf >= 2 && isfinite(tr->zp[n]);
        gw[n] = used[n] ? (tr->gw ? tr->gw[n] : 1) : 0;
        n_used += used[n];
    }
    if (n_used < 4) {
        free(used), free(gw);
        return PLX_EINVAL;
    }
    double *w = PLX_NEW(double, (size_t)F * P), *vis = PLX_NEW(double, (size_t)F * P);
    for (size_t k = 0; k < (size_t)F * P; k++) w[k] = used[k % P] ? tr->w[k] : 0;
    est e;
    est_alloc(&e, F, P);
    est_in in = {.F = F, .P = P, .x = tr->x, .w = w, .X0 = tr->X0, .Y0 = tr->Y0, .zp = tr->zp, .gw = gw, .o = o,
                 .force_flip = -1, .force_degen = -1};
    double *ebuf = PLX_NEW(double, (size_t)F * P + 1), *nrms = PLX_NEW(double, P), sigma = 0;
    int rounds = PLX_MAX(o->rounds, 1), st = PLX_OK;
    for (int round = 0; round < rounds; round++) {
        if ((st = estimate(&in, &e)) != PLX_OK) break;
        /* robust noise level from the entries in use */
        int m = 0;
        for (int f = 0; f < F; f++)
            for (int n = 0; n < P; n++)
                if (w[(size_t)f * P + n] > 0) ebuf[m++] = residual(tr->x, F, P, &e, f, n);
        sigma = median(ebuf, m) / 1.1774; /* median of a 2-D Gaussian residual norm = 1.1774 σ */
        for (int n = 0; n < P; n++) {
            double sw = 0, se = 0;
            for (int f = 0; f < F; f++) {
                double wf = w[(size_t)f * P + n];
                if (wf <= 0) continue;
                double r = residual(tr->x, F, P, &e, f, n);
                sw += wf, se += wf * r * r;
            }
            nrms[n] = sw > 0 ? sqrt(se / sw) : NAN;
        }
        if (round == rounds - 1) break;
        m = 0;
        for (int n = 0; n < P; n++)
            if (used[n] && isfinite(nrms[n])) ebuf[m++] = nrms[n];
        double med_rms = median(ebuf, m);
        for (int f = 0; f < F; f++) {
            if (grid && o->visibility)
                plx_fu_grid_visibility(e.S, used, tr->nx, tr->ny, e.R + f * 9, e.scale[f], e.t[f * 2], e.t[f * 2 + 1],
                                       tr->img_w, tr->img_h, vis + (size_t)f * P);
            for (int n = 0; n < P; n++) {
                size_t k = (size_t)f * P + n;
                if (!used[n] || tr->w[k] <= 0) {
                    w[k] = 0;
                    continue;
                }
                double r = residual(tr->x, F, P, &e, f, n) / (o->robust_k * fmax(sigma, 1e-9));
                double wk = tr->w[k] / (1 + r * r);
                if (o->nonrigid_k > 0 && med_rms > 0) {
                    double q = nrms[n] / (o->nonrigid_k * med_rms);
                    wk /= 1 + q * q;
                }
                if (grid && o->visibility) wk *= vis[k];
                w[k] = wk;
            }
        }
    }
    if (st != PLX_OK) {
        est_free(&e);
        free(used), free(gw), free(w), free(vis), free(ebuf), free(nrms);
        return st;
    }

    /* outputs */
    out->F = F, out->P = P;
    out->S = PLX_NEW(double, (size_t)3 * P);
    out->sd = PLX_NEW(double, (size_t)3 * P);
    out->R = PLX_NEW(double, (size_t)F * 9);
    out->M = PLX_NEW(double, (size_t)F * 6);
    out->scale = PLX_NEW(double, F), out->tx = PLX_NEW(double, F), out->ty = PLX_NEW(double, F);
    out->yaw = PLX_NEW(double, F), out->pitch = PLX_NEW(double, F), out->roll = PLX_NEW(double, F);
    out->w = w;
    out->vis = vis;
    out->res = PLX_NEW(double, (size_t)F * P);
    out->node_rms = nrms;
    out->frame_rms = PLX_NEW(double, F);
    out->used = used;
    memcpy(out->M, e.M, sizeof(double) * F * 6);
    memcpy(out->R, e.R, sizeof(double) * F * 9);
    for (int f = 0; f < F; f++) {
        out->scale[f] = e.scale[f], out->tx[f] = e.t[f * 2], out->ty[f] = e.t[f * 2 + 1];
        plx_fu_angles(e.R + f * 9, &out->yaw[f], &out->pitch[f], &out->roll[f]);
    }
    if (grid)
        for (int f = 0; f < F; f++)
            plx_fu_grid_visibility(e.S, used, tr->nx, tr->ny, e.R + f * 9, e.scale[f], e.t[f * 2], e.t[f * 2 + 1],
                                   tr->img_w, tr->img_h, vis + (size_t)f * P);
    else
        for (size_t k = 0; k < (size_t)F * P; k++) vis[k] = 1;
    double sw = 0, se = 0;
    for (int f = 0; f < F; f++) {
        double fw = 0, fe = 0;
        for (int n = 0; n < P; n++) {
            size_t k = (size_t)f * P + n;
            out->res[k] = used[n] && tr->w[k] > 0 ? residual(tr->x, F, P, &e, f, n) : NAN;
            if (w[k] > 0) fw += w[k], fe += w[k] * out->res[k] * out->res[k];
        }
        out->frame_rms[f] = fw > 0 ? sqrt(fe / fw) : NAN;
        sw += fw, se += fe;
    }
    out->rms = sw > 0 ? sqrt(se / sw) : NAN;
    out->sigma = sigma;
    out->gap = e.gap;
    out->L_min_eig = e.min_eig;
    memcpy(out->spread_deg, e.spread, sizeof e.spread);
    memcpy(out->axis, e.axis, sizeof e.axis);
    out->degenerate = e.degenerate;
    out->depth_ratio = e.ratio;
    out->prior_scale = e.prior_scale;
    out->corr = e.corr, out->corr_mirror = e.corr_mirror;
    out->gauge_rms = e.gauge_rms;
    for (int n = 0; n < P; n++)
        for (int r = 0; r < 3; r++) out->S[r * P + n] = used[n] ? e.S[r * P + n] : NAN;

    /* bootstrap: per-entry noise sd = per-node level (shrunk to the pooled one) / sqrt(weight) */
    for (size_t k = 0; k < (size_t)3 * P; k++) out->sd[k] = NAN;
    if (o->boot > 1) {
        double pooled_num = 0, pooled_den = 0;
        double *tau2 = PLX_NEW(double, P);
        for (int n = 0; n < P; n++) {
            double num = 0;
            int nf = 0;
            for (int f = 0; f < F; f++) {
                size_t k = (size_t)f * P + n;
                if (w[k] <= 0) continue;
                num += w[k] * out->res[k] * out->res[k];
                nf++;
            }
            /* 2 nf coordinates, 3 of them spent on the node's own position */
            tau2[n] = 2 * nf > 3 ? num / (2 * nf - 3) : NAN;
            pooled_num += num, pooled_den += fmax(2 * nf - 3, 0);
        }
        double pooled = pooled_den > 0 ? pooled_num / pooled_den : 0, nu0 = 6;
        double *nsd = PLX_NEW(double, (size_t)F * P);
        for (int n = 0; n < P; n++) {
            int nf = 0;
            for (int f = 0; f < F; f++) nf += w[(size_t)f * P + n] > 0;
            double dofn = fmax(2 * nf - 3, 0), t2 = isfinite(tau2[n]) ? (dofn * tau2[n] + nu0 * pooled) / (dofn + nu0) : pooled;
            for (int f = 0; f < F; f++) {
                size_t k = (size_t)f * P + n;
                nsd[k] = w[k] > 0 ? sqrt(t2 / w[k]) : 0;
            }
        }
        boot_ctx bc = {.in = &in, .main = &e, .noise_sd = nsd, .Sb = PLX_NEW(double, (size_t)o->boot * 3 * P),
                       .ok = PLX_NEW(uint8_t, o->boot), .seed = o->seed};
        plx_parallel_for(o->boot, 1, boot_one, &bc);
        int nb = 0;
        double *acc = PLX_NEW(double, (size_t)3 * P), *acc2 = PLX_NEW(double, (size_t)3 * P), rs = 0, rs2 = 0;
        for (int b = 0; b < o->boot; b++) {
            if (!bc.ok[b]) continue;
            const double *Sb = bc.Sb + (size_t)b * 3 * P;
            for (size_t k = 0; k < (size_t)3 * P; k++) acc[k] += Sb[k], acc2[k] += Sb[k] * Sb[k];
            double ratio;
            corr_ratio(P, Sb + 2 * P, e.S + 2 * P, gw, NULL, &ratio);
            rs += ratio, rs2 += ratio * ratio;
            nb++;
        }
        if (nb > 1) {
            for (int n = 0; n < P; n++)
                for (int r = 0; r < 3; r++) {
                    size_t k = (size_t)r * P + n;
                    double m = acc[k] / nb;
                    out->sd[k] = used[n] ? sqrt(fmax(acc2[k] / nb - m * m, 0) * nb / (nb - 1)) : NAN;
                }
            double m = rs / nb;
            out->scale_sd = sqrt(fmax(rs2 / nb - m * m, 0) * nb / (nb - 1));
        }
        out->n_boot = nb;
        free(acc), free(acc2), free(bc.Sb), free(bc.ok), free(nsd), free(tau2);
    }
    est_free(&e);
    free(gw), free(ebuf);
    return PLX_OK;
}

void plx_fu_shape_free(plx_fu_shape *s) {
    free(s->S), free(s->sd), free(s->R), free(s->M), free(s->scale), free(s->tx), free(s->ty);
    free(s->yaw), free(s->pitch), free(s->roll), free(s->w), free(s->vis), free(s->res), free(s->node_rms);
    free(s->frame_rms), free(s->used);
    memset(s, 0, sizeof *s);
}
