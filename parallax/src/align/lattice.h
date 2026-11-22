/* Lattice displacement fields on the canonical frame Ω (docs/DESIGN.md §2, §4).
 *
 * A field u: Ω -> R² (canonical pixels) is stored at the nodes of a regular lattice and
 * interpolated bilinearly (hat basis functions B_n). A lattice of spacing h has nodes at
 *     x_j = -0.5 + h j  (j = 0 .. 128/h),   y_i = -0.5 + h i  (i = 0 .. 160/h),
 * so it spans exactly [-0.5, 127.5] x [-0.5, 159.5] (the pixel AREA of Ω) for h in {4, 8, 16, 32}:
 * 8 px gives the 17 x 21 = 357-node lattice of the design, 16 px 9 x 11, 32 px 5 x 6. All share
 * the origin, so a coarse field is represented exactly on any finer lattice (plx_lat_resample is
 * then exact refinement). The node set is σ-symmetric: node column j mirrors to nx - 1 - j.
 * Outside the lattice (the margin of Ω+) a field is extended by clamping the evaluation point
 * into the lattice rectangle.
 *
 * STORAGE. float u[2 * ny * nx], component-major: u[c * ny * nx + i * nx + j] is component c
 * (0 = x, 1 = y) at row i, column j. This is the [2, 21, 17] layout of flow/u.plxt.
 *
 * MIRROR. The twin of a field under σ(x, y) = (127 - x, y) is (σ*u)(p) = σ*(u(σp)) with
 * σ*(vx, vy) = (-vx, vy): if W(p) = T(p + u(p)) aligns a photo, then twin(T) and the twin field
 * align its left-right flip (canon.h's twin rule, applied pointwise). */
#ifndef PLX_ALIGN_LATTICE_H
#define PLX_ALIGN_LATTICE_H

#include "align/canon.h"
#include "la/la.h"

typedef struct {
    int    nx, ny; /* columns, rows */
    double step;   /* spacing, canonical px */
} plx_lattice;

#define PLX_LAT_ORIGIN (-0.5) /* x and y of node (0, 0) */
#define PLX_LAT_MAX_NODES (33 * 41)

plx_lattice plx_lattice_make(int step); /* step in {4, 8, 16, 32} */
static inline int plx_lat_nodes(const plx_lattice *L) { return L->nx * L->ny; }
static inline int plx_lat_dof(const plx_lattice *L) { return 2 * L->nx * L->ny; }
static inline double plx_lat_x(const plx_lattice *L, int j) { return PLX_LAT_ORIGIN + L->step * j; }
static inline double plx_lat_y(const plx_lattice *L, int i) { return PLX_LAT_ORIGIN + L->step * i; }

/* The four hat functions that are non-zero at (x, y) (clamped into the lattice): node indices
 * (i * nx + j) and weights, summing to 1. */
typedef struct { int n[4]; float w[4]; } plx_lat_taps;
void plx_lat_taps_at(const plx_lattice *L, double x, double y, plx_lat_taps *t);
static inline void plx_lat_eval_taps(const plx_lattice *L, const float *u, const plx_lat_taps *t, double v[2]) {
    const float *uy = u + plx_lat_nodes(L);
    v[0] = (double)t->w[0] * u[t->n[0]] + (double)t->w[1] * u[t->n[1]] + (double)t->w[2] * u[t->n[2]] +
           (double)t->w[3] * u[t->n[3]];
    v[1] = (double)t->w[0] * uy[t->n[0]] + (double)t->w[1] * uy[t->n[1]] + (double)t->w[2] * uy[t->n[2]] +
           (double)t->w[3] * uy[t->n[3]];
}
/* u at canonical (x, y): bilinear inside, clamped outside. */
void plx_lat_eval(const plx_lattice *L, const float *u, double x, double y, double v[2]);

/* out (on lattice `to`) = u (on lattice `from`) evaluated at the nodes of `to`. Exact (the same
 * function) when `to` refines `from`. */
void plx_lat_resample(const plx_lattice *from, const float *u, const plx_lattice *to, float *out);

/* Composition at the nodes: out(p_n) = d(p_n) + u(p_n + d(p_n)). If the photo is currently
 * sampled at T(p + u(p)) and the image-side increment d means "sample at p + d(p) instead",
 * then out is the new total field. out must not alias d or u. */
void plx_lat_compose(const plx_lattice *L, const float *d, const float *u, float *out);

/* σ-twin of a field (see header comment). out may alias u. */
void plx_lat_mirror(const plx_lattice *L, const float *u, float *out);

/* Bending energy of the field,
 *     E(u) = Σ_c Σ [ (Δxx u_c)² + (Δyy u_c)² + 2 (Δxy u_c)² ] / h²,
 * over all second differences Δxx (u[j-1] - 2u[j] + u[j+1] along a row), Δyy (along a column)
 * and mixed differences Δxy (u[i,j] - u[i,j+1] - u[i+1,j] + u[i+1,j+1]) that fit on the
 * lattice. The 1/h² makes it the discretisation of ∫ u_xx² + 2 u_xy² + u_yy² dA, so a weight
 * means the same on every lattice. Its null space is exactly the affine fields (each component
 * in span{1, x, y}): foreshortening and shear are free. σ-symmetric. */
double plx_lat_bending_energy(const plx_lattice *L, const float *u);
/* The matrix R (dof x dof, both triangles) with E(u) = uᵀ R u. */
void plx_lat_bending_matrix(const plx_lattice *L, plx_mat *R);
/* out (dof doubles) = R u, from the sparse stencils (cheap: no dense product). */
void plx_lat_bending_apply(const plx_lattice *L, const float *u, double *out);

/* ---- similarity gauge -------------------------------------------------------------------
 * The gauge is defined on a FIXED set of points independent of the lattice: the 8-px grid
 * points inside the face core, x in {23.5, 31.5, ..., 103.5}, y in {23.5, ..., 135.5} (11 x 15,
 * σ-symmetric), equally weighted. plx_lat_fit_sim returns the canonical similarity S
 * (canon.h) minimising Σ |S(p) - (p + u(p))|² over them. */
plx_sim plx_lat_fit_sim(const plx_lattice *L, const float *u);
/* u <- S⁻¹(p + u(p)) - p at the nodes, for the S fitted above (returned). With T <- T o S the
 * full warp W(p) = T(p + u(p)) is unchanged everywhere on the lattice rectangle (S⁻¹ is affine
 * and bilinear weights sum to 1), and u has no similarity component left. */
plx_sim plx_lat_sim_gauge(const plx_lattice *L, float *u);
/* u <- S⁻¹(p + u(p)) - p at the nodes for a given canonical similarity S: the field that, with
 * T o S, gives the same full warp as u with T. */
void plx_lat_csim_pullback(const plx_lattice *L, plx_sim S, float *u);

/* Minimum over the face-core box [23.5, 103.5] x [23.5, 135.5] of det(I + ∇u): the Jacobian
 * determinant of W relative to its similarity. Within a lattice cell u is bilinear, so the
 * determinant is affine in (x, y) and its minimum over (cell ∩ core) is at a corner: exact. */
double plx_lat_min_jdet(const plx_lattice *L, const float *u);

/* Largest |u| over lattice nodes inside the core box. */
double plx_lat_max_abs(const plx_lattice *L, const float *u);

/* The core box (σ-symmetric) used by the gauge and the field statistics. */
#define PLX_LAT_CORE_X0 23.5
#define PLX_LAT_CORE_X1 103.5
#define PLX_LAT_CORE_Y0 23.5
#define PLX_LAT_CORE_Y1 135.5

#endif
