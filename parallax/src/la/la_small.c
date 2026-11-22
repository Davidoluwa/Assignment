/* Fixed-size kernels for per-point / per-frame work: no allocation, no threading. */
#include "la/la_internal.h"

#include <string.h>

/* Jacobi rotation that diagonalises [[a, b], [b, c]]: with t the smaller root of
 * t^2 + 2 zeta t - 1 = 0, zeta = (c - a) / (2b), the eigenvalues are a - t b and c + t b. Neither
 * formula subtracts nearly equal quantities, unlike (a+c)/2 -+ sqrt(...). */
void plx_eig2_sym(double a, double b, double c, double w[2], double V[4]) {
    double cs = 1, sn = 0, l1 = a, l2 = c;
    if (b != 0) {
        double zeta = (c - a) / (2.0 * b);
        double t = copysign(1.0, zeta) / (fabs(zeta) + hypot(1.0, zeta));
        cs = 1.0 / hypot(1.0, t);
        sn = cs * t;
        l1 = a - t * b;
        l2 = c + t * b;
    }
    /* eigenvector of l1 is (cs, -sn), of l2 is (sn, cs) */
    if (l1 >= l2) {
        w[0] = l1; w[1] = l2;
        V[0] = cs; V[1] = sn;
        V[2] = -sn; V[3] = cs;
    } else { /* swap columns and negate the new second one to keep det = +1 */
        w[0] = l2; w[1] = l1;
        V[0] = sn; V[1] = -cs;
        V[2] = cs; V[3] = sn;
    }
}

static void cross3(const double *a, const double *b, double *out) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

/* One-sided Jacobi on the columns of W = A J: converges in a few sweeps for 3x3, and gives the
 * small singular values to relative accuracy (no A^T A). The left vectors come from normalised
 * columns; the third is the cross product of the first two, so U is orthogonal even when A is
 * rank deficient. */
void plx_svd3(const double A[9], double U[9], double s[3], double V[9]) {
    double W[9], J[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    memcpy(W, A, sizeof W);
    static const int P[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    for (int sweep = 0; sweep < 30; sweep++) {
        int rotated = 0;
        for (int k = 0; k < 3; k++) {
            const int p = P[k][0], q = P[k][1];
            double al = 0, be = 0, ga = 0;
            for (int i = 0; i < 3; i++) {
                al += W[3 * i + p] * W[3 * i + p];
                be += W[3 * i + q] * W[3 * i + q];
                ga += W[3 * i + p] * W[3 * i + q];
            }
            if (!(fabs(ga) > DBL_EPSILON * sqrt(al) * sqrt(be))) continue;
            rotated = 1;
            double zeta = (be - al) / (2.0 * ga);
            double t = copysign(1.0, zeta) / (fabs(zeta) + hypot(1.0, zeta));
            double c = 1.0 / hypot(1.0, t), sn = c * t;
            for (int i = 0; i < 3; i++) {
                double wp = W[3 * i + p], wq = W[3 * i + q];
                W[3 * i + p] = c * wp - sn * wq;
                W[3 * i + q] = sn * wp + c * wq;
                double jp = J[3 * i + p], jq = J[3 * i + q];
                J[3 * i + p] = c * jp - sn * jq;
                J[3 * i + q] = sn * jp + c * jq;
            }
        }
        if (!rotated) break;
    }
    double sv[3], col[3][3];
    for (int j = 0; j < 3; j++) sv[j] = hypot(hypot(W[j], W[3 + j]), W[6 + j]);
    int o[3] = {0, 1, 2};
    for (int i = 0; i < 3; i++) /* sort descending, 3 elements */
        for (int j = i + 1; j < 3; j++)
            if (sv[o[j]] > sv[o[i]]) { int t = o[i]; o[i] = o[j]; o[j] = t; }
    for (int j = 0; j < 3; j++) {
        s[j] = sv[o[j]];
        for (int i = 0; i < 3; i++) {
            col[j][i] = W[3 * i + o[j]];
            V[3 * i + j] = J[3 * i + o[j]];
        }
    }
    double u[3][3];
    if (s[0] > 0) {
        for (int i = 0; i < 3; i++) u[0][i] = col[0][i] / s[0];
    } else {
        u[0][0] = 1; u[0][1] = u[0][2] = 0;
    }
    if (s[1] > DBL_EPSILON * s[0]) {
        for (int i = 0; i < 3; i++) u[1][i] = col[1][i] / s[1];
    } else { /* any unit vector orthogonal to u0: Gram-Schmidt the least aligned axis */
        int e = 0;
        for (int i = 1; i < 3; i++)
            if (fabs(u[0][i]) < fabs(u[0][e])) e = i;
        double d = u[0][e], nrm = 0;
        for (int i = 0; i < 3; i++) {
            u[1][i] = (i == e) - d * u[0][i];
            nrm += u[1][i] * u[1][i];
        }
        for (int i = 0; i < 3; i++) u[1][i] /= sqrt(nrm);
    }
    cross3(u[0], u[1], u[2]);
    if (u[2][0] * col[2][0] + u[2][1] * col[2][1] + u[2][2] * col[2][2] < 0)
        for (int i = 0; i < 3; i++) u[2][i] = -u[2][i];
    for (int j = 0; j < 3; j++)
        for (int i = 0; i < 3; i++) U[3 * i + j] = u[j][i];
}

static double det3(const double *M) {
    return M[0] * (M[4] * M[8] - M[5] * M[7]) - M[1] * (M[3] * M[8] - M[5] * M[6]) + M[2] * (M[3] * M[7] - M[4] * M[6]);
}

/* Orthogonal Procrustes: the closest orthogonal matrix is U V^T; if that is a reflection, flipping
 * the direction of the smallest singular value costs the least, giving U diag(1,1,-1) V^T. */
void plx_nearest_rotation3(const double A[9], double R[9]) {
    double U[9], s[3], V[9];
    plx_svd3(A, U, s, V);
    const double d = det3(U) * det3(V) < 0 ? -1.0 : 1.0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R[3 * i + j] = U[3 * i] * V[3 * j] + U[3 * i + 1] * V[3 * j + 1] + d * U[3 * i + 2] * V[3 * j + 2];
}
