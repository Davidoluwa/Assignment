/* Preconditioned conjugate gradient (Hestenes-Stiefel) for SPD systems given as a matvec. */
#include "la/la_internal.h"

#include <stdlib.h>

int plx_cg(int n, plx_matvec_fn matvec, void *ctx, const double *b, double *x, const double *diag,
           double tol, int max_iter, plx_cg_info *info) {
    PLX_ASSERT(n >= 1 && tol >= 0);
    if (max_iter <= 0) max_iter = 2 * n;
    double *r = plx_malloc(sizeof(double) * n), *z = plx_malloc(sizeof(double) * n);
    double *p = plx_malloc(sizeof(double) * n), *Ap = plx_malloc(sizeof(double) * n);
    double *inv = NULL;
    if (diag) {
        inv = plx_malloc(sizeof(double) * n);
        for (int i = 0; i < n; i++) {
            PLX_ASSERT(diag[i] > 0);
            inv[i] = 1.0 / diag[i];
        }
    }
    const double bnorm = plx_vec_norm(b, (size_t)n);
    matvec(x, Ap, ctx);
    for (int i = 0; i < n; i++) {
        r[i] = b[i] - Ap[i];
        z[i] = inv ? inv[i] * r[i] : r[i];
        p[i] = z[i];
    }
    double rz = la_dot(r, z, (size_t)n);
    int rc = PLX_ENOCONV, it = 0;
    for (;; it++) {
        if (plx_vec_norm(r, (size_t)n) <= tol * bnorm) {
            rc = PLX_OK;
            break;
        }
        if (it == max_iter) break;
        matvec(p, Ap, ctx);
        double pAp = la_dot(p, Ap, (size_t)n);
        if (!(pAp > 0)) {
            rc = PLX_ENOTSPD;
            break;
        }
        double alpha = rz / pAp;
        la_axpy(x, alpha, p, (size_t)n);
        la_axpy(r, -alpha, Ap, (size_t)n);
        for (int i = 0; i < n; i++) z[i] = inv ? inv[i] * r[i] : r[i];
        double rz_new = la_dot(r, z, (size_t)n), beta = rz_new / rz;
        rz = rz_new;
        for (int i = 0; i < n; i++) p[i] = fma(beta, p[i], z[i]);
    }
    if (info) { /* the recurrence residual drifts from the true one; report the true one */
        matvec(x, Ap, ctx);
        for (int i = 0; i < n; i++) r[i] = b[i] - Ap[i];
        info->iters = it;
        info->rel_residual = bnorm > 0 ? plx_vec_norm(r, (size_t)n) / bnorm : plx_vec_norm(r, (size_t)n);
    }
    free(r);
    free(z);
    free(p);
    free(Ap);
    free(inv);
    return rc;
}
