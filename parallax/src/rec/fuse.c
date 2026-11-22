/* Score-level fusion: per-split score columns, logistic and fixed-weight fusion. */
#include "rec/fuse.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int plx_logreg_fit(const plx_mat *Phi, const int *y, double l2, double *w) {
    int n = Phi->rows, F = Phi->cols;
    PLX_ASSERT(n >= 1 && F >= 1);
    memset(w, 0, sizeof(double) * F);
    double *g = PLX_NEW(double, F);
    plx_mat H = plx_mat_new(F, F), Pw = plx_mat_new(n, F);
    int rc = PLX_OK;
    for (int it = 0; it < 50; it++) {
        memset(g, 0, sizeof(double) * F);
        for (int p = 0; p < n; p++) {
            const double *x = &PLX_M(Phi, p, 0);
            double pr = 1.0 / (1.0 + exp(-plx_vec_dot(x, w, (size_t)F)));
            double res = (pr - (y[p] != 0)) / n, sw = sqrt(pr * (1.0 - pr) / n);
            plx_vec_axpy(g, res, x, (size_t)F);
            for (int j = 0; j < F; j++) PLX_M(&Pw, p, j) = sw * x[j];
        }
        plx_syrk(PLX_TRANS, 1.0, &Pw, 0.0, &H);
        for (int j = 0; j < F; j++) {
            double r = j < F - 1 ? l2 : 1e-9; /* the bias is (almost) unpenalised */
            g[j] += r * w[j];
            PLX_M(&H, j, j) += r;
        }
        rc = plx_chol(&H);
        if (rc != PLX_OK) {
            plx_log("logistic regression: Hessian not SPD");
            break;
        }
        plx_chol_solve_vec(&H, g);
        plx_vec_axpy(w, -1.0, g, (size_t)F);
        if (plx_vec_norm(g, (size_t)F) < 1e-9 * (1.0 + plx_vec_norm(w, (size_t)F))) break;
    }
    free(g);
    plx_mat_free(&H);
    plx_mat_free(&Pw);
    return rc;
}

static void add_col(plx_fuse_cols *c, const char *name, const char *suffix, double *train, double *test) {
    PLX_ASSERT(c->n_cols < PLX_FUSE_MAX_COLS);
    snprintf(c->name[c->n_cols], sizeof c->name[0], "%s%s", name, suffix);
    c->train[c->n_cols] = train;
    c->test[c->n_cols] = test;
    c->n_cols++;
}

int plx_fuse_columns(const plx_score_source *src, int n_src, const plx_rec_split *s, plx_fuse_cols *out) {
    memset(out, 0, sizeof *out);
    int ntr = PLX_MAX(s->n_train, 1), nte = PLX_MAX(s->n_test, 1);
    for (int k = 0; k < n_src; k++) {
        const plx_score_source *q = &src[k];
        if (q->fn) {
            double *tr = PLX_NEW(double, ntr), *te = PLX_NEW(double, nte);
            for (int i = 0; i < s->n_train; i++) tr[i] = q->fn(s->train[i].a, s->train[i].b, q->ctx);
            for (int i = 0; i < s->n_test; i++) te[i] = q->fn(s->test[i].a, s->test[i].b, q->ctx);
            add_col(out, q->name, "", tr, te);
            continue;
        }
        PLX_ASSERT(!q->sup || q->cfg->sup != PLX_SUP_NONE);
        plx_rec_scores sc = {.test_unsup = PLX_NEW(double, nte), .train_unsup = PLX_NEW(double, ntr),
                             .test_sup = q->sup ? PLX_NEW(double, nte) : NULL,
                             .train_sup = q->sup ? PLX_NEW(double, ntr) : NULL};
        int rc = plx_rec_run_split(q->cfg, q->lay, q->feats, s, &sc);
        if (rc != PLX_OK) {
            free(sc.test_unsup);
            free(sc.train_unsup);
            free(sc.test_sup);
            free(sc.train_sup);
            plx_fuse_cols_free(out);
            return rc;
        }
        if (q->unsup) add_col(out, q->name, "/u", sc.train_unsup, sc.test_unsup);
        else {
            free(sc.train_unsup);
            free(sc.test_unsup);
        }
        if (q->sup) add_col(out, q->name, "/s", sc.train_sup, sc.test_sup);
    }
    return PLX_OK;
}

void plx_fuse_cols_free(plx_fuse_cols *c) {
    for (int k = 0; k < c->n_cols; k++) {
        free(c->train[k]);
        free(c->test[k]);
    }
    memset(c, 0, sizeof *c);
}

void plx_fuse_standardise(const plx_fuse_cols *c, int n_train, double *m, double *sd) {
    for (int k = 0; k < c->n_cols; k++) {
        double s = 0.0, ss = 0.0;
        int cnt = 0;
        for (int i = 0; i < n_train; i++) {
            double v = c->train[k][i];
            if (!isfinite(v)) continue;
            s += v;
            ss += v * v;
            cnt++;
        }
        m[k] = cnt ? s / cnt : 0.0;
        double var = cnt > 1 ? (ss - cnt * m[k] * m[k]) / (cnt - 1) : 0.0;
        sd[k] = var > 0.0 ? sqrt(var) : 1.0;
    }
}

static double zval(double v, double m, double sd) { return isfinite(v) ? (v - m) / sd : 0.0; }

int plx_fuse_learned(const plx_fuse_cols *c, const plx_pair *train, int n_train, int n_test, const int *use,
                     int n_use, double l2, double *train_score, double *test_score, double *w_out) {
    double m[PLX_FUSE_MAX_COLS], sd[PLX_FUSE_MAX_COLS];
    plx_fuse_standardise(c, n_train, m, sd);
    int F = n_use + 1;
    plx_mat Phi = plx_mat_new(n_train, F);
    int *y = PLX_NEW(int, n_train);
    for (int i = 0; i < n_train; i++) {
        for (int j = 0; j < n_use; j++) PLX_M(&Phi, i, j) = zval(c->train[use[j]][i], m[use[j]], sd[use[j]]);
        PLX_M(&Phi, i, n_use) = 1.0;
        y[i] = train[i].same;
    }
    double w[PLX_FUSE_MAX_COLS + 1];
    int rc = plx_logreg_fit(&Phi, y, l2, w);
    for (int i = 0; i < n_train; i++) train_score[i] = plx_vec_dot(&PLX_M(&Phi, i, 0), w, (size_t)F);
    for (int i = 0; i < n_test; i++) {
        double s = w[n_use];
        for (int j = 0; j < n_use; j++) s += w[j] * zval(c->test[use[j]][i], m[use[j]], sd[use[j]]);
        test_score[i] = s;
    }
    if (w_out) memcpy(w_out, w, sizeof(double) * F);
    plx_mat_free(&Phi);
    free(y);
    return rc;
}

void plx_fuse_fixed(const plx_fuse_cols *c, int n_train, int n_test, const int *use, int n_use, const double *w,
                    double *train_score, double *test_score) {
    double m[PLX_FUSE_MAX_COLS], sd[PLX_FUSE_MAX_COLS];
    plx_fuse_standardise(c, n_train, m, sd);
    for (int i = 0; i < n_train; i++) {
        double s = 0.0;
        for (int j = 0; j < n_use; j++) s += w[j] * zval(c->train[use[j]][i], m[use[j]], sd[use[j]]);
        train_score[i] = s;
    }
    for (int i = 0; i < n_test; i++) {
        double s = 0.0;
        for (int j = 0; j < n_use; j++) s += w[j] * zval(c->test[use[j]][i], m[use[j]], sd[use[j]]);
        test_score[i] = s;
    }
}
