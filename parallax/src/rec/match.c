/* Similarity models: raw histogram distances, Hellinger + PCA-whitening + cosine, and
 * pair-supervised metrics on top of that embedding. */
#include "rec/match.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Rows are embedded in chunks so the imputed-descriptor scratch stays small whatever n is. */
enum { EMBED_CHUNK = 512 };

plx_match_cfg plx_match_default(void) {
    return (plx_match_cfg){.score = PLX_SCORE_EMBED, .pca_dim = 250, .per_block = true, .whiten = true,
                           .pca_max_train = 6000, .pca_iters = 2, .sup = PLX_SUP_NONE, .sup_pca = 100,
                           .sup_dim = 50, .sup_ridge = 0.1, .logit_l2 = 0.01, .seed = 0x5eedf00d};
}

void plx_match_describe(const plx_match_cfg *c, bool supervised, char *buf, size_t cap) {
    static const char *const score_names[] = {"embed", "chi2", "cos", "cell"};
    static const char *const sup_names[] = {"none", "lda", "jb", "logit"};
    if (c->score != PLX_SCORE_EMBED) {
        snprintf(buf, cap, "score=%s", score_names[c->score]);
        return;
    }
    int at = snprintf(buf, cap, "score=embed pca=%d per_block=%d whiten=%d pca_n=%d pca_iters=%d", c->pca_dim,
                      (int)c->per_block, (int)c->whiten, c->pca_max_train, c->pca_iters);
    if (supervised && c->sup != PLX_SUP_NONE && at > 0 && (size_t)at < cap) {
        int more = snprintf(buf + at, cap - (size_t)at, " | sup=%s sup_pca=%d sup_dim=%d ridge=%g", sup_names[c->sup],
                            c->sup_pca, c->sup_dim, c->sup_ridge);
        if (c->sup == PLX_SUP_LOGIT && more > 0 && (size_t)(at + more) < cap)
            snprintf(buf + at + more, cap - (size_t)(at + more), " l2=%g", c->logit_l2);
    }
}

/* ---- helpers --------------------------------------------------------------------------- */

/* x~ = mean + w (x - mean) per cell: a cell the face cannot vouch for (w -> 0) is replaced by
 * what an average face has there, so it pulls the embedding towards nothing in particular
 * instead of towards "this cell is empty". With uniform confidence x~ = x exactly. */
static void impute_rows(const plx_match_model *m, const float *x, const float *cw, int n, float *out) {
    const plx_desc_layout *L = &m->lay;
    memcpy(out, x, sizeof(float) * (size_t)n * L->d);
    if (!cw) return;
    for (int r = 0; r < n; r++)
        for (int i = 0; i < L->n_blocks; i++)
            for (int c = 0; c < L->cells_x[i] * L->cells_y[i]; c++) {
                float wc = PLX_CLAMP(cw[(size_t)r * L->n_cells + L->cell_off[i] + c], 0.0f, 1.0f);
                if (wc >= 1.0f) continue;
                int j0 = L->off[i] + c * L->bins[i];
                float *o = out + (size_t)r * L->d;
                for (int j = j0; j < j0 + L->bins[i]; j++) o[j] = (float)(m->mean[j] + wc * (o[j] - m->mean[j]));
            }
}

static void normalise(float *v, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)v[i] * v[i];
    if (s <= 0.0) return;
    float k = (float)(1.0 / sqrt(s));
    for (int i = 0; i < n; i++) v[i] *= k;
}

static double cosine(const float *a, const float *b, int n) {
    double ab = 0.0, aa = 0.0, bb = 0.0;
    for (int i = 0; i < n; i++) {
        ab += (double)a[i] * b[i];
        aa += (double)a[i] * a[i];
        bb += (double)b[i] * b[i];
    }
    return aa > 0.0 && bb > 0.0 ? ab / sqrt(aa * bb) : 0.0;
}

static int group_dim(const plx_match_model *m, int g) { return m->cfg.pca_dim == 0 ? m->g_len[g] : m->pca[g].k; }

/* unsupervised embedding of n rows (n <= EMBED_CHUNK) */
static void embed_unsup(const plx_match_model *m, const float *x, const float *cw, int n, float *y) {
    int d = m->lay.d, k = m->k;
    float *imp = plx_malloc(sizeof(float) * (size_t)n * d);
    impute_rows(m, x, cw, n, imp);
    if (m->cfg.pca_dim == 0) {
        memcpy(y, imp, sizeof(float) * (size_t)n * d);
    } else {
        float *slice = plx_malloc(sizeof(float) * (size_t)n * d);
        float *proj = plx_malloc(sizeof(float) * (size_t)n * k);
        int at = 0;
        for (int g = 0; g < m->n_groups; g++) {
            int dg = m->g_len[g], kg = m->pca[g].k;
            for (int r = 0; r < n; r++)
                memcpy(slice + (size_t)r * dg, imp + (size_t)r * d + m->g_off[g], sizeof(float) * dg);
            plx_pca_project_f32(&m->pca[g], slice, n, m->cfg.whiten, proj);
            for (int r = 0; r < n; r++) memcpy(y + (size_t)r * k + at, proj + (size_t)r * kg, sizeof(float) * kg);
            at += kg;
        }
        free(slice);
        free(proj);
    }
    /* per-block unit norm first, so each block gets one vote in the final cosine whatever its size */
    for (int r = 0; r < n; r++) {
        float *row = y + (size_t)r * k;
        if (m->cfg.per_block)
            for (int g = 0, at = 0; g < m->n_groups; at += group_dim(m, g), g++) normalise(row + at, group_dim(m, g));
        normalise(row, k);
    }
    free(imp);
}

static void project_sup(const plx_match_model *m, const float *u, float *z) {
    for (int i = 0; i < m->m; i++) {
        const double *p = &PLX_M(&m->P, i, 0);
        double s = 0.0;
        for (int j = 0; j < m->k; j++) s += p[j] * (u[j] - m->mu[j]);
        z[i] = (float)s;
    }
}

int plx_match_embed_dim(const plx_match_model *m, bool supervised) { return supervised ? m->m : m->k; }

void plx_match_embed(const plx_match_model *m, const float *x, const float *cw, int n, bool supervised,
                     float *y) {
    PLX_ASSERT(m->cfg.score == PLX_SCORE_EMBED && (!supervised || m->has_sup));
    int d = m->lay.d, nc = m->lay.n_cells, k = m->k, out = plx_match_embed_dim(m, supervised);
    float *u = supervised ? plx_malloc(sizeof(float) * (size_t)EMBED_CHUNK * k) : NULL;
    for (int r0 = 0; r0 < n; r0 += EMBED_CHUNK) {
        int nr = PLX_MIN(EMBED_CHUNK, n - r0);
        const float *xr = x + (size_t)r0 * d, *cr = cw ? cw + (size_t)r0 * nc : NULL;
        if (!u) {
            embed_unsup(m, xr, cr, nr, y + (size_t)r0 * out);
            continue;
        }
        embed_unsup(m, xr, cr, nr, u);
        for (int r = 0; r < nr; r++) project_sup(m, u + (size_t)r * k, y + (size_t)(r0 + r) * out);
    }
    free(u);
}

/* Label-free reduction ahead of the supervised stage: the top sup_pca principal directions of
 * the embedded PCA subsample, whitened. It is fitted here, on the large unlabelled set, rather
 * than on the pair images: whitening estimated from a few thousand images inflates the norm of
 * any image outside that set, and pair-score models such as JB are not scale-invariant, so
 * fitting it on the pair images would score training pairs and test pairs on different scales. */
static int fit_reduction(plx_match_model *m, const float *X, int n) {
    int k = m->k;
    float *Y = plx_malloc(sizeof(float) * (size_t)n * k);
    plx_match_embed(m, X, NULL, n, false, Y);
    m->mu = PLX_NEW(double, k);
    plx_mat St = plx_mat_new(k, k);
    plx_cov_f32(Y, n, k, m->mu, &St);
    free(Y);
    double *ev = PLX_NEW(double, k);
    plx_mat V = plx_mat_new(k, k);
    int rc = plx_eigh(&St, ev, &V);
    plx_mat_free(&St);
    if (rc != PLX_OK) plx_log("embedding covariance eigenproblem failed: %s", plx_la_status_str(rc));
    int r = 0;
    while (r < PLX_MIN(m->cfg.sup_pca, PLX_MIN(k, n - 1)) && ev[r] > 1e-9 * ev[0]) r++;
    m->T0 = plx_mat_new(r, k);
    for (int i = 0; i < r; i++)
        for (int j = 0; j < k; j++) PLX_M(&m->T0, i, j) = PLX_M(&V, j, i) / sqrt(ev[i]);
    plx_mat_free(&V);
    free(ev);
    return rc;
}

/* ---- unsupervised fit ------------------------------------------------------------------ */

int plx_match_fit(const plx_match_cfg *cfg, const plx_desc_layout *lay, const plx_feats *f, plx_match_model *m) {
    memset(m, 0, sizeof *m);
    m->cfg = *cfg;
    m->lay = *lay;
    PLX_ASSERT(f->d == lay->d && f->n_cells == lay->n_cells);
    if (cfg->score != PLX_SCORE_EMBED) return PLX_OK;
    int d = lay->d;
    m->mean = PLX_NEW(double, d);
    plx_col_mean_f32(f->x, f->n, d, m->mean);
    m->n_groups = cfg->per_block ? lay->n_blocks : 1;
    for (int g = 0; g < m->n_groups; g++) {
        m->g_off[g] = cfg->per_block ? lay->off[g] : 0;
        m->g_len[g] = cfg->per_block ? lay->dim[g] : d;
    }
    if (cfg->pca_dim == 0) {
        m->k = d;
        return PLX_OK;
    }
    /* deterministic subsample, kept in index order for memory locality */
    int nsub = PLX_MIN(f->n, cfg->pca_max_train > 0 ? cfg->pca_max_train : f->n);
    PLX_ASSERT(nsub >= 2);
    int *idx = PLX_NEW(int, f->n);
    for (int i = 0; i < f->n; i++) idx[i] = i;
    plx_rng rng;
    plx_rng_seed(&rng, cfg->seed);
    plx_rng_shuffle(&rng, idx, f->n);
    uint8_t *pick = PLX_NEW(uint8_t, f->n);
    for (int i = 0; i < nsub; i++) pick[idx[i]] = 1;
    float *X = plx_malloc(sizeof(float) * (size_t)nsub * d);
    for (int i = 0, r = 0; i < f->n; i++) {
        if (!pick[i]) continue;
        impute_rows(m, f->x + (size_t)i * d, f->cw ? f->cw + (size_t)i * f->n_cells : NULL, 1, X + (size_t)r * d);
        r++;
    }
    free(idx);
    free(pick);

    int rc = PLX_OK;
    float *slice = m->n_groups > 1 ? plx_malloc(sizeof(float) * (size_t)nsub * d) : NULL;
    plx_pca_opts po = {.method = PLX_PCA_AUTO, .power_iters = cfg->pca_iters, .seed = cfg->seed};
    for (int g = 0; g < m->n_groups && rc == PLX_OK; g++) {
        int dg = m->g_len[g];
        const float *Xg = X;
        if (slice) {
            for (int r = 0; r < nsub; r++)
                memcpy(slice + (size_t)r * dg, X + (size_t)r * d + m->g_off[g], sizeof(float) * dg);
            Xg = slice;
        }
        int kg = PLX_MIN(cfg->pca_dim, PLX_MIN(nsub - 1, dg));
        rc = plx_pca_f32(Xg, nsub, dg, kg, &po, &m->pca[g]);
        if (rc != PLX_OK) plx_log("PCA of group %d failed: %s", g, plx_la_status_str(rc));
        m->k += kg;
    }
    free(slice);
    if (rc == PLX_OK && cfg->sup != PLX_SUP_NONE) rc = fit_reduction(m, X, nsub);
    free(X);
    return rc;
}

/* ---- supervised fit -------------------------------------------------------------------- */

static void clear_sup(plx_match_model *m) {
    plx_mat_free(&m->P);
    free(m->lam);
    free(m->w);
    free(m->f_mean);
    free(m->f_scale);
    m->lam = m->w = m->f_mean = m->f_scale = NULL;
    m->has_sup = false;
    m->m = 0;
}

/* Pair-LDA basis, in the whitened reduced coordinates T0 (y - mu) where the total covariance is
 * I: the within-pair covariance Sw comes from the same-labelled pairs (Var(y_a - y_b | same) =
 * 2 Sw), and the generalised eigenvectors of (I, Sw) give z = P (y - mu) with unit within-pair
 * variance and between-pair variance lam: the simultaneous diagonalisation the joint Bayesian
 * model needs. Fewer same pairs than dimensions makes Sw singular; the ridge keeps directions
 * where a few pairs agree by chance from looking infinitely reliable. */
static int fit_pair_lda(plx_match_model *m, const float *y, int np, const int *pa, const int *pb, const int *same) {
    int k = m->k, r = m->T0.rows, rc = PLX_OK;
    const plx_mat T0 = m->T0;
    int ns = 0;
    for (int p = 0; p < np; p++) ns += same[p] != 0;
    PLX_ASSERT(ns >= 2 && ns < np);
    float *D = plx_malloc(sizeof(float) * (size_t)ns * r);
    double *diff = PLX_NEW(double, k);
    for (int p = 0, q = 0; p < np; p++) {
        if (!same[p]) continue;
        const float *ya = y + (size_t)pa[p] * k, *yb = y + (size_t)pb[p] * k;
        for (int j = 0; j < k; j++) diff[j] = (double)ya[j] - yb[j];
        for (int i = 0; i < r; i++) D[(size_t)q * r + i] = (float)plx_vec_dot(&PLX_M(&T0, i, 0), diff, (size_t)k);
        q++;
    }
    free(diff);
    plx_mat Sw = plx_mat_new(r, r), I = plx_mat_identity(r);
    plx_scatter_f32(D, ns, r, NULL, &Sw);
    free(D);
    plx_mat_scale(&Sw, 1.0 / (2.0 * ns));
    double tr = 0.0;
    for (int j = 0; j < r; j++) tr += PLX_M(&Sw, j, j);
    for (int j = 0; j < r; j++) PLX_M(&Sw, j, j) += m->cfg.sup_ridge * tr / r;

    double *ev = PLX_NEW(double, r);
    plx_mat V = plx_mat_new(r, r);
    if (rc == PLX_OK) rc = plx_eigh_gen(&I, &Sw, ev, &V);
    if (rc != PLX_OK) plx_log("pair-LDA eigenproblem failed: %s", plx_la_status_str(rc));
    m->m = PLX_MIN(m->cfg.sup_dim, r);
    m->P = plx_mat_new(m->m, k);
    m->lam = PLX_NEW(double, m->m);
    for (int i = 0; i < m->m; i++) {
        for (int j = 0; j < k; j++) {
            double s = 0.0;
            for (int t = 0; t < r; t++) s += PLX_M(&V, t, i) * PLX_M(&T0, t, j);
            PLX_M(&m->P, i, j) = s;
        }
        m->lam[i] = ev[i] - 1.0;
    }
    free(ev);
    plx_mat_free(&V);
    plx_mat_free(&I);
    plx_mat_free(&Sw);
    return rc;
}

/* JB log-likelihood ratio in the pair-LDA basis, per coordinate with between variance lam and
 * within variance 1: log N([u v]; 0, [[lam+1, lam], [lam, lam+1]]) - log N(u; 0, lam+1)
 * - log N(v; 0, lam+1) = a (u^2 + v^2) + b u v + const, with
 *   a = -lam^2 / (2 (lam+1)(2 lam+1)),  b = lam / (2 lam + 1). */
static void fit_jb(plx_match_model *m) {
    m->w = PLX_NEW(double, 2 * m->m);
    for (int i = 0; i < m->m; i++) {
        double l = fmax(m->lam[i], 0.0); /* negative: noise exceeds signal, the dimension says nothing */
        m->w[2 * i] = -l * l / (2.0 * (l + 1.0) * (2.0 * l + 1.0));
        m->w[2 * i + 1] = l / (2.0 * l + 1.0);
    }
}

static void pair_features(const float *u, const float *v, int mm, double *phi) {
    for (int i = 0; i < mm; i++) {
        double d = (double)u[i] - v[i];
        phi[2 * i] = (double)u[i] * v[i];
        phi[2 * i + 1] = d * d;
    }
}

/* L2-regularised logistic regression by Newton's method (IRLS) on standardised pair features.
 * Objective: mean log-loss + 0.5 l2 ||w||^2 (bias unpenalised). */
static int fit_logit(plx_match_model *m, const float *y, int np, const int *pa, const int *pb, const int *same) {
    int mm = m->m, F = 2 * mm + 1, k = m->k;
    plx_mat Phi = plx_mat_new(np, F);
    float *za = PLX_NEW(float, mm), *zb = PLX_NEW(float, mm);
    for (int p = 0; p < np; p++) {
        project_sup(m, y + (size_t)pa[p] * k, za);
        project_sup(m, y + (size_t)pb[p] * k, zb);
        pair_features(za, zb, mm, &PLX_M(&Phi, p, 0));
        PLX_M(&Phi, p, F - 1) = 1.0;
    }
    free(za);
    free(zb);
    m->f_mean = PLX_NEW(double, F - 1);
    m->f_scale = PLX_NEW(double, F - 1);
    for (int j = 0; j < F - 1; j++) {
        double s = 0.0, ss = 0.0;
        for (int p = 0; p < np; p++) s += PLX_M(&Phi, p, j);
        s /= np;
        for (int p = 0; p < np; p++) ss += (PLX_M(&Phi, p, j) - s) * (PLX_M(&Phi, p, j) - s);
        double sd = sqrt(ss / np);
        m->f_mean[j] = s;
        m->f_scale[j] = sd > 0.0 ? 1.0 / sd : 0.0;
        for (int p = 0; p < np; p++) PLX_M(&Phi, p, j) = (PLX_M(&Phi, p, j) - s) * m->f_scale[j];
    }
    m->w = PLX_NEW(double, F);
    double *g = PLX_NEW(double, F);
    plx_mat H = plx_mat_new(F, F), Pw = plx_mat_new(np, F);
    int rc = PLX_OK;
    for (int it = 0; it < 50; it++) {
        memset(g, 0, sizeof(double) * F);
        for (int p = 0; p < np; p++) {
            double t = plx_vec_dot(&PLX_M(&Phi, p, 0), m->w, (size_t)F);
            double pr = 1.0 / (1.0 + exp(-t));
            double res = (pr - (same[p] != 0)) / np, sw = sqrt(pr * (1.0 - pr) / np);
            plx_vec_axpy(g, res, &PLX_M(&Phi, p, 0), (size_t)F);
            for (int j = 0; j < F; j++) PLX_M(&Pw, p, j) = sw * PLX_M(&Phi, p, j);
        }
        plx_syrk(PLX_TRANS, 1.0, &Pw, 0.0, &H);
        for (int j = 0; j < F; j++) {
            double l2 = j < F - 1 ? m->cfg.logit_l2 : 1e-9; /* the bias is (almost) unpenalised */
            g[j] += l2 * m->w[j];
            PLX_M(&H, j, j) += l2;
        }
        rc = plx_chol(&H);
        if (rc != PLX_OK) {
            plx_log("logistic regression: Hessian not SPD");
            break;
        }
        plx_chol_solve_vec(&H, g);
        plx_vec_axpy(m->w, -1.0, g, (size_t)F);
        if (plx_vec_norm(g, (size_t)F) < 1e-9 * (1.0 + plx_vec_norm(m->w, (size_t)F))) break;
    }
    free(g);
    plx_mat_free(&H);
    plx_mat_free(&Pw);
    plx_mat_free(&Phi);
    return rc;
}

int plx_match_fit_pairs(plx_match_model *m, const float *y, int n, int n_pairs, const int *pa, const int *pb,
                        const int *same) {
    PLX_ASSERT(m->cfg.score == PLX_SCORE_EMBED && m->cfg.sup != PLX_SUP_NONE);
    for (int p = 0; p < n_pairs; p++) PLX_ASSERT(pa[p] >= 0 && pa[p] < n && pb[p] >= 0 && pb[p] < n);
    clear_sup(m);
    int rc = fit_pair_lda(m, y, n_pairs, pa, pb, same);
    if (rc == PLX_OK && m->cfg.sup == PLX_SUP_JB) fit_jb(m);
    if (rc == PLX_OK && m->cfg.sup == PLX_SUP_LOGIT) rc = fit_logit(m, y, n_pairs, pa, pb, same);
    m->has_sup = rc == PLX_OK;
    return rc;
}

void plx_match_free(plx_match_model *m) {
    clear_sup(m);
    free(m->mean);
    free(m->mu);
    plx_mat_free(&m->T0);
    for (int g = 0; g < m->n_groups; g++)
        if (m->pca[g].mean) plx_pca_free(&m->pca[g]);
    memset(m, 0, sizeof *m);
}

/* ---- scoring --------------------------------------------------------------------------- */

static double score_raw(const plx_match_model *m, const plx_feats *f, int a, int b) {
    const float *x = f->x + (size_t)a * f->d, *y = f->x + (size_t)b * f->d;
    switch (m->cfg.score) {
    case PLX_SCORE_CHI2: {
        double s = 0.0;
        for (int j = 0; j < f->d; j++) {
            double t = (double)x[j] + y[j];
            if (t > 0.0) s += ((double)x[j] - y[j]) * ((double)x[j] - y[j]) / t;
        }
        return -s;
    }
    case PLX_SCORE_COS: return cosine(x, y, f->d);
    case PLX_SCORE_CELL: {
        const plx_desc_layout *L = &m->lay;
        double num = 0.0, den = 0.0;
        for (int i = 0; i < L->n_blocks; i++)
            for (int c = 0; c < L->cells_x[i] * L->cells_y[i]; c++) {
                int cell = L->cell_off[i] + c;
                double w = f->cw ? sqrt(fmax(f->cw[(size_t)a * f->n_cells + cell], 0.0f) *
                                        fmax(f->cw[(size_t)b * f->n_cells + cell], 0.0f))
                                 : 1.0;
                if (w <= 0.0) continue;
                const float *xc = x + L->off[i] + c * L->bins[i], *yc = y + L->off[i] + c * L->bins[i];
                double dot = 0.0;
                for (int k = 0; k < L->bins[i]; k++) dot += (double)xc[k] * yc[k];
                num += w * dot;
                den += w;
            }
        return den > 0.0 ? num / den : 0.0;
    }
    default: plx_die("score_raw: not a raw score");
    }
}

double plx_match_score_embedded(const plx_match_model *m, const float *u, const float *v, bool supervised) {
    if (!supervised) return cosine(u, v, m->k);
    int mm = m->m;
    switch (m->cfg.sup) {
    case PLX_SUP_LDA: return cosine(u, v, mm);
    case PLX_SUP_JB: {
        double s = 0.0;
        for (int i = 0; i < mm; i++)
            s += m->w[2 * i] * ((double)u[i] * u[i] + (double)v[i] * v[i]) + m->w[2 * i + 1] * (double)u[i] * v[i];
        return s;
    }
    case PLX_SUP_LOGIT: {
        double *phi = plx_malloc(sizeof(double) * 2 * mm), s = m->w[2 * mm];
        pair_features(u, v, mm, phi);
        for (int j = 0; j < 2 * mm; j++) s += m->w[j] * (phi[j] - m->f_mean[j]) * m->f_scale[j];
        free(phi);
        return s;
    }
    case PLX_SUP_NONE: break;
    }
    plx_die("supervised score without a supervised method");
}

void plx_match_score(const plx_match_model *m, const plx_feats *f, const int *a, const int *b, int n,
                     bool supervised, double *score) {
    if (m->cfg.score != PLX_SCORE_EMBED) {
        PLX_ASSERT(!supervised);
        for (int i = 0; i < n; i++) score[i] = score_raw(m, f, a[i], b[i]);
        return;
    }
    /* embed each distinct image once */
    int *row = PLX_NEW(int, f->n);
    int nu = 0;
    for (int i = 0; i < f->n; i++) row[i] = -1;
    for (int i = 0; i < n; i++) {
        if (row[a[i]] < 0) row[a[i]] = nu++;
        if (row[b[i]] < 0) row[b[i]] = nu++;
    }
    float *X = plx_malloc(sizeof(float) * (size_t)nu * f->d);
    float *C = f->cw ? plx_malloc(sizeof(float) * (size_t)nu * f->n_cells) : NULL;
    for (int i = 0; i < f->n; i++) {
        if (row[i] < 0) continue;
        memcpy(X + (size_t)row[i] * f->d, f->x + (size_t)i * f->d, sizeof(float) * f->d);
        if (C) memcpy(C + (size_t)row[i] * f->n_cells, f->cw + (size_t)i * f->n_cells, sizeof(float) * f->n_cells);
    }
    int e = plx_match_embed_dim(m, supervised);
    float *Y = plx_malloc(sizeof(float) * (size_t)nu * e);
    plx_match_embed(m, X, C, nu, supervised, Y);
    for (int i = 0; i < n; i++)
        score[i] = plx_match_score_embedded(m, Y + (size_t)row[a[i]] * e, Y + (size_t)row[b[i]] * e, supervised);
    free(X);
    free(C);
    free(Y);
    free(row);
}
