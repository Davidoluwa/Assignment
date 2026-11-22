/* LFW protocol splits and evaluation loops. */
#include "rec/proto.h"

#include <stdlib.h>
#include <string.h>

static plx_pair *copy_pairs(const plx_pair *p, int n) {
    plx_pair *out = PLX_NEW(plx_pair, PLX_MAX(n, 1));
    memcpy(out, p, sizeof(plx_pair) * (size_t)n);
    return out;
}

void plx_rec_split_view2(const plx_pairs *pl, const plx_lfw *db, int fold, plx_rec_split *s) {
    PLX_ASSERT(fold >= 0 && fold < pl->n_folds && pl->n_folds >= 2);
    memset(s, 0, sizeof *s);
    s->n_images = db->n_images;
    s->img_id = db->img_id;
    s->fit_img = PLX_NEW(uint8_t, db->n_images);
    plx_pairs_fold_identity_mask(pl, db, fold, s->fit_img);
    for (int i = 0; i < db->n_images; i++) s->fit_img[i] = !s->fit_img[i];
    s->train = PLX_NEW(plx_pair, pl->n);
    s->test = PLX_NEW(plx_pair, pl->n);
    for (int i = 0; i < pl->n; i++) {
        if (pl->p[i].fold == fold) s->test[s->n_test++] = pl->p[i];
        else s->train[s->n_train++] = pl->p[i];
    }
    int bad = plx_rec_split_check(s, db);
    if (bad) plx_die("View 2 fold %d: %d protocol violations in the split", fold, bad);
}

void plx_rec_split_view1(const plx_pairs *dev_train, const plx_pairs *dev_test, const plx_lfw *db,
                         plx_rec_split *s) {
    memset(s, 0, sizeof *s);
    s->n_images = db->n_images;
    s->img_id = db->img_id;
    s->fit_img = PLX_NEW(uint8_t, db->n_images);
    /* the dev lists are single-fold, so "fold 0" is the whole test list */
    plx_pairs_fold_identity_mask(dev_test, db, 0, s->fit_img);
    for (int i = 0; i < db->n_images; i++) s->fit_img[i] = !s->fit_img[i];
    s->train = copy_pairs(dev_train->p, dev_train->n);
    s->n_train = dev_train->n;
    s->test = copy_pairs(dev_test->p, dev_test->n);
    s->n_test = dev_test->n;
    int bad = plx_rec_split_check(s, db);
    if (bad) plx_die("View 1: %d protocol violations in the split", bad);
}

void plx_rec_split_free(plx_rec_split *s) {
    free(s->fit_img);
    free(s->train);
    free(s->test);
    memset(s, 0, sizeof *s);
}

int plx_rec_split_check(const plx_rec_split *s, const plx_lfw *db) {
    PLX_ASSERT(s->n_images == db->n_images);
    int bad = 0;
    uint8_t *test_id = PLX_NEW(uint8_t, db->n_ids);
    for (int i = 0; i < s->n_test; i++) {
        test_id[db->img_id[s->test[i].a]] = 1;
        test_id[db->img_id[s->test[i].b]] = 1;
    }
    for (int i = 0; i < db->n_images; i++) bad += s->fit_img[i] && test_id[db->img_id[i]];
    for (int i = 0; i < s->n_train; i++) {
        const plx_pair *p = &s->train[i];
        bad += !s->fit_img[p->a] + !s->fit_img[p->b];
        for (int j = 0; j < s->n_test; j++) {
            const plx_pair *q = &s->test[j];
            bad += (p->a == q->a && p->b == q->b) || (p->a == q->b && p->b == q->a);
        }
    }
    free(test_id);
    return bad;
}

static void pairs_ab(const plx_pair *p, int n, int **a, int **b) {
    *a = PLX_NEW(int, PLX_MAX(n, 1));
    *b = PLX_NEW(int, PLX_MAX(n, 1));
    for (int i = 0; i < n; i++) {
        (*a)[i] = p[i].a;
        (*b)[i] = p[i].b;
    }
}

static void score_pairs(const plx_match_model *m, const plx_feats *all, const plx_pair *p, int n, bool sup,
                        double *score) {
    int *a, *b;
    pairs_ab(p, n, &a, &b);
    plx_match_score(m, all, a, b, n, sup, score);
    free(a);
    free(b);
}

static int uf_find(int *parent, int x) {
    while (parent[x] != x) x = parent[x] = parent[parent[x]];
    return x;
}

/* Out-of-fold groups for the training pairs: identities linked by any pair are one component,
 * components are dealt largest first to the currently smallest group, so no identity has pairs
 * in two groups and the groups come out nearly equal in size. */
static void oof_groups(const plx_rec_split *s, int *group) {
    int nid = 0;
    for (int i = 0; i < s->n_images; i++) nid = PLX_MAX(nid, s->img_id[i] + 1);
    int *parent = PLX_NEW(int, nid), *size = PLX_NEW(int, nid), *gof = PLX_NEW(int, nid);
    for (int i = 0; i < nid; i++) parent[i] = i;
    for (int p = 0; p < s->n_train; p++) {
        int x = uf_find(parent, s->img_id[s->train[p].a]), y = uf_find(parent, s->img_id[s->train[p].b]);
        if (x != y) parent[x] = y;
    }
    for (int p = 0; p < s->n_train; p++) size[uf_find(parent, s->img_id[s->train[p].a])]++;
    int *order = PLX_NEW(int, nid), nc = 0;
    for (int i = 0; i < nid; i++)
        if (size[i] > 0) order[nc++] = i;
    for (int i = 1; i < nc; i++) /* insertion sort by size, descending (stable, deterministic) */
        for (int j = i; j > 0 && size[order[j]] > size[order[j - 1]]; j--) {
            int t = order[j];
            order[j] = order[j - 1];
            order[j - 1] = t;
        }
    int load[PLX_REC_OOF_FOLDS] = {0};
    for (int i = 0; i < nc; i++) {
        int g = 0;
        for (int h = 1; h < PLX_REC_OOF_FOLDS; h++)
            if (load[h] < load[g]) g = h;
        gof[order[i]] = g;
        load[g] += size[order[i]];
    }
    for (int p = 0; p < s->n_train; p++) group[p] = gof[uf_find(parent, s->img_id[s->train[p].a])];
    free(parent);
    free(size);
    free(gof);
    free(order);
}

/* Supervised stage: embed the training-pair images with the (label-free) unsupervised stage,
 * then fit on the pairs. With oof != NULL the training pairs get out-of-fold scores, and the
 * test pairs the mean score of the same out-of-fold models: a pair-score model's scale depends
 * on how many pairs it was fitted on (JB's log-likelihood ratios grow with the between/within
 * ratios it can resolve), so a threshold fitted on out-of-fold scores is only valid for models
 * of that same size. */
static int run_supervised(plx_match_model *m, const plx_feats *all, const plx_rec_split *s, double *test_sup,
                          double *oof) {
    int *row = PLX_NEW(int, s->n_images), nu = 0;
    for (int i = 0; i < s->n_images; i++) row[i] = -1;
    for (int p = 0; p < s->n_train; p++) {
        int ab[2] = {s->train[p].a, s->train[p].b};
        for (int t = 0; t < 2; t++) {
            if (!s->fit_img[ab[t]]) plx_die("training pair %d uses an image outside the fit set", p);
            if (row[ab[t]] < 0) row[ab[t]] = nu++;
        }
    }
    int d = all->d, nc = all->n_cells;
    float *X = plx_malloc(sizeof(float) * (size_t)nu * d);
    float *C = all->cw ? plx_malloc(sizeof(float) * (size_t)nu * nc) : NULL;
    for (int i = 0; i < s->n_images; i++) {
        if (row[i] < 0) continue;
        memcpy(X + (size_t)row[i] * d, all->x + (size_t)i * d, sizeof(float) * d);
        if (C) memcpy(C + (size_t)row[i] * nc, all->cw + (size_t)i * nc, sizeof(float) * nc);
    }
    float *Y = plx_malloc(sizeof(float) * (size_t)nu * m->k);
    plx_match_embed(m, X, C, nu, false, Y);
    free(X);
    free(C);
    int np = s->n_train;
    int *pa = PLX_NEW(int, np), *pb = PLX_NEW(int, np), *same = PLX_NEW(int, np);
    for (int p = 0; p < np; p++) {
        pa[p] = row[s->train[p].a];
        pb[p] = row[s->train[p].b];
        same[p] = s->train[p].same;
    }
    free(row);

    int rc = PLX_OK;
    if (oof) {
        int *group = PLX_NEW(int, np), *qa = PLX_NEW(int, np), *qb = PLX_NEW(int, np), *qs = PLX_NEW(int, np);
        plx_pair *held = PLX_NEW(plx_pair, np);
        int *held_idx = PLX_NEW(int, np);
        double *hs = PLX_NEW(double, np), *ts = PLX_NEW(double, PLX_MAX(s->n_test, 1));
        for (int t = 0; t < s->n_test; t++) test_sup[t] = 0.0;
        oof_groups(s, group);
        for (int g = 0; g < PLX_REC_OOF_FOLDS && rc == PLX_OK; g++) {
            int nq = 0, nh = 0;
            for (int p = 0; p < np; p++) {
                if (group[p] == g) {
                    held[nh] = s->train[p];
                    held_idx[nh++] = p;
                } else {
                    qa[nq] = pa[p];
                    qb[nq] = pb[p];
                    qs[nq++] = same[p];
                }
            }
            PLX_ASSERT(nh > 0);
            rc = plx_match_fit_pairs(m, Y, nu, nq, qa, qb, qs);
            if (rc != PLX_OK) break;
            score_pairs(m, all, held, nh, true, hs);
            for (int h = 0; h < nh; h++) oof[held_idx[h]] = hs[h];
            score_pairs(m, all, s->test, s->n_test, true, ts);
            for (int t = 0; t < s->n_test; t++) test_sup[t] += ts[t] / PLX_REC_OOF_FOLDS;
        }
        free(group);
        free(qa);
        free(qb);
        free(qs);
        free(held);
        free(held_idx);
        free(hs);
        free(ts);
    } else {
        rc = plx_match_fit_pairs(m, Y, nu, np, pa, pb, same);
        if (rc == PLX_OK) score_pairs(m, all, s->test, s->n_test, true, test_sup);
    }
    free(Y);
    free(pa);
    free(pb);
    free(same);
    return rc;
}

int plx_rec_run_split(const plx_match_cfg *cfg, const plx_desc_layout *lay, const plx_feats *all,
                      const plx_rec_split *s, const plx_rec_scores *out) {
    PLX_ASSERT(all->n == s->n_images && all->d == lay->d && all->n_cells == lay->n_cells);
    PLX_ASSERT(out->test_unsup && (!out->train_sup || out->test_sup));
    PLX_ASSERT(!out->test_sup || (cfg->sup != PLX_SUP_NONE && cfg->score == PLX_SCORE_EMBED));
    int d = all->d, nc = all->n_cells;

    /* unsupervised stage: a compact copy of the fit images, and nothing else */
    int nfit = 0;
    for (int i = 0; i < s->n_images; i++) nfit += s->fit_img[i] != 0;
    float *X = plx_malloc(sizeof(float) * (size_t)nfit * d);
    float *C = all->cw ? plx_malloc(sizeof(float) * (size_t)nfit * nc) : NULL;
    for (int i = 0, r = 0; i < s->n_images; i++) {
        if (!s->fit_img[i]) continue;
        memcpy(X + (size_t)r * d, all->x + (size_t)i * d, sizeof(float) * d);
        if (C) memcpy(C + (size_t)r * nc, all->cw + (size_t)i * nc, sizeof(float) * nc);
        r++;
    }
    plx_feats train = {.n = nfit, .d = d, .n_cells = nc, .x = X, .cw = C};
    plx_match_model m;
    double t0 = plx_now();
    int rc = plx_match_fit(cfg, lay, &train, &m);
    free(X);
    free(C);
    double t1 = plx_now();
    if (rc == PLX_OK) {
        score_pairs(&m, all, s->test, s->n_test, false, out->test_unsup);
        if (out->train_unsup) score_pairs(&m, all, s->train, s->n_train, false, out->train_unsup);
        if (out->test_sup) rc = run_supervised(&m, all, s, out->test_sup, out->train_sup);
    }
    plx_log("split: unsupervised fit on %d images %.1fs, scoring%s %.1fs", nfit, t1 - t0,
            out->test_sup ? " + supervised stage" : "", plx_now() - t1);
    plx_match_free(&m);
    return rc;
}

int plx_rec_eval_view2(const plx_match_cfg *cfg, const plx_desc_layout *lay, const plx_feats *all,
                       const plx_pairs *pl, const plx_lfw *db, double *score_unsup, double *score_sup,
                       plx_verif_report *r_unsup, plx_verif_report *r_sup) {
    for (int k = 0; k < pl->n_folds; k++) {
        plx_rec_split s;
        plx_rec_split_view2(pl, db, k, &s);
        plx_rec_scores sc = {.test_unsup = PLX_NEW(double, PLX_MAX(s.n_test, 1)),
                             .test_sup = score_sup ? PLX_NEW(double, PLX_MAX(s.n_test, 1)) : NULL};
        double t0 = plx_now();
        int rc = plx_rec_run_split(cfg, lay, all, &s, &sc);
        plx_log("View 2 fold %d: %.1fs", k, plx_now() - t0);
        /* test pairs were taken from pl in order, so fold k's pairs map back by a running index */
        for (int i = 0, j = 0; i < pl->n; i++)
            if (pl->p[i].fold == k) {
                score_unsup[i] = sc.test_unsup[j];
                if (score_sup) score_sup[i] = sc.test_sup[j];
                j++;
            }
        free(sc.test_unsup);
        free(sc.test_sup);
        plx_rec_split_free(&s);
        if (rc != PLX_OK) return rc;
    }
    int rc = plx_verif_evaluate(pl->p, pl->n, score_unsup, r_unsup);
    if (rc == PLX_OK && score_sup) rc = plx_verif_evaluate(pl->p, pl->n, score_sup, r_sup);
    return rc;
}

int plx_rec_eval_view1(const plx_match_cfg *cfg, const plx_desc_layout *lay, const plx_feats *all,
                       const plx_pairs *dev_train, const plx_pairs *dev_test, const plx_lfw *db,
                       double *score_unsup, double *score_sup, plx_verif_report *r_unsup,
                       plx_verif_report *r_sup) {
    plx_rec_split s;
    plx_rec_split_view1(dev_train, dev_test, db, &s);
    int nt = PLX_MAX(s.n_train, 1);
    plx_rec_scores sc = {.test_unsup = score_unsup, .test_sup = score_sup, .train_unsup = PLX_NEW(double, nt),
                         .train_sup = score_sup ? PLX_NEW(double, nt) : NULL};
    int rc = plx_rec_run_split(cfg, lay, all, &s, &sc);
    if (rc == PLX_OK)
        rc = plx_verif_split(s.train, s.n_train, sc.train_unsup, s.test, s.n_test, score_unsup, r_unsup);
    if (rc == PLX_OK && score_sup)
        rc = plx_verif_split(s.train, s.n_train, sc.train_sup, s.test, s.n_test, score_sup, r_sup);
    free(sc.train_unsup);
    free(sc.train_sup);
    plx_rec_split_free(&s);
    return rc;
}
