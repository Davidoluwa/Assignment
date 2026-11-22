/* Confidence-weighted cell-histogram descriptors on the canonical texture. */
#include "rec/rec.h"

#include "ops/ops.h"

#include <stdlib.h>
#include <string.h>

static const char *const block_names[] = {"lbp1", "lbp2", "ori"};

int plx_desc_parse_blocks(const char *spec, plx_desc_params *p) {
    int n = 0;
    const char *s = spec;
    while (*s) {
        if (n == PLX_DESC_MAX_BLOCKS) {
            plx_log("descriptor spec '%s': more than %d blocks", spec, PLX_DESC_MAX_BLOCKS);
            return PLX_EINVAL;
        }
        plx_block_spec b = {.level = 0};
        size_t len = strcspn(s, "@:,");
        int kind = -1;
        for (int k = 0; k < 3; k++)
            if (strlen(block_names[k]) == len && strncmp(s, block_names[k], len) == 0) kind = k;
        s += len;
        char *end = NULL;
        if (kind >= 0 && *s == '@') {
            b.level = (int)strtol(s + 1, &end, 10);
            s = end;
        }
        if (kind < 0 || *s != ':' || b.level < 0 || b.level > 1) {
            plx_log("descriptor spec '%s': expected kind[@level]:cell with kind in lbp1|lbp2|ori, level 0|1", spec);
            return PLX_EINVAL;
        }
        b.cell = (int)strtol(s + 1, &end, 10);
        if (end == s + 1 || b.cell < 2 || (*end != ',' && *end != '\0')) {
            plx_log("descriptor spec '%s': bad cell size", spec);
            return PLX_EINVAL;
        }
        b.kind = (plx_block_kind)kind;
        p->block[n++] = b;
        s = *end == ',' ? end + 1 : end;
    }
    if (n == 0) {
        plx_log("descriptor spec '%s': no blocks", spec);
        return PLX_EINVAL;
    }
    p->n_blocks = n;
    return PLX_OK;
}

void plx_desc_describe(const plx_desc_params *p, char *buf, size_t cap) {
    size_t at = 0;
#define DESC_PUT(...) do { int k_ = snprintf(buf + at, at < cap ? cap - at : 0, __VA_ARGS__); if (k_ > 0) at += (size_t)k_; } while (0)
    DESC_PUT("blocks=");
    for (int i = 0; i < p->n_blocks; i++)
        DESC_PUT("%s%s@%d:%d", i ? "," : "", block_names[p->block[i].kind], p->block[i].level, p->block[i].cell);
    DESC_PUT(" region=%d,%d,%d,%d sqrt=%d", p->x0, p->y0, p->w, p->h, (int)p->sqrt_hist);
    const plx_photo_params *ph = &p->photo;
    DESC_PUT(" photo=%g,%g,%g,%g,%g", ph->gamma, ph->dog_s0, ph->dog_s1, ph->ce_alpha, ph->ce_tau);
    const plx_orient_params *o = &p->orient;
    DESC_PUT(" orient=%d,%d,%g,%g,%g,%g", o->k, (int)o->signed_, o->sigma_d, o->sigma_n, o->sigma_s, o->eps);
#undef DESC_PUT
}

static plx_cell_grid block_grid(const plx_desc_params *p, const plx_block_spec *b, int cells_x, int cells_y) {
    return (plx_cell_grid){.x0 = p->x0 >> b->level, .y0 = p->y0 >> b->level, .w = p->w >> b->level,
                           .h = p->h >> b->level, .cells_x = cells_x, .cells_y = cells_y,
                           .norm = p->sqrt_hist ? PLX_HNORM_L1_SQRT : PLX_HNORM_L1, .eps = 1e-6f};
}

plx_desc_layout plx_desc_layout_of(const plx_desc_params *p) {
    PLX_ASSERT(p->n_blocks >= 1 && p->n_blocks <= PLX_DESC_MAX_BLOCKS);
    /* even region bounds keep every level-1 cell boundary on a level-0 pixel boundary */
    PLX_ASSERT(p->x0 >= 0 && p->y0 >= 0 && p->w > 0 && p->h > 0 && p->x0 % 2 == 0 && p->y0 % 2 == 0 &&
               p->w % 2 == 0 && p->h % 2 == 0 && p->x0 + p->w <= PLX_REC_TEX_W && p->y0 + p->h <= PLX_REC_TEX_H);
    plx_desc_layout L = {.n_blocks = p->n_blocks};
    for (int i = 0; i < p->n_blocks; i++) {
        const plx_block_spec *b = &p->block[i];
        int wl = p->w >> b->level, hl = p->h >> b->level;
        /* nearest whole number of cells; plx_cell_grid spreads the remainder evenly */
        L.cells_x[i] = PLX_CLAMP((wl + b->cell / 2) / b->cell, 1, wl);
        L.cells_y[i] = PLX_CLAMP((hl + b->cell / 2) / b->cell, 1, hl);
        L.bins[i] = b->kind == PLX_BLOCK_ORI ? p->orient.k : PLX_LBP_UNIFORM_BINS;
        L.off[i] = L.d;
        L.cell_off[i] = L.n_cells;
        L.dim[i] = L.cells_x[i] * L.cells_y[i] * L.bins[i];
        L.d += L.dim[i];
        L.n_cells += L.cells_x[i] * L.cells_y[i];
    }
    return L;
}

/* Per-level working images, made on first use. */
typedef struct {
    plx_img raw, conf, norm; /* norm: photometrically normalised raw */
    bool has_norm;
} level_imgs;

static void level_prepare(level_imgs *lv, int level, const plx_img *tex, const plx_img *conf) {
    if (lv[level].raw.px) return;
    if (level == 0) {
        lv[0].raw = plx_img_clone(tex);
        if (conf) {
            lv[0].conf = plx_img_clone(conf);
        } else {
            lv[0].conf = plx_img_new(tex->w, tex->h, 1);
            plx_img_fill(&lv[0].conf, 1.0f);
        }
    } else {
        level_prepare(lv, 0, tex, conf);
        lv[1].raw = plx_downsample2(&lv[0].raw);
        lv[1].conf = plx_downsample2(&lv[0].conf);
    }
}

static const plx_img *level_norm(level_imgs *lv, const plx_desc_params *p, int level) {
    if (!lv[level].has_norm) {
        lv[level].norm = plx_img_clone(&lv[level].raw);
        plx_photo_normalize(&lv[level].norm, &lv[level].conf, &p->photo);
        lv[level].has_norm = true;
    }
    return &lv[level].norm;
}

void plx_desc_compute(const plx_desc_params *p, const plx_img *tex, const plx_img *conf, float *desc,
                      float *cell_w) {
    PLX_ASSERT(tex->w == PLX_REC_TEX_W && tex->h == PLX_REC_TEX_H && tex->c == 1);
    PLX_ASSERT(!conf || (conf->w == tex->w && conf->h == tex->h && conf->c == 1));
    plx_desc_layout L = plx_desc_layout_of(p);
    uint8_t uniform[256];
    plx_lbp_uniform_table(uniform);
    level_imgs lv[2];
    memset(lv, 0, sizeof lv);
    uint8_t *codes = plx_malloc((size_t)PLX_REC_TEX_W * PLX_REC_TEX_H);
    float *cw = PLX_NEW(float, L.n_cells);

    for (int i = 0; i < p->n_blocks; i++) {
        const plx_block_spec *b = &p->block[i];
        level_prepare(lv, b->level, tex, conf);
        const plx_img *cf = &lv[b->level].conf;
        plx_cell_grid g = block_grid(p, b, L.cells_x[i], L.cells_y[i]);
        float *out = desc + L.off[i], *bw = cw + L.cell_off[i];
        if (b->kind == PLX_BLOCK_ORI) {
            plx_img ch = plx_orient_channels(&lv[b->level].raw, &p->orient);
            plx_hist_channels(&ch, cf->px, &g, out, bw);
            plx_img_free(&ch);
        } else {
            const plx_img *nv = level_norm(lv, p, b->level);
            size_t np = (size_t)nv->w * nv->h;
            plx_lbp(nv, 0, b->kind == PLX_BLOCK_LBP1 ? 1 : 2, codes);
            for (size_t k = 0; k < np; k++) codes[k] = uniform[codes[k]];
            plx_hist_labels(codes, nv->w, nv->h, PLX_LBP_UNIFORM_BINS, cf->px, &g, out, bw);
        }
        if (cell_w) {
            /* mean confidence of the cell: total weight over its pixel count */
            int ncell = L.cells_x[i] * L.cells_y[i];
            int *area = PLX_NEW(int, ncell);
            for (int y = 0; y < g.h; y++)
                for (int x = 0; x < g.w; x++)
                    area[(y * g.cells_y / g.h) * g.cells_x + x * g.cells_x / g.w]++;
            for (int c = 0; c < ncell; c++) cell_w[L.cell_off[i] + c] = bw[c] / (float)area[c];
            free(area);
        }
    }
    for (int k = 0; k < 2; k++) {
        plx_img_free(&lv[k].raw);
        plx_img_free(&lv[k].conf);
        plx_img_free(&lv[k].norm);
    }
    free(codes);
    free(cw);
}
