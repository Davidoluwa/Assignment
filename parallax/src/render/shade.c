#include "render/render.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

plx_shade plx_shade_defaults(void) {
    return (plx_shade){.light = {-0.4f, -0.5f, 1.0f}, .ambient = 0.15f, .smooth = true, .use_texture = false};
}

static void normalize3(double v[3]) {
    double n = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (n > 0) for (int k = 0; k < 3; k++) v[k] /= n;
}

/* Unnormalised face normal (v1 - v0) x (v2 - v0) in object space: length = 2 * area, so
 * summing these gives area-weighted vertex normals. */
static void face_normal(const plx_mesh *m, int t, double n[3]) {
    const float *a = m->v + 3 * m->tri[3 * t], *b = m->v + 3 * m->tri[3 * t + 1], *c = m->v + 3 * m->tri[3 * t + 2];
    double u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, w[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    n[0] = u[1] * w[2] - u[2] * w[1];
    n[1] = u[2] * w[0] - u[0] * w[2];
    n[2] = u[0] * w[1] - u[1] * w[0];
}

static void rotate(const double R[9], const double v[3], double out[3]) {
    for (int r = 0; r < 3; r++) out[r] = R[3 * r] * v[0] + R[3 * r + 1] * v[1] + R[3 * r + 2] * v[2];
}

plx_img plx_render_shaded(const plx_mesh *m, const plx_camera *cam, const plx_raster_opts *o, const plx_shade *sh) {
    PLX_ASSERT(cam->s > 0); /* a negative scale would mirror the surface and invert its normals */
    plx_raster_opts ro = *o;
    ro.want_ids = true;
    plx_raster r;
    plx_render(m, cam, &ro, &r);

    double L[3] = {sh->light[0], sh->light[1], sh->light[2]};
    normalize3(L);
    double *vn = NULL;
    if (sh->smooth) {
        vn = PLX_NEW(double, (size_t)m->nv * 3);
        for (int t = 0; t < m->nt; t++) {
            double fn[3];
            face_normal(m, t, fn);
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++) vn[3 * (size_t)m->tri[3 * t + j] + k] += fn[k];
        }
    }
    bool tex = sh->use_texture && (m->na == 1 || m->na == 3);
    int oc = tex ? m->na : 1;
    plx_img out = plx_img_new(o->w, o->h, oc);
    size_t n = (size_t)o->w * o->h;
    for (size_t i = 0; i < n; i++) {
        int t = r.tri[i];
        if (t < 0) {
            for (int k = 0; k < oc; k++) out.px[k * n + i] = o->bg;
            continue;
        }
        double nobj[3] = {0, 0, 0}, nc[3];
        if (vn) {
            for (int j = 0; j < 3; j++) {
                double b = r.bary.px[j * n + i];
                for (int k = 0; k < 3; k++) nobj[k] += b * vn[3 * (size_t)m->tri[3 * t + j] + k];
            }
        } else {
            face_normal(m, t, nobj);
        }
        rotate(cam->R, nobj, nc);
        normalize3(nc);
        if (nc[2] < 0) for (int k = 0; k < 3; k++) nc[k] = -nc[k]; /* two-sided */
        double lam = PLX_MAX(0.0, nc[0] * L[0] + nc[1] * L[1] + nc[2] * L[2]);
        float I = (float)(sh->ambient + (1.0 - sh->ambient) * lam);
        for (int k = 0; k < oc; k++) out.px[k * n + i] = tex ? I * r.attr.px[k * n + i] : I;
    }
    free(vn);
    plx_raster_free(&r);
    return out;
}

/* ---- multi-pose montage --------------------------------------------------------------------- */

/* Short enough to fit a small tile: only the non-zero angles, in whole degrees. */
static char *pose_label(const plx_pose *p) {
    const double deg = 180.0 / M_PI, a[3] = {p->yaw * deg, p->pitch * deg, p->roll * deg};
    static const char *const name[3] = {"yaw", "pitch", "roll"};
    char buf[64];
    int len = 0;
    for (int k = 0; k < 3; k++)
        if (lround(a[k]) != 0)
            len += snprintf(buf + len, sizeof buf - (size_t)len, "%s%s %+ld", len ? " " : "", name[k], lround(a[k]));
    return plx_strdup(len ? buf : "frontal");
}

plx_img plx_render_views(const plx_img *z, const uint8_t *mask, const plx_img *tex, const plx_pose *poses, int k) {
    PLX_ASSERT(k > 0 && (!tex || tex->c == 1 || tex->c == 3));
    int W = z->w, H = z->h;
    plx_mesh m = plx_mesh_heightfield(z, mask, tex);
    double c[3];
    plx_mesh_centroid(&m, c);

    /* One scale for all poses: the largest image-plane extent over all views must fit the
     * tile with a 2 px margin. Unreferenced vertices are ignored. */
    uint8_t *used = PLX_NEW(uint8_t, m.nv);
    for (int i = 0; i < 3 * m.nt; i++) used[m.tri[i]] = 1;
    double s = 1.0;
    for (int j = 0; j < k; j++) {
        plx_camera cam = plx_camera_make(poses[j].yaw, poses[j].pitch, poses[j].roll, 1.0, c, 0, 0);
        double ex = 0, ey = 0, p[3];
        for (int i = 0; i < m.nv; i++) {
            if (!used[i]) continue;
            plx_camera_project(&cam, m.v + 3 * i, p);
            ex = PLX_MAX(ex, fabs(p[0]));
            ey = PLX_MAX(ey, fabs(p[1]));
        }
        if (ex > 0) s = PLX_MIN(s, (0.5 * W - 2) / ex);
        if (ey > 0) s = PLX_MIN(s, (0.5 * H - 2) / ey);
    }
    free(used);

    int rows = tex ? 2 : 1;
    plx_img *tiles = PLX_NEW(plx_img, (size_t)rows * k);
    char **labels = PLX_NEW(char *, (size_t)rows * k);
    plx_raster_opts ro = {.w = W, .h = H, .cull = PLX_CULL_BACK, .bg = 0.0f};
    plx_shade sh = plx_shade_defaults();
    for (int j = 0; j < k; j++) {
        plx_camera cam = plx_camera_make(poses[j].yaw, poses[j].pitch, poses[j].roll, s, c, 0.5 * (W - 1),
                                         0.5 * (H - 1));
        tiles[j] = plx_render_shaded(&m, &cam, &ro, &sh);
        labels[j] = pose_label(&poses[j]);
        if (tex) {
            plx_raster r;
            plx_render(&m, &cam, &ro, &r);
            tiles[k + j] = r.attr;
            r.attr = (plx_img){0};
            plx_raster_free(&r);
        }
    }
    plx_montage_opts mo = plx_montage_defaults();
    mo.cols = k;
    mo.text_scale = W >= 200 ? 2 : 1;
    plx_img out = plx_viz_montage(tiles, rows * k, (const char *const *)labels, &mo);
    for (int i = 0; i < rows * k; i++) {
        plx_img_free(&tiles[i]);
        free(labels[i]);
    }
    free(tiles);
    free(labels);
    plx_mesh_free(&m);
    return out;
}
