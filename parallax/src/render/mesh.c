#include "render/render.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

plx_mesh plx_mesh_new(int nv, int nt, int na) {
    PLX_ASSERT(nv >= 0 && nt >= 0 && na >= 0);
    plx_mesh m = {nv, nt, na, PLX_NEW(float, (size_t)nv * 3), NULL, PLX_NEW(int, (size_t)nt * 3)};
    if (na > 0) m.attr = PLX_NEW(float, (size_t)nv * na);
    return m;
}

void plx_mesh_free(plx_mesh *m) {
    free(m->v);
    free(m->attr);
    free(m->tri);
    memset(m, 0, sizeof *m);
}

plx_mesh plx_mesh_heightfield(const plx_img *z, const uint8_t *mask, const plx_img *tex) {
    int W = z->w, H = z->h, na = tex ? tex->c : 0;
    PLX_ASSERT(!tex || (tex->w == W && tex->h == H));
    size_t n = (size_t)W * H;
    uint8_t *ok = PLX_NEW(uint8_t, n);
    for (size_t i = 0; i < n; i++) ok[i] = isfinite(z->px[i]) && (!mask || mask[i]);
    int cells = 0;
    for (int y = 0; y + 1 < H; y++)
        for (int x = 0; x + 1 < W; x++) {
            size_t i = (size_t)y * W + x;
            cells += ok[i] && ok[i + 1] && ok[i + W] && ok[i + W + 1];
        }
    plx_mesh m = plx_mesh_new((int)n, 2 * cells, na);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            size_t i = (size_t)y * W + x;
            m.v[3 * i] = (float)x;
            m.v[3 * i + 1] = (float)y;
            m.v[3 * i + 2] = ok[i] ? z->px[i] : 0.0f;
            for (int k = 0; k < na; k++) m.attr[i * na + k] = PLX_PX(tex, x, y, k);
        }
    int *t = m.tri;
    for (int y = 0; y + 1 < H; y++)
        for (int x = 0; x + 1 < W; x++) {
            int i = y * W + x;
            if (!(ok[i] && ok[i + 1] && ok[i + W] && ok[i + W + 1])) continue;
            /* Both clockwise on screen (y down), i.e. front-facing seen from +z. */
            t[0] = i; t[1] = i + 1; t[2] = i + W;
            t[3] = i + 1; t[4] = i + W + 1; t[5] = i + W;
            t += 6;
        }
    free(ok);
    return m;
}

void plx_mesh_centroid(const plx_mesh *m, double c[3]) {
    uint8_t *used = PLX_NEW(uint8_t, m->nv);
    for (int i = 0; i < 3 * m->nt; i++) used[m->tri[i]] = 1;
    double s[3] = {0, 0, 0};
    int cnt = 0;
    for (int i = 0; i < m->nv; i++) {
        if (!used[i]) continue;
        for (int k = 0; k < 3; k++) s[k] += m->v[3 * i + k];
        cnt++;
    }
    free(used);
    for (int k = 0; k < 3; k++) c[k] = cnt ? s[k] / cnt : 0.0;
}

/* ---- camera -------------------------------------------------------------------------------- */

static void matmul3(const double A[9], const double B[9], double C[9]) {
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) C[3 * r + c] = A[3 * r] * B[c] + A[3 * r + 1] * B[3 + c] + A[3 * r + 2] * B[6 + c];
}

void plx_rot_ypr(double yaw, double pitch, double roll, double R[9]) {
    double cy = cos(yaw), sy = sin(yaw), cp = cos(pitch), sp = sin(pitch), cr = cos(roll), sr = sin(roll);
    const double Ry[9] = {cy, 0, sy, 0, 1, 0, -sy, 0, cy};
    const double Rx[9] = {1, 0, 0, 0, cp, -sp, 0, sp, cp};
    const double Rz[9] = {cr, -sr, 0, sr, cr, 0, 0, 0, 1};
    double T[9];
    matmul3(Rx, Ry, T);
    matmul3(Rz, T, R);
}

plx_camera plx_camera_make(double yaw, double pitch, double roll, double s, const double centre[3], double tx,
                           double ty) {
    plx_camera cam = {.s = s, .t = {tx, ty, 0}};
    plx_rot_ypr(yaw, pitch, roll, cam.R);
    for (int k = 0; k < 3; k++) cam.centre[k] = centre ? centre[k] : 0.0;
    return cam;
}

void plx_camera_project(const plx_camera *cam, const float v[3], double p[3]) {
    double d[3] = {v[0] - cam->centre[0], v[1] - cam->centre[1], v[2] - cam->centre[2]};
    for (int r = 0; r < 3; r++)
        p[r] = cam->s * (cam->R[3 * r] * d[0] + cam->R[3 * r + 1] * d[1] + cam->R[3 * r + 2] * d[2]) + cam->t[r];
}
