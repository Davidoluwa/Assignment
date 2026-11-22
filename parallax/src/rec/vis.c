/* Visibility of the canonical face surface under a photo's head pose (rung L3). */
#include "rec/rec.h"

#include "render/render.h"

#include <math.h>
#include <stdlib.h>

/* canvas for the occlusion test: the rotated surface stays within the canonical frame plus a
 * margin for |yaw|, |pitch| up to ~60 degrees */
enum { VIS_MARGIN = 48 };

void plx_rec_visibility(const plx_img *z, double yaw, double pitch, float power, bool relative, plx_img *conf) {
    const int W = PLX_REC_TEX_W, H = PLX_REC_TEX_H;
    PLX_ASSERT(z->w == W && z->h == H && z->c == 1 && conf->w == W && conf->h == H && conf->c == 1);
    double R[9];
    plx_rot_ypr(yaw, pitch, 0.0, R);

    /* z-buffer of the surface seen from this pose; a point is hidden when the rendered surface
     * in front of it is more than a pixel closer to the camera */
    plx_mesh mesh = plx_mesh_heightfield(z, NULL, NULL);
    const double centre[3] = {PLX_REC_C0X, PLX_REC_C0Y, 0.0};
    plx_camera cam = plx_camera_make(yaw, pitch, 0.0, 1.0, centre, PLX_REC_C0X + VIS_MARGIN, PLX_REC_C0Y + VIS_MARGIN);
    plx_raster_opts ro = {.w = W + 2 * VIS_MARGIN, .h = H + 2 * VIS_MARGIN, .cull = PLX_CULL_NONE};
    plx_raster ras;
    plx_render(&mesh, &cam, &ro, &ras);

    float *f = PLX_NEW(float, (size_t)W * H); /* factor per pixel; NaN = no surface there */
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            float zv = PLX_PX(z, x, y, 0);
            f[y * W + x] = NAN;
            if (!isfinite(zv)) continue;
            /* surface slope by central differences, one-sided at the edge of the depth support */
            float zl = x > 0 ? PLX_PX(z, x - 1, y, 0) : NAN, zr = x < W - 1 ? PLX_PX(z, x + 1, y, 0) : NAN;
            float zu = y > 0 ? PLX_PX(z, x, y - 1, 0) : NAN, zd = y < H - 1 ? PLX_PX(z, x, y + 1, 0) : NAN;
            double gx = isfinite(zl) && isfinite(zr) ? 0.5 * (zr - zl) : isfinite(zr) ? zr - zv : isfinite(zl) ? zv - zl : 0.0;
            double gy = isfinite(zu) && isfinite(zd) ? 0.5 * (zd - zu) : isfinite(zd) ? zd - zv : isfinite(zu) ? zv - zu : 0.0;
            /* outward normal of z(x, y) with z towards the viewer: (-z_x, -z_y, 1) */
            double nn = sqrt(gx * gx + gy * gy + 1.0), nx = -gx / nn, ny = -gy / nn, nz = 1.0 / nn;
            double c = R[6] * nx + R[7] * ny + R[8] * nz; /* rotated normal . view direction (+z) */
            if (relative) c /= nz;                         /* foreshortening relative to the frontal view */
            c = PLX_CLAMP(c, 0.0, 1.0);
            double p[3];
            const float v[3] = {(float)x, (float)y, zv};
            plx_camera_project(&cam, v, p);
            int px = (int)lround(p[0]), py = (int)lround(p[1]);
            if (px >= 0 && py >= 0 && px < ras.w && py < ras.h) {
                float d = PLX_PX(&ras.depth, px, py, 0);
                if (isfinite(d) && d > p[2] + 1.0) c = 0.0;
            }
            f[y * W + x] = (float)(power == 1.0f ? c : pow(c, power));
        }
    /* outside the depth support (hair, background, beyond the core) the model makes no claim of
     * its own; the nearest supported pixel of the same row speaks for it, so a far cheek's
     * low visibility extends to the contour beyond it */
    for (int y = 0; y < H; y++) {
        int first = -1, last = -1;
        for (int x = 0; x < W; x++)
            if (isfinite(f[y * W + x])) {
                if (first < 0) first = x;
                last = x;
            }
        for (int x = 0; x < W; x++) {
            float v = f[y * W + x];
            if (!isfinite(v)) v = first < 0 ? 1.0f : f[y * W + (x < first ? first : last)];
            PLX_PX(conf, x, y, 0) *= v;
        }
    }
    free(f);
    plx_raster_free(&ras);
    plx_mesh_free(&mesh);
}
