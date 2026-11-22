// plx-deps: img core
#include "test.h"

#include "core/core.h"
#include "img/image.h"
#include "img/imgio.h"
#include "img/jpeg.h"
#include "img/png.h"
#include "img/pnm.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static char g_tmp[256];
static const char *tmp_path(const char *name) {
    snprintf(g_tmp, sizeof g_tmp, "/tmp/plx_test_img_%d_%s", (int)getpid(), name);
    return g_tmp;
}

/* ---- embedded JPEGs written by PIL, with CRC-32 of PIL's decoded pixels ------------------- */
/* 33x25 RGB, 942 bytes; PIL pixel CRC-32 0x0E961BD4 */
static const uint8_t k_prog_rst[] = {
    0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
    0xFF, 0xDB, 0x00, 0x43, 0x00, 0x0A, 0x07, 0x07, 0x08, 0x07, 0x06, 0x0A, 0x08, 0x08, 0x08, 0x0B, 0x0A, 0x0A, 0x0B, 0x0E,
    0x18, 0x10, 0x0E, 0x0D, 0x0D, 0x0E, 0x1D, 0x15, 0x16, 0x11, 0x18, 0x23, 0x1F, 0x25, 0x24, 0x22, 0x1F, 0x22, 0x21, 0x26,
    0x2B, 0x37, 0x2F, 0x26, 0x29, 0x34, 0x29, 0x21, 0x22, 0x30, 0x41, 0x31, 0x34, 0x39, 0x3B, 0x3E, 0x3E, 0x3E, 0x25, 0x2E,
    0x44, 0x49, 0x43, 0x3C, 0x48, 0x37, 0x3D, 0x3E, 0x3B, 0xFF, 0xDB, 0x00, 0x43, 0x01, 0x0A, 0x0B, 0x0B, 0x0E, 0x0D, 0x0E,
    0x1C, 0x10, 0x10, 0x1C, 0x3B, 0x28, 0x22, 0x28, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B,
    0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B,
    0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0x3B, 0xFF, 0xC2,
    0x00, 0x11, 0x08, 0x00, 0x19, 0x00, 0x21, 0x03, 0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01, 0xFF, 0xC4, 0x00,
    0x19, 0x00, 0x00, 0x03, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04,
    0x06, 0x05, 0x02, 0x03, 0xFF, 0xC4, 0x00, 0x17, 0x01, 0x00, 0x03, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0xFF, 0xDD, 0x00, 0x04, 0x00, 0x02, 0xFF, 0xDA, 0x00, 0x0C, 0x03,
    0x01, 0x00, 0x02, 0x10, 0x03, 0x10, 0x00, 0x00, 0x01, 0x8D, 0x71, 0x4B, 0x28, 0xA9, 0x75, 0x69, 0x66, 0x83, 0xFF, 0xD0,
    0x8E, 0x01, 0x16, 0x27, 0xA7, 0x5A, 0xCF, 0xFF, 0xD1, 0xCE, 0x70, 0x72, 0xD6, 0x50, 0xE8, 0xD7, 0xFF, 0xC4, 0x00, 0x1F,
    0x10, 0x00, 0x02, 0x02, 0x01, 0x04, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x03, 0x00,
    0x01, 0x04, 0x10, 0x11, 0x14, 0x41, 0x31, 0x34, 0x42, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x01, 0x05, 0x02, 0x8B,
    0x45, 0x9D, 0x7F, 0xFF, 0xD0, 0x24, 0x90, 0xDC, 0xFF, 0xD1, 0xD3, 0xFF, 0xD2, 0xC3, 0xF5, 0xF2, 0xCB, 0x71, 0xFF, 0xD3,
    0xD3, 0xFF, 0xD4, 0x05, 0x82, 0xA1, 0x32, 0xE8, 0xBF, 0xFF, 0xD5, 0xBA, 0xB6, 0x17, 0x1C, 0x08, 0x7F, 0xFF, 0xD6, 0xE1,
    0x26, 0x77, 0xFF, 0xD7, 0xF8, 0x4F, 0x8F, 0xFF, 0xD0, 0xEA, 0x7F, 0xFF, 0xC4, 0x00, 0x18, 0x11, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x11, 0x02, 0x10, 0xFF, 0xDA, 0x00,
    0x08, 0x01, 0x03, 0x01, 0x01, 0x3F, 0x01, 0xE4, 0xD6, 0xE8, 0xCB, 0xFF, 0xD0, 0x87, 0x2F, 0xFF, 0xD1, 0x5D, 0x7C, 0xFF,
    0xC4, 0x00, 0x17, 0x11, 0x00, 0x03, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x01, 0x11, 0x02, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x02, 0x01, 0x01, 0x3F, 0x01, 0xD3, 0x86, 0x5D, 0x3F, 0xFF, 0xD0,
    0x1E, 0x6A, 0x87, 0xFF, 0xD1, 0x59, 0x88, 0x87, 0xFF, 0xC4, 0x00, 0x1E, 0x10, 0x00, 0x02, 0x02, 0x02, 0x02, 0x03, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x11, 0x10, 0x71, 0x21, 0x31, 0x22, 0x61, 0xE1,
    0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x06, 0x3F, 0x02, 0x2E, 0x9B, 0x3F, 0xFF, 0xD0, 0xE6, 0x35, 0x8F, 0xFF, 0xD1,
    0xC7, 0xFF, 0xD2, 0x5D, 0xC1, 0xFB, 0x24, 0xEF, 0x48, 0xFF, 0xD3, 0xC7, 0xFF, 0xD4, 0x4A, 0x31, 0x4B, 0x92, 0xD4, 0x5B,
    0x8D, 0x1F, 0xFF, 0xD5, 0xE8, 0xF2, 0x8A, 0x67, 0xFF, 0xD6, 0xFA, 0x2D, 0x9F, 0xFF, 0xD7, 0xC7, 0xFF, 0xD0, 0x5B, 0xC7,
    0xFF, 0xC4, 0x00, 0x22, 0x10, 0x00, 0x02, 0x01, 0x02, 0x05, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x01, 0x11, 0x31, 0x61, 0x10, 0x21, 0x41, 0x51, 0x71, 0x91, 0xA1, 0xB1, 0xC1, 0xF0, 0xFF, 0xDA, 0x00, 0x08,
    0x01, 0x01, 0x00, 0x01, 0x3F, 0x21, 0x3A, 0x44, 0x11, 0xFF, 0xD0, 0x7C, 0x91, 0x16, 0xE8, 0xA1, 0xFF, 0xD1, 0x2A, 0xE1,
    0x66, 0x7F, 0xFF, 0xD2, 0x5C, 0xE7, 0xBC, 0x48, 0x5F, 0x54, 0x21, 0x1F, 0xFF, 0xD3, 0xD7, 0x0F, 0xFF, 0xD4, 0x82, 0x33,
    0x62, 0x24, 0xC1, 0x26, 0x94, 0x3F, 0xFF, 0xD5, 0x69, 0x72, 0x6A, 0xD4, 0x47, 0x0B, 0x14, 0x3F, 0xFF, 0xD6, 0xB4, 0xC3,
    0xED, 0x7C, 0x1F, 0xFF, 0xD7, 0xD3, 0xC1, 0xED, 0x3F, 0xFF, 0xD0, 0x7F, 0x6B, 0x61, 0xFF, 0xDA, 0x00, 0x0C, 0x03, 0x01,
    0x00, 0x02, 0x00, 0x03, 0x00, 0x00, 0x00, 0x10, 0xCD, 0xBF, 0xFF, 0xD0, 0x0A, 0xDF, 0xFF, 0xD1, 0x20, 0xCF, 0xFF, 0xC4,
    0x00, 0x16, 0x11, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
    0x00, 0x11, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x03, 0x01, 0x01, 0x3F, 0x10, 0x2C, 0x18, 0xDE, 0x17, 0xFF, 0xD0, 0x9D, 0xE9,
    0x7F, 0xFF, 0xD1, 0x5D, 0x2D, 0x6F, 0xFF, 0xC4, 0x00, 0x17, 0x11, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x11, 0x21, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x02, 0x01, 0x01, 0x3F,
    0x10, 0x61, 0xA4, 0x83, 0xB7, 0xFF, 0xD0, 0x8B, 0x65, 0xFF, 0xD1, 0x28, 0xC5, 0xFF, 0xC4, 0x00, 0x22, 0x10, 0x01, 0x00,
    0x02, 0x02, 0x00, 0x06, 0x03, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x11, 0x21, 0x31, 0x10,
    0x41, 0x51, 0x71, 0xB1, 0xF0, 0x61, 0x81, 0xA1, 0xF1, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x01, 0x3F, 0x10, 0xD4,
    0x06, 0x6C, 0x72, 0xD5, 0x3F, 0xFF, 0xD0, 0x00, 0x08, 0xDD, 0x11, 0x2C, 0x8E, 0xCC, 0x4F, 0xFF, 0xD1, 0xC4, 0x04, 0xC1,
    0x4E, 0x80, 0xB6, 0x7F, 0xFF, 0xD2, 0xCD, 0x41, 0x44, 0x9B, 0x50, 0x0D, 0x3C, 0x9A, 0xFD, 0x8A, 0xA1, 0x55, 0x64, 0x38,
    0x67, 0xFF, 0xD3, 0x6E, 0xD7, 0xBF, 0x9E, 0x1F, 0xFF, 0xD4, 0xB4, 0x0D, 0x4B, 0xCA, 0xE3, 0x4B, 0xB8, 0xDD, 0x60, 0x29,
    0x45, 0x56, 0x6B, 0xA9, 0x8F, 0x33, 0xFF, 0xD5, 0xB1, 0x05, 0x91, 0x55, 0xCC, 0xB8, 0x83, 0x7A, 0x94, 0x5C, 0xAF, 0xBE,
    0xE7, 0xFF, 0xD6, 0xFE, 0xAE, 0x7A, 0x0E, 0xA9, 0xFF, 0xD7, 0xF4, 0x7D, 0xCD, 0xFD, 0xBC, 0x19, 0xFF, 0xD0, 0xE1, 0x0F,
    0xFF, 0xD9,
};
/* 20x12 L, 391 bytes; PIL pixel CRC-32 0xB2853AC6 */
static const uint8_t k_gray_prog[] = {
    0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
    0xFF, 0xDB, 0x00, 0x43, 0x00, 0x0A, 0x07, 0x07, 0x08, 0x07, 0x06, 0x0A, 0x08, 0x08, 0x08, 0x0B, 0x0A, 0x0A, 0x0B, 0x0E,
    0x18, 0x10, 0x0E, 0x0D, 0x0D, 0x0E, 0x1D, 0x15, 0x16, 0x11, 0x18, 0x23, 0x1F, 0x25, 0x24, 0x22, 0x1F, 0x22, 0x21, 0x26,
    0x2B, 0x37, 0x2F, 0x26, 0x29, 0x34, 0x29, 0x21, 0x22, 0x30, 0x41, 0x31, 0x34, 0x39, 0x3B, 0x3E, 0x3E, 0x3E, 0x25, 0x2E,
    0x44, 0x49, 0x43, 0x3C, 0x48, 0x37, 0x3D, 0x3E, 0x3B, 0xFF, 0xC2, 0x00, 0x0B, 0x08, 0x00, 0x0C, 0x00, 0x14, 0x01, 0x01,
    0x11, 0x00, 0xFF, 0xC4, 0x00, 0x17, 0x00, 0x00, 0x03, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x03, 0x04, 0x06, 0x05, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x98, 0x7B, 0x1E,
    0xD1, 0x53, 0xFF, 0x00, 0xFF, 0xC4, 0x00, 0x1A, 0x10, 0x00, 0x02, 0x02, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x03, 0x02, 0x04, 0x10, 0x12, 0x13, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x01,
    0x05, 0x02, 0xAC, 0xAE, 0xCE, 0xB5, 0x5A, 0x09, 0x18, 0x11, 0xD9, 0x85, 0x70, 0x27, 0xFF, 0xC4, 0x00, 0x1E, 0x10, 0x01,
    0x00, 0x01, 0x02, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x11, 0x31, 0x02,
    0x10, 0x12, 0x21, 0x22, 0x32, 0x91, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x06, 0x3F, 0x02, 0x30, 0xB6, 0x9C, 0x55,
    0xCB, 0x4D, 0x88, 0xD7, 0x7A, 0x4E, 0x87, 0x93, 0xFF, 0xC4, 0x00, 0x1C, 0x10, 0x01, 0x00, 0x02, 0x02, 0x03, 0x01, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x11, 0x21, 0x41, 0x10, 0x31, 0x51, 0x71, 0xFF, 0xDA,
    0x00, 0x08, 0x01, 0x01, 0x00, 0x01, 0x3F, 0x21, 0xCA, 0x41, 0xEE, 0xA3, 0xC5, 0x0E, 0xAB, 0x8A, 0xD3, 0x27, 0x90, 0x5E,
    0xD4, 0xB7, 0x1E, 0xDA, 0xFD, 0x8F, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x00, 0x10, 0x97, 0xFF, 0xC4, 0x00,
    0x1C, 0x10, 0x01, 0x00, 0x03, 0x01, 0x00, 0x03, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
    0x11, 0x21, 0x31, 0x41, 0x51, 0x61, 0x71, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x01, 0x3F, 0x10, 0x21, 0x4C, 0xE9,
    0x48, 0x7E, 0xBE, 0x20, 0x54, 0x35, 0xC3, 0xBF, 0x2A, 0x5D, 0x76, 0x56, 0x9C, 0x60, 0x56, 0x6B, 0x5E, 0x7A, 0xE4, 0x3A,
    0xF2, 0x05, 0x1B, 0xB1, 0x81, 0xB3, 0x2C, 0x16, 0x7F, 0xFF, 0xD9,
};
/* 24x17 RGB, 444 bytes; PIL pixel CRC-32 0xAF775B42 */
static const uint8_t k_422_opt_rst[] = {
    0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
    0xFF, 0xDB, 0x00, 0x43, 0x00, 0x0D, 0x09, 0x0A, 0x0B, 0x0A, 0x08, 0x0D, 0x0B, 0x0A, 0x0B, 0x0E, 0x0E, 0x0D, 0x0F, 0x13,
    0x20, 0x15, 0x13, 0x12, 0x12, 0x13, 0x27, 0x1C, 0x1E, 0x17, 0x20, 0x2E, 0x29, 0x31, 0x30, 0x2E, 0x29, 0x2D, 0x2C, 0x33,
    0x3A, 0x4A, 0x3E, 0x33, 0x36, 0x46, 0x37, 0x2C, 0x2D, 0x40, 0x57, 0x41, 0x46, 0x4C, 0x4E, 0x52, 0x53, 0x52, 0x32, 0x3E,
    0x5A, 0x61, 0x5A, 0x50, 0x60, 0x4A, 0x51, 0x52, 0x4F, 0xFF, 0xDB, 0x00, 0x43, 0x01, 0x0E, 0x0E, 0x0E, 0x13, 0x11, 0x13,
    0x26, 0x15, 0x15, 0x26, 0x4F, 0x35, 0x2D, 0x35, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F,
    0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F,
    0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0x4F, 0xFF, 0xC0,
    0x00, 0x11, 0x08, 0x00, 0x11, 0x00, 0x18, 0x03, 0x01, 0x21, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01, 0xFF, 0xC4, 0x00,
    0x17, 0x00, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00,
    0x04, 0x07, 0xFF, 0xC4, 0x00, 0x2A, 0x10, 0x00, 0x01, 0x04, 0x01, 0x01, 0x04, 0x0B, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x03, 0x11, 0x04, 0x12, 0x05, 0x06, 0x31, 0x33, 0x13, 0x16, 0x21, 0x22, 0x32, 0x35,
    0x41, 0x42, 0x72, 0x82, 0xB1, 0xF0, 0xFF, 0xC4, 0x00, 0x17, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x02, 0x01, 0x00, 0xFF, 0xC4, 0x00, 0x1A, 0x11, 0x00, 0x02, 0x03, 0x01,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x12, 0x11, 0x21, 0xFF,
    0xDD, 0x00, 0x04, 0x00, 0x03, 0xFF, 0xDA, 0x00, 0x0C, 0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3F, 0x00, 0xE7,
    0x18, 0xD0, 0x99, 0xF2, 0x63, 0x85, 0xBC, 0x5C, 0x69, 0x25, 0xCB, 0xDD, 0x57, 0xC3, 0x8E, 0x1F, 0xD3, 0x42, 0xDA, 0x34,
    0x46, 0xA3, 0x68, 0xE7, 0x3C, 0xB1, 0x21, 0x0D, 0x20, 0xCC, 0xAD, 0x0C, 0x95, 0xCC, 0x06, 0xC0, 0x34, 0xA5, 0x61, 0x8C,
    0xB0, 0x77, 0x5F, 0x0B, 0x16, 0x46, 0xC9, 0x3C, 0xF2, 0xBE, 0x56, 0x77, 0x81, 0x07, 0x48, 0xB1, 0xC3, 0x87, 0x6A, 0xDD,
    0xB5, 0x36, 0x93, 0xF2, 0x22, 0x8C, 0x33, 0x58, 0x7D, 0x51, 0x0E, 0x6F, 0x86, 0xD6, 0x5B, 0x5F, 0xA8, 0x4A, 0xA7, 0xC4,
    0xCF, 0xFF, 0xD0, 0x39, 0xD5, 0x61, 0x3C, 0x06, 0x56, 0xCE, 0xE8, 0xDE, 0x7D, 0x1C, 0x2C, 0x12, 0xA5, 0x58, 0x27, 0x42,
    0xA1, 0xCF, 0x1F, 0xDE, 0xE0, 0xB2, 0xE6, 0xF9, 0x94, 0x5F, 0x02, 0x99, 0x92, 0x8D, 0x8E, 0xE4, 0xFD, 0x8F, 0xE1, 0x52,
    0xC3, 0x8F, 0xFF, 0xD9,
};

/* Real LFW files with CRC-32 of PIL's RGB pixels; checked only when data/lfw exists. */
static const struct { const char *path; uint32_t crc; } k_lfw_ref[] = {
    {"data/lfw/AJ_Cook/AJ_Cook_0001.jpg", 0x6C12AC79u},
    {"data/lfw/George_W_Bush/George_W_Bush_0001.jpg", 0xACB07FA5u},
    {"data/lfw/Zico/Zico_0001.jpg", 0x8A9572C2u},
};

static const char *k_fuzz_corpus[] = {
    "data/lfw/AJ_Cook/AJ_Cook_0001.jpg",
    "data/lfw/Andre_Agassi/Andre_Agassi_0017.jpg",
    "data/lfw/Beatriz_Merino/Beatriz_Merino_0001.jpg",
    "data/lfw/Carla_Del_Ponte/Carla_Del_Ponte_0002.jpg",
    "data/lfw/Colin_Powell/Colin_Powell_0011.jpg",
    "data/lfw/David_Nalbandian/David_Nalbandian_0014.jpg",
    "data/lfw/Edwina_Currie/Edwina_Currie_0003.jpg",
    "data/lfw/George_Blaney/George_Blaney_0001.jpg",
    "data/lfw/George_W_Bush/George_W_Bush_0461.jpg",
    "data/lfw/Halle_Berry/Halle_Berry_0007.jpg",
    "data/lfw/Ishaq_Shahryar/Ishaq_Shahryar_0001.jpg",
    "data/lfw/Jeb_Bush/Jeb_Bush_0005.jpg",
    "data/lfw/John_Blaney/John_Blaney_0002.jpg",
    "data/lfw/Julian_Fantino/Julian_Fantino_0001.jpg",
    "data/lfw/Lance_Armstrong/Lance_Armstrong_0008.jpg",
    "data/lfw/Luiz_Inacio_Lula_da_Silva/Luiz_Inacio_Lula_da_Silva_0044.jpg",
    "data/lfw/Megawati_Sukarnoputri/Megawati_Sukarnoputri_0026.jpg",
    "data/lfw/Muhammad_Saeed_al-Sahhaf/Muhammad_Saeed_al-Sahhaf_0004.jpg",
    "data/lfw/Paul_Burrell/Paul_Burrell_0003.jpg",
    "data/lfw/Ray_Young/Ray_Young_0001.jpg",
    "data/lfw/Rolf_Eckrodt/Rolf_Eckrodt_0002.jpg",
    "data/lfw/Shireen_Amir_Begum/Shireen_Amir_Begum_0001.jpg",
    "data/lfw/Tiger_Woods/Tiger_Woods_0008.jpg",
    "data/lfw/Valentino_Rossi/Valentino_Rossi_0002.jpg",
};

static uint32_t decode_crc(const uint8_t *data, size_t len, int *w, int *h, int *c) {
    uint8_t *px;
    int st = plx_jpeg_decode_u8(data, len, w, h, c, &px);
    if (st < 0) return 0;
    uint32_t crc = plx_crc32(0, px, (size_t)*w * *h * *c);
    free(px);
    return crc;
}

TEST(jpeg_matches_pil_on_embedded_files) {
    int w, h, c;
    CHECK(decode_crc(k_prog_rst, sizeof k_prog_rst, &w, &h, &c) == 0x0E961BD4u);
    CHECK(w == 33 && h == 25 && c == 3);
    CHECK(decode_crc(k_gray_prog, sizeof k_gray_prog, &w, &h, &c) == 0xB2853AC6u);
    CHECK(w == 20 && h == 12 && c == 1);
    CHECK(decode_crc(k_422_opt_rst, sizeof k_422_opt_rst, &w, &h, &c) == 0xAF775B42u);
    CHECK(w == 24 && h == 17 && c == 3);

    CHECK(plx_jpeg_info(k_prog_rst, sizeof k_prog_rst, &w, &h, &c) == PLX_OK && w == 33 && h == 25 && c == 3);
    plx_img im;
    CHECK(plx_jpeg_decode(k_gray_prog, sizeof k_gray_prog, &im) == PLX_OK);
    CHECK(im.w == 20 && im.h == 12 && im.c == 1);
    float lo = 1, hi = 0;
    for (int i = 0; i < 240; i++) {
        lo = PLX_MIN(lo, im.px[i]);
        hi = PLX_MAX(hi, im.px[i]);
    }
    CHECK(lo >= 0.0f && hi <= 1.0f && hi > lo);
    plx_img_free(&im);
}

TEST(jpeg_matches_pil_on_lfw) {
    for (size_t i = 0; i < sizeof k_lfw_ref / sizeof k_lfw_ref[0]; i++) {
        uint8_t *data;
        size_t len;
        if (plx_read_file(k_lfw_ref[i].path, &data, &len) < 0) {
            printf("  (skipping %s: not found)\n", k_lfw_ref[i].path);
            continue;
        }
        int w, h, c;
        CHECK(decode_crc(data, len, &w, &h, &c) == k_lfw_ref[i].crc);
        CHECK(w == 250 && h == 250 && c == 3);
        free(data);
    }
}

/* Copies an embedded file with one byte replaced. */
static uint8_t *patched(const uint8_t *src, size_t len, size_t at, uint8_t v) {
    uint8_t *p = plx_malloc(len);
    memcpy(p, src, len);
    p[at] = v;
    return p;
}

static size_t find_marker(const uint8_t *d, size_t len, uint8_t m, int nth) {
    for (size_t i = 0; i + 1 < len; i++)
        if (d[i] == 0xFF && d[i + 1] == m && nth-- == 0) return i;
    return len;
}

TEST(jpeg_rejects_bad_and_unsupported_input) {
    int w, h, c;
    uint8_t *px = (uint8_t *)1;
    CHECK(plx_jpeg_decode_u8(NULL, 0, &w, &h, &c, &px) == PLX_EFORMAT && px == NULL);
    CHECK(plx_jpeg_decode_u8((const uint8_t *)"\xFF\xD8\xFF\xD9", 4, &w, &h, &c, &px) == PLX_EFORMAT);
    CHECK(plx_jpeg_info((const uint8_t *)"GIF89a", 6, &w, &h, &c) == PLX_EFORMAT);

    size_t len = sizeof k_422_opt_rst, sof = find_marker(k_422_opt_rst, len, 0xC0, 0);
    PLX_ASSERT(sof < len);
    uint8_t *p = patched(k_422_opt_rst, len, sof + 1, 0xC9); /* arithmetic coding */
    CHECK(plx_jpeg_decode_u8(p, len, &w, &h, &c, &px) == PLX_EUNSUPPORTED);
    free(p);
    p = patched(k_422_opt_rst, len, sof + 4, 12); /* 12-bit precision */
    CHECK(plx_jpeg_decode_u8(p, len, &w, &h, &c, &px) == PLX_EUNSUPPORTED);
    free(p);
    size_t dht = find_marker(k_422_opt_rst, len, 0xC4, 0);
    PLX_ASSERT(dht < len);
    p = patched(k_422_opt_rst, len, dht + 5, 3); /* three 1-bit codes: over-subscribed table */
    CHECK(plx_jpeg_decode_u8(p, len, &w, &h, &c, &px) == PLX_EFORMAT);
    free(p);

    /* A 4-component (CMYK-style) frame header. */
    static const uint8_t cmyk[] = {0xFF, 0xD8, 0xFF, 0xC0, 0, 20, 8, 0, 8, 0, 8, 4,
                                   1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0, 4, 0x11, 0, 0xFF, 0xD9};
    CHECK(plx_jpeg_info(cmyk, sizeof cmyk, &w, &h, &c) == PLX_EUNSUPPORTED);
    CHECK(plx_jpeg_decode_u8(cmyk, sizeof cmyk, &w, &h, &c, &px) == PLX_EUNSUPPORTED);

    /* Truncation inside entropy data is an error, never a silently zero-filled image. */
    CHECK(plx_jpeg_decode_u8(k_422_opt_rst, len - 40, &w, &h, &c, &px) == PLX_EFORMAT && px == NULL);
    /* ...but a progressive file that stops cleanly between scans decodes what it has. */
    size_t second_scan = find_marker(k_prog_rst, sizeof k_prog_rst, 0xDA, 1);
    PLX_ASSERT(second_scan < sizeof k_prog_rst);
    CHECK(plx_jpeg_decode_u8(k_prog_rst, second_scan, &w, &h, &c, &px) == PLX_OK && w == 33);
    free(px);
}

/* ---- mutation fuzzing: every outcome must be a clean status, under ASan/UBSan too -------- */
typedef struct { int ok, eformat, eunsupported, other; } fuzz_stats;

static void fuzz_one(const uint8_t *data, size_t len, fuzz_stats *fs) {
    int w, h, c;
    uint8_t *px;
    plx_jpeg_info(data, len, &w, &h, &c);
    int st = plx_jpeg_decode_u8(data, len, &w, &h, &c, &px);
    if (st == PLX_OK) {
        CHECK(px && w > 0 && h > 0 && (c == 1 || c == 3));
        uint32_t sum = 0;
        for (size_t i = 0; i < (size_t)w * h * c; i++) sum += px[i]; /* ASan checks the extent */
        (void)sum;
        free(px);
        fs->ok++;
    } else {
        CHECK(px == NULL);
        if (st == PLX_EFORMAT) fs->eformat++;
        else if (st == PLX_EUNSUPPORTED) fs->eunsupported++;
        else fs->other++;
    }
}

/* Offsets of segment length fields (the two bytes after FF xx for markers that carry one). */
static int length_fields(const uint8_t *d, size_t len, size_t *off, int cap) {
    int n = 0;
    for (size_t i = 2; i + 3 < len && n < cap; i++) {
        if (d[i] != 0xFF) continue;
        uint8_t m = d[i + 1];
        if (m == 0 || m == 0xFF || (m >= 0xD0 && m <= 0xD9) || m == 0x01) continue;
        off[n++] = i + 2;
        if (m == 0xDA) break; /* entropy data follows; FF there is stuffed */
        i += 1 + ((size_t)d[i + 2] << 8 | d[i + 3]);
    }
    return n;
}

static size_t mutate(plx_rng *r, const uint8_t *src, size_t len, uint8_t *dst) {
    memcpy(dst, src, len);
    size_t n = len;
    switch (plx_rng_below(r, 7)) {
    case 0: /* bit flips */
        for (int k = 1 + (int)plx_rng_below(r, 4); k > 0; k--) dst[plx_rng_below(r, (uint32_t)n)] ^= (uint8_t)(1u << plx_rng_below(r, 8));
        break;
    case 1: /* random bytes anywhere */
        for (int k = 1 + (int)plx_rng_below(r, 8); k > 0; k--) dst[plx_rng_below(r, (uint32_t)n)] = (uint8_t)plx_rng_u64(r);
        break;
    case 2: /* random bytes in the headers, where parsing decisions are made */
        for (int k = 1 + (int)plx_rng_below(r, 3); k > 0; k--) dst[plx_rng_below(r, (uint32_t)PLX_MIN(n, (size_t)700))] = (uint8_t)plx_rng_u64(r);
        break;
    case 3: /* truncation */
        n = plx_rng_below(r, (uint32_t)len);
        break;
    case 4: { /* marker length corruption */
        size_t off[64];
        int nf = length_fields(dst, n, off, 64);
        if (nf == 0) break;
        size_t o = off[plx_rng_below(r, (uint32_t)nf)];
        int seglen = dst[o] << 8 | dst[o + 1];
        uint32_t kind = plx_rng_below(r, 3);
        int v = kind == 0 ? (int)plx_rng_below(r, 65536) : kind == 1 ? seglen + (int)plx_rng_below(r, 9) - 4 : (int)plx_rng_below(r, 4);
        dst[o] = (uint8_t)(v >> 8);
        dst[o + 1] = (uint8_t)v;
        break;
    }
    case 5: { /* stray markers inside the data: RSTn, EOI, SOS, DHT */
        static const uint8_t ms[] = {0xD0, 0xD3, 0xD9, 0xDA, 0xC4, 0xDD, 0x00, 0xFF};
        size_t at = plx_rng_below(r, (uint32_t)(n - 1));
        dst[at] = 0xFF;
        dst[at + 1] = ms[plx_rng_below(r, sizeof ms)];
        break;
    }
    default: { /* cut a random range out */
        size_t a = plx_rng_below(r, (uint32_t)n), b = a + plx_rng_below(r, (uint32_t)PLX_MIN(n - a, (size_t)200));
        memmove(dst + a, dst + b, n - b);
        n -= b - a;
        break;
    }
    }
    return n;
}

TEST(jpeg_survives_mutations) {
    const uint8_t *files[32];
    size_t lens[32];
    uint8_t *owned[32];
    int nf = 0, nown = 0;
    files[nf] = k_prog_rst, lens[nf++] = sizeof k_prog_rst;
    files[nf] = k_gray_prog, lens[nf++] = sizeof k_gray_prog;
    files[nf] = k_422_opt_rst, lens[nf++] = sizeof k_422_opt_rst;
    for (size_t i = 0; i < sizeof k_fuzz_corpus / sizeof k_fuzz_corpus[0]; i++) {
        uint8_t *d;
        size_t l;
        if (plx_read_file(k_fuzz_corpus[i], &d, &l) < 0) continue;
        owned[nown++] = d;
        files[nf] = d, lens[nf++] = l;
    }
    /* Defaults keep the unit test quick; raise them for long campaigns. */
    const char *env = getenv("PLX_FUZZ_ITERS"), *seed = getenv("PLX_FUZZ_SEED");
    int iters = env ? atoi(env) : 3000;
    plx_rng r;
    plx_rng_seed(&r, seed ? strtoull(seed, NULL, 10) : 20240611);
    fuzz_stats fs = {0};
    uint8_t *buf = plx_malloc(64 * 1024);
    for (int it = 0; it < iters; it++) {
        int f = (int)plx_rng_below(&r, (uint32_t)nf);
        PLX_ASSERT(lens[f] <= 64 * 1024);
        size_t n = mutate(&r, files[f], lens[f], buf);
        /* a copy of exactly n bytes, so ASan flags any read past the end of the input */
        uint8_t *exact = plx_malloc(n ? n : 1);
        memcpy(exact, buf, n);
        fuzz_one(exact, n, &fs);
        free(exact);
    }
    printf("  %d mutations of %d files: %d decoded, %d EFORMAT, %d EUNSUPPORTED, %d other\n", iters, nf, fs.ok,
           fs.eformat, fs.eunsupported, fs.other);
    CHECK(fs.other == 0);
    CHECK(fs.ok + fs.eformat + fs.eunsupported == iters);
    free(buf);
    for (int i = 0; i < nown; i++) free(owned[i]);
}

/* ---- a reference inflater and PNG reader, test-only, to check the encoder end to end ------ */
typedef struct {
    uint8_t *p;
    size_t n, cap;
} vec;

static void vec_push(vec *v, uint8_t b) {
    if (v->n == v->cap) {
        v->cap = v->cap ? 2 * v->cap : 1024;
        v->p = plx_realloc(v->p, v->cap);
    }
    v->p[v->n++] = b;
}

typedef struct {
    const uint8_t *p;
    size_t n, pos;
    uint32_t buf;
    int nb;
    bool err;
} inbits;

static int ib_bits(inbits *s, int n) {
    while (s->nb < n) {
        if (s->pos >= s->n) {
            s->err = true;
            return 0;
        }
        s->buf |= (uint32_t)s->p[s->pos++] << s->nb;
        s->nb += 8;
    }
    int v = (int)(s->buf & ((1u << n) - 1));
    s->buf >>= n;
    s->nb -= n;
    return v;
}

typedef struct { int count[16], sym[320]; } ihuff;

/* Like zlib, refuses over-subscribed and incomplete codes (a lone 1-bit code excepted). */
static bool ih_build(ihuff *h, const uint8_t *len, int n) {
    memset(h->count, 0, sizeof h->count);
    for (int i = 0; i < n; i++) h->count[len[i]]++;
    int left = 1, offs[16] = {0};
    for (int l = 1; l < 16; l++) {
        left = 2 * left - h->count[l];
        if (left < 0) return false;
    }
    for (int l = 1; l < 15; l++) offs[l + 1] = offs[l] + h->count[l];
    for (int i = 0; i < n; i++)
        if (len[i]) h->sym[offs[len[i]]++] = i;
    return left == 0 || (n - h->count[0] == 1 && h->count[1] == 1);
}

static int ih_decode(inbits *s, const ihuff *h) {
    int code = 0, first = 0, index = 0;
    for (int l = 1; l < 16; l++) {
        code |= ib_bits(s, 1);
        int count = h->count[l];
        if (code - count < first) return h->sym[index + (code - first)];
        index += count;
        first = (first + count) << 1;
        code <<= 1;
    }
    return -1;
}

static const int t_lbase[29] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
                                35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
static const int t_lext[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
static const int t_dbase[30] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769,
                                1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
static const int t_dext[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

static bool inflate_codes(inbits *s, vec *out, const ihuff *lit, const ihuff *dist) {
    for (;;) {
        int sym = ih_decode(s, lit);
        if (sym < 0 || s->err) return false;
        if (sym < 256) {
            vec_push(out, (uint8_t)sym);
        } else if (sym == 256) {
            return true;
        } else {
            sym -= 257;
            if (sym >= 29) return false;
            int len = t_lbase[sym] + ib_bits(s, t_lext[sym]);
            int ds = ih_decode(s, dist);
            if (ds < 0 || ds >= 30) return false;
            size_t d = (size_t)(t_dbase[ds] + ib_bits(s, t_dext[ds]));
            if (s->err || d > out->n) return false;
            for (int k = 0; k < len; k++) vec_push(out, out->p[out->n - d]);
        }
    }
}

static bool zlib_inflate(const uint8_t *z, size_t n, vec *out) {
    if (n < 6 || (z[0] & 15) != 8 || (z[0] << 8 | z[1]) % 31 != 0 || (z[1] & 0x20)) return false;
    inbits s = {z, n - 4, 2, 0, 0, false};
    ihuff lit, dist;
    int final;
    do {
        final = ib_bits(&s, 1);
        int type = ib_bits(&s, 2);
        if (type == 0) {
            s.buf = 0;
            s.nb = 0;
            if (s.pos + 4 > s.n) return false;
            unsigned len = s.p[s.pos] | s.p[s.pos + 1] << 8, nlen = s.p[s.pos + 2] | s.p[s.pos + 3] << 8;
            s.pos += 4;
            if (len != (~nlen & 0xFFFF) || s.pos + len > s.n) return false;
            for (unsigned k = 0; k < len; k++) vec_push(out, s.p[s.pos++]);
        } else if (type == 1) {
            uint8_t l[288], d[32];
            for (int i = 0; i < 288; i++) l[i] = i < 144 ? 8 : i < 256 ? 9 : i < 280 ? 7 : 8;
            memset(d, 5, sizeof d);
            ih_build(&lit, l, 288);
            ih_build(&dist, d, 32);
            if (!inflate_codes(&s, out, &lit, &dist)) return false;
        } else if (type == 2) {
            static const int order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
            int hlit = ib_bits(&s, 5) + 257, hdist = ib_bits(&s, 5) + 1, hclen = ib_bits(&s, 4) + 4;
            if (hlit > 286 || hdist > 30) return false;
            uint8_t cl[19] = {0}, lens[316];
            for (int i = 0; i < hclen; i++) cl[order[i]] = (uint8_t)ib_bits(&s, 3);
            ihuff clh;
            if (!ih_build(&clh, cl, 19)) return false;
            for (int i = 0; i < hlit + hdist;) {
                int sym = ih_decode(&s, &clh), rep = 0, val = 0;
                if (sym < 0) return false;
                if (sym < 16) { lens[i++] = (uint8_t)sym; continue; }
                if (sym == 16) { if (i == 0) return false; val = lens[i - 1]; rep = 3 + ib_bits(&s, 2); }
                else if (sym == 17) rep = 3 + ib_bits(&s, 3);
                else rep = 11 + ib_bits(&s, 7);
                if (i + rep > hlit + hdist) return false;
                while (rep--) lens[i++] = (uint8_t)val;
            }
            if (!lens[256] || !ih_build(&lit, lens, hlit) || !ih_build(&dist, lens + hlit, hdist)) return false;
            if (!inflate_codes(&s, out, &lit, &dist)) return false;
        } else {
            return false;
        }
        if (s.err) return false;
    } while (!final);
    if (s.pos != n - 4) return false;
    uint32_t adler = (uint32_t)z[n - 4] << 24 | (uint32_t)z[n - 3] << 16 | (uint32_t)z[n - 2] << 8 | z[n - 1];
    return adler == plx_adler32(1, out->p, out->n);
}

static uint32_t be32(const uint8_t *p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }

/* Reads back a PNG this module wrote: rows unfiltered, samples as stored (16-bit big-endian). */
static uint8_t *png_read(const char *path, int *w, int *h, int *depth, int *ctype, size_t *idat_chunks) {
    uint8_t *f;
    size_t len;
    if (plx_read_file(path, &f, &len) < 0) return NULL;
    uint8_t *raw = NULL;
    vec z = {0}, out = {0};
    bool ok = len >= 8 && memcmp(f, "\x89PNG\r\n\x1A\n", 8) == 0, ended = false;
    *idat_chunks = 0;
    for (size_t p = 8; ok && !ended;) {
        if (p + 12 > len) { ok = false; break; }
        uint32_t n = be32(f + p);
        if (p + 12 + n > len || plx_crc32(0, f + p + 4, n + 4) != be32(f + p + 8 + n)) { ok = false; break; }
        const uint8_t *type = f + p + 4, *d = f + p + 8;
        if (memcmp(type, "IHDR", 4) == 0) {
            *w = (int)be32(d);
            *h = (int)be32(d + 4);
            *depth = d[8];
            *ctype = d[9];
        } else if (memcmp(type, "IDAT", 4) == 0) {
            for (uint32_t k = 0; k < n; k++) vec_push(&z, d[k]);
            (*idat_chunks)++;
        } else if (memcmp(type, "IEND", 4) == 0) {
            ended = true;
        }
        p += 12 + n;
    }
    static const int chans[7] = {1, 0, 3, 0, 2, 0, 4};
    if (ok && ended && zlib_inflate(z.p, z.n, &out)) {
        int bpp = chans[*ctype] * *depth / 8;
        size_t rb = (size_t)*w * bpp;
        if (out.n == (rb + 1) * *h) {
            raw = plx_malloc(rb * *h);
            for (int y = 0; y < *h; y++) {
                const uint8_t *src = out.p + y * (rb + 1) + 1;
                uint8_t *cur = raw + y * rb, *up = y ? cur - rb : NULL;
                int ft = src[-1];
                for (size_t x = 0; x < rb; x++) {
                    int a = x >= (size_t)bpp ? cur[x - bpp] : 0, b = up ? up[x] : 0;
                    int c = up && x >= (size_t)bpp ? up[x - bpp] : 0, pred = 0;
                    if (ft == 1) pred = a;
                    else if (ft == 2) pred = b;
                    else if (ft == 3) pred = (a + b) / 2;
                    else if (ft == 4) {
                        int pp = a + b - c, pa = abs(pp - a), pb = abs(pp - b), pc = abs(pp - c);
                        pred = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
                    }
                    cur[x] = (uint8_t)(src[x] + pred);
                }
            }
        }
    }
    free(f);
    free(z.p);
    free(out.p);
    return raw;
}

TEST(checksums_match_known_vectors) {
    CHECK(plx_crc32(0, "123456789", 9) == 0xCBF43926u);
    CHECK(plx_crc32(plx_crc32(0, "1234", 4), "56789", 5) == 0xCBF43926u);
    CHECK(plx_adler32(1, "Wikipedia", 9) == 0x11E60398u);
    CHECK(plx_crc32(0, "", 0) == 0 && plx_adler32(1, "", 0) == 1);
    uint8_t big[20000];
    memset(big, 0xFF, sizeof big); /* long enough to exercise the deferred modulo */
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < sizeof big; i++) {
        a = (a + big[i]) % 65521;
        b = (b + a) % 65521;
    }
    CHECK(plx_adler32(1, big, sizeof big) == (b << 16 | a));
}

static bool zlib_roundtrip(const uint8_t *src, size_t n, size_t *zlen) {
    uint8_t *z;
    *zlen = plx_zlib_compress(src, n, &z);
    vec out = {0};
    bool ok = zlib_inflate(z, *zlen, &out) && out.n == n && (n == 0 || memcmp(out.p, src, n) == 0);
    free(z);
    free(out.p);
    return ok;
}

TEST(deflate_roundtrips) {
    plx_rng r;
    plx_rng_seed(&r, 7);
    size_t n = 300000, zl;
    uint8_t *buf = plx_malloc(n);
    CHECK(zlib_roundtrip(buf, 0, &zl));
    buf[0] = 42;
    CHECK(zlib_roundtrip(buf, 1, &zl));

    for (size_t i = 0; i < n; i++) buf[i] = (uint8_t)plx_rng_u64(&r); /* incompressible: stored blocks */
    CHECK(zlib_roundtrip(buf, n, &zl) && zl < n + n / 1000 + 64);

    memset(buf, 'a', n); /* one long run */
    CHECK(zlib_roundtrip(buf, n, &zl) && zl < n / 100);

    static const char *words[] = {"face ", "recognition ", "parallax ", "eigen ", "\n", "kernel ", "the "};
    for (size_t i = 0; i < n;) /* text-like: random words, ~2.8 bits of entropy per word */
        for (const char *w = words[plx_rng_below(&r, 7)]; *w && i < n; w++) buf[i++] = (uint8_t)*w;
    CHECK(zlib_roundtrip(buf, n, &zl) && zl < n / 8);

    /* Fibonacci-skewed symbol frequencies force code lengths past 15 bits without the limit. */
    size_t k = 0, fib[30] = {1, 1};
    for (int i = 2; i < 30; i++) fib[i] = fib[i - 1] + fib[i - 2];
    for (int s = 0; s < 27 && k < n; s++)
        for (size_t j = 0; j < fib[s] && k < n; j++) buf[k++] = (uint8_t)(s * 9);
    for (size_t i = k; i > 1; i--) { /* shuffle so LZ77 cannot hide the skew */
        size_t j = plx_rng_below(&r, (uint32_t)i), t = buf[i - 1];
        buf[i - 1] = buf[j];
        buf[j] = (uint8_t)t;
    }
    CHECK(zlib_roundtrip(buf, k, &zl));

    for (size_t i = 0; i < n; i++) buf[i] = (uint8_t)(i % 251 ^ (i / 997)); /* periodic, long distances */
    CHECK(zlib_roundtrip(buf, n, &zl));
    free(buf);
}

static bool png_check_u8(int w, int h, int c, uint32_t seed, size_t *idat_chunks) {
    plx_rng r;
    plx_rng_seed(&r, seed);
    size_t n = (size_t)w * h * c;
    uint8_t *px = plx_malloc(n);
    for (int y = 0; y < h; y++) /* smooth gradient plus noise, so filters and matches both matter */
        for (int x = 0; x < w; x++)
            for (int k = 0; k < c; k++)
                px[((size_t)y * w + x) * c + k] = (uint8_t)(x * 3 + y * 5 + k * 40 + (int)plx_rng_below(&r, 6));
    char path[256];
    strcpy(path, tmp_path("t.png"));
    bool ok = plx_png_write_u8(path, px, w, h, c) == PLX_OK;
    int rw, rh, depth, ctype;
    uint8_t *back = ok ? png_read(path, &rw, &rh, &depth, &ctype, idat_chunks) : NULL;
    static const int want_ctype[5] = {0, 0, 4, 2, 6};
    ok = back && rw == w && rh == h && depth == 8 && ctype == want_ctype[c] && memcmp(back, px, n) == 0;
    free(back);
    free(px);
    unlink(path);
    return ok;
}

TEST(png_roundtrips) {
    size_t chunks;
    CHECK(png_check_u8(1, 1, 1, 1, &chunks));
    CHECK(png_check_u8(1, 1, 4, 2, &chunks));
    CHECK(png_check_u8(3, 7, 3, 3, &chunks));
    CHECK(png_check_u8(250, 250, 3, 4, &chunks));
    CHECK(png_check_u8(64, 33, 2, 5, &chunks));
    CHECK(png_check_u8(70001, 2, 1, 6, &chunks)); /* rows longer than a stored block */
    CHECK(png_check_u8(30000, 9, 4, 7, &chunks) && chunks > 1); /* 120 KB rows, several IDATs */
    CHECK(plx_png_write_u8(tmp_path("bad.png"), (const uint8_t *)"x", 1, 1, 5) == PLX_EINVAL);

    int w = 300, h = 41;
    uint16_t *g = plx_malloc(sizeof *g * w * h);
    for (int i = 0; i < w * h; i++) g[i] = (uint16_t)(i * 2654435761u >> 16);
    char path[256];
    strcpy(path, tmp_path("g16.png"));
    CHECK(plx_png_write_gray16(path, g, w, h) == PLX_OK);
    int rw, rh, depth, ctype;
    uint8_t *back = png_read(path, &rw, &rh, &depth, &ctype, &chunks);
    CHECK(back && rw == w && rh == h && depth == 16 && ctype == 0);
    bool same = back != NULL;
    for (int i = 0; same && i < w * h; i++) same = (back[2 * i] << 8 | back[2 * i + 1]) == g[i];
    CHECK(same);
    free(back);
    free(g);
    unlink(path);
}

TEST(pnm_headers_and_roundtrips) {
    static const uint8_t pgm[] = "P5 # comment\n#another\n 3\t2\r\n# mid\n255\n\x00\x10\x20\x30\x40\xFF";
    plx_img im;
    CHECK(plx_pnm_decode(pgm, sizeof pgm - 1, &im) == PLX_OK);
    CHECK(im.w == 3 && im.h == 2 && im.c == 1);
    CHECK_NEAR(PLX_PX(&im, 1, 0, 0), 16.0 / 255, 1e-7);
    CHECK_NEAR(PLX_PX(&im, 2, 1, 0), 1.0, 1e-7);
    plx_img_free(&im);

    static const uint8_t ppm16[] = "P6\n1 1\n1000\n\x01\xF4\x03\xE8\x00\x00";
    CHECK(plx_pnm_decode(ppm16, sizeof ppm16 - 1, &im) == PLX_OK);
    CHECK(im.c == 3 && fabs(PLX_PX(&im, 0, 0, 0) - 0.5) < 1e-7 && PLX_PX(&im, 0, 0, 1) == 1.0f);
    plx_img_free(&im);

    static const char *bad[] = {"P5\n3 2\n255\n\x01", "P5\n3\n", "P5 0 2 255\n", "P7\n1 1\n255\n\x01",
                                "P5\n99999999999 1\n255\n", "P5\n1 1\n256x\n\x00", "P5\n1 1\n255", "Pf\n1 1\n0\nabcd"};
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) CHECK(plx_pnm_decode((const uint8_t *)bad[i], strlen(bad[i]), &im) < 0);

    /* Big-endian PFM (positive scale), stored bottom row first. */
    static const uint8_t pfm_be[] = "Pf\n1 2\n1.0\n\x3F\x80\x00\x00\x40\x00\x00\x00";
    CHECK(plx_pnm_decode(pfm_be, sizeof pfm_be - 1, &im) == PLX_OK);
    CHECK(im.w == 1 && im.h == 2 && PLX_PX(&im, 0, 0, 0) == 2.0f && PLX_PX(&im, 0, 1, 0) == 1.0f);
    plx_img_free(&im);

    uint8_t px[5 * 4 * 3];
    for (int i = 0; i < 60; i++) px[i] = (uint8_t)(i * 37);
    char path[256];
    strcpy(path, tmp_path("t.ppm"));
    CHECK(plx_pnm_write_u8(path, px, 5, 4, 3) == PLX_OK);
    CHECK(plx_pnm_read(path, &im) == PLX_OK && im.w == 5 && im.h == 4 && im.c == 3);
    uint8_t back[60];
    plx_img_to_u8(&im, back, 1.0f / 255.0f);
    CHECK(memcmp(back, px, 60) == 0);
    plx_img_free(&im);
    unlink(path);

    plx_img f = plx_img_new(3, 2, 3);
    for (int i = 0; i < 18; i++) f.px[i] = (float)i * -1.25f + 0.1f;
    strcpy(path, tmp_path("t.pfm"));
    CHECK(plx_pfm_write(path, &f) == PLX_OK);
    uint8_t *raw;
    size_t len;
    CHECK(plx_read_file(path, &raw, &len) == PLX_OK && len == 12 + 18 * 4);
    float first;
    memcpy(&first, raw + 12, 4); /* first stored sample is the bottom-left red value */
    CHECK(first == PLX_PX(&f, 0, 1, 0));
    free(raw);
    CHECK(plx_pnm_read(path, &im) == PLX_OK && memcmp(im.px, f.px, sizeof(float) * 18) == 0);
    plx_img_free(&im);
    plx_img_free(&f);
    unlink(path);
}

TEST(imgio_dispatches_by_magic_and_extension) {
    plx_img im = plx_img_new(4, 3, 3);
    for (int i = 0; i < 36; i++) im.px[i] = (float)i / 35.0f;
    im.px[0] = -0.5f; /* clamps to 0 in 8-bit formats */
    im.px[1] = 7.0f;  /* clamps to 1 */
    char path[256];

    strcpy(path, tmp_path("x.PPM"));
    CHECK(plx_img_save(path, &im) == PLX_OK);
    plx_img back;
    CHECK(plx_img_load(path, &back) == PLX_OK && back.c == 3 && back.px[0] == 0.0f && back.px[1] == 1.0f);
    CHECK(fabs(back.px[20] - im.px[20]) <= 0.5 / 255 + 1e-6);
    plx_img_free(&back);
    unlink(path);

    strcpy(path, tmp_path("x.pfm"));
    CHECK(plx_img_save(path, &im) == PLX_OK);
    CHECK(plx_img_load(path, &back) == PLX_OK && memcmp(back.px, im.px, sizeof(float) * 36) == 0);
    plx_img_free(&back);
    unlink(path);

    strcpy(path, tmp_path("x.png"));
    CHECK(plx_img_save(path, &im) == PLX_OK);
    CHECK(plx_img_load(path, &back) == PLX_EUNSUPPORTED); /* write-only format */
    unlink(path);

    CHECK(plx_img_save(tmp_path("x.pgm"), &im) == PLX_EINVAL);
    CHECK(plx_img_save(tmp_path("x.tiff"), &im) == PLX_EUNSUPPORTED);
    CHECK(plx_img_save(tmp_path("noext"), &im) == PLX_EUNSUPPORTED);
    plx_img_free(&im);

    plx_img j;
    strcpy(path, tmp_path("x.jpg"));
    CHECK(plx_write_file(path, k_422_opt_rst, sizeof k_422_opt_rst) == PLX_OK);
    CHECK(plx_img_load(path, &j) == PLX_OK && j.w == 24 && j.h == 17 && j.c == 3);
    plx_img_free(&j);
    unlink(path);
    CHECK(plx_img_load(tmp_path("missing.jpg"), &j) == PLX_EIO);
}

int main(void) {
    RUN(jpeg_matches_pil_on_embedded_files);
    RUN(jpeg_matches_pil_on_lfw);
    RUN(jpeg_rejects_bad_and_unsupported_input);
    RUN(jpeg_survives_mutations);
    RUN(checksums_match_known_vectors);
    RUN(deflate_roundtrips);
    RUN(png_roundtrips);
    RUN(pnm_headers_and_roundtrips);
    RUN(imgio_dispatches_by_magic_and_extension);
    return TEST_REPORT();
}
