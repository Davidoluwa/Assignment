/* Minimal test harness. One file per module: tests/test_<module>.c
 *
 *   // plx-deps: la core
 *   #include "test.h"
 *   TEST(cholesky_solves) { CHECK_NEAR(x, 1.0, 1e-12); }
 *   int main(void) { RUN(cholesky_solves); return TEST_REPORT(); }
 */
#ifndef PLX_TEST_H
#define PLX_TEST_H

#include <math.h>
#include <stdio.h>

static int plx_test_checks = 0, plx_test_fails = 0;

#define TEST(name) static void name(void)
#define RUN(name) \
    do { int before = plx_test_fails; name(); \
         printf("%s %s\n", plx_test_fails == before ? "ok  " : "FAIL", #name); } while (0)

#define CHECK(cond) \
    do { plx_test_checks++; \
         if (!(cond)) { plx_test_fails++; printf("  %s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); } } while (0)

#define CHECK_NEAR(a, b, tol) \
    do { double a_ = (double)(a), b_ = (double)(b), t_ = (double)(tol); plx_test_checks++; \
         if (!(fabs(a_ - b_) <= t_)) { plx_test_fails++; \
             printf("  %s:%d: %s = %.12g, want %.12g (tol %g)\n", __FILE__, __LINE__, #a, a_, b_, t_); } } while (0)

#define TEST_REPORT() \
    (printf("%d checks, %d failed\n", plx_test_checks, plx_test_fails), plx_test_fails ? 1 : 0)

#endif
