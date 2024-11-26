#include "../include/dgetrf_util.h"
#include "../include/matrix.h"

#include <cstdlib>
#include <iostream>
#include <math.h>
#include <mkl.h>
#include <vector>

// source: chameleon flops.h
double dgetrf_fmuls(double n) {
    return 0.5 * n * (n * (n - (1. / 3.) * n - 1.) + n) + (2. / 3.) * n;
}

double dgetrf_fadds(double n) {
    return 0.5 * n * (n * (n - (1. / 3.) * n) - n) + (1. / 6.) * n;
}

double dgetrf_flops(double n) {
    double flops = dgetrf_fmuls(n) + dgetrf_fadds(n);
    return flops;
}

bool dgetrf_check(int N, std::vector<double> &A_facto, unsigned seed, double &err) {
    int lda = N;

    std::vector<double> L(N * N);
    std::vector<double> U(N * N);

    LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', N, N, A_facto.data(), lda, U.data(), lda);
    LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'L', N, N, A_facto.data(), lda, L.data(), lda);
    LAPACKE_dlaset(LAPACK_COL_MAJOR, 'U', N, N, 0., 1., L.data(), lda);

    auto &A0 = A_facto;
    matrix_init_sym_diag_dom(N, A0, seed);

    // norm(residu) = ||LU - A0||
    double A0norm, Rnorm;
    double eps = LAPACKE_dlamch('e');
    double result;

    A0norm = LAPACKE_dlange(LAPACK_COL_MAJOR, '1', N, N, A0.data(), N);

    // A = -L*U + A
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, -1.0, L.data(), lda,
                U.data(), lda, 1.0, A0.data(), lda);

    // rnorm = A-L*U
    Rnorm = LAPACKE_dlange(LAPACK_COL_MAJOR, '1', N, N, A0.data(), N);

    result = Rnorm / (A0norm * N * eps);

    bool check = true;
    if (std::isnan(result) || std::isinf(result) || (result == 0.0 || result > 10.0)) {
        check = false;
    }
    err = result;
    return check;
}
