#include "../include/dgetrf.h"
#include "../include/dgetrf_util.h"
#include "../include/matrix.h"
#include <cstdlib>
#include <iostream>
#include <math.h>

#include "mkl.h"
#include "oneapi/mkl/blas.hpp"
#include <mkl_lapacke.h>
#include <omp.h>
#include <sycl/sycl.hpp>

void dgetrf_nopiv_ld(int N, int lda, double *A) {
    for (int k = 0; k < N; ++k) {
        double pivot = 1.0 / A[IDX(k, k, lda)];

        for (int i = k + 1; i < N; ++i) {
            // update pivot
            A[IDX(i, k, lda)] *= pivot;
        }

        for (int i = k + 1; i < N; ++i) {
            for (int j = k + 1; j < N; ++j) {
                // update Schur directly in A
                A[IDX(i, j, lda)] -= A[IDX(i, k, lda)] * A[IDX(k, j, lda)];
            }
        }
    }
}

int dgetf2_nopiv(int M, int N, double *A, int LDA) {
    double mzone = (double)-1.0;
    double alpha;
    double sfmin;
    int i, j, k;

    /* Check input arguments */
    if (M < 0) {
        fprintf(stderr, "Illegal value of M");
        return -1;
    }
    if (N < 0) {
        fprintf(stderr, "Illegal value of N");
        return -1;
    }
    if ((LDA < std::max(1, M)) && (M > 0)) {
        fprintf(stderr, "Illegal value of LDA");
        return -1;
    }

    /* Quick return */
    if ((M == 0) || (N == 0))
        return 0;

    sfmin = LAPACKE_dlamch_work('S');
    k = std::min(M, N);
    for (i = 0; i < k; i++) {
        alpha = A[IDX(i, i, LDA)];
        if (alpha != (double)0.0) {
            /* Compute elements J+1:M of J-th column. */
            if (i < M) {
                if (std::abs(alpha) > sfmin) {
                    alpha = 1.0 / alpha;
                    cblas_dscal(M - i - 1, alpha, &(A[IDX(i + 1, i, LDA)]), 1);
                } else {
                    for (j = i + 1; j < M; j++)
                        A[IDX(j, i, LDA)] = A[IDX(j, i, LDA)] / alpha;
                }
            }
        } else {
            return i;
        }

        if (i < k) {
            /* Update trailing submatrix */
            cblas_dger(CblasColMajor, M - i - 1, N - i - 1, mzone, &A[IDX(i + 1, i, LDA)],
                       1, &A[IDX(i, i + 1, LDA)], LDA, &A[IDX(i + 1, i + 1, LDA)], LDA);
        }
    }

    return 0;
}

void dgetrf_nopiv_ib(int M, int N, int IB, double *A, int LDA) {
    double zone = (double)1.0;
    double mzone = (double)-1.0;
    int i, k, sb;

    /* Check input arguments */
    if (M < 0) {
        fprintf(stderr, "Illegal value of M");
        exit(EXIT_FAILURE);
    }
    if (N < 0) {
        fprintf(stderr, "Illegal value of N");
        exit(EXIT_FAILURE);
    }
    if (IB < 0) {
        fprintf(stderr, "Illegal value of IB");
        exit(EXIT_FAILURE);
    }
    if ((LDA < std::max(1, M)) && (M > 0)) {
        fprintf(stderr, "Illegal value of LDA");
        exit(EXIT_FAILURE);
    }

    /* Quick return */
    if ((M == 0) || (N == 0) || (IB == 0))
        return;

    k = std::min(M, N);
    for (i = 0; i < k; i += IB) {
        sb = std::min(IB, k - i);
        /*
         * Factor diagonal and subdiagonal blocks and test for exact singularity.
         */
        int res = dgetf2_nopiv(M - i, sb, &A[IDX(i, i, LDA)], LDA);
        if (res < 0)
            exit(EXIT_FAILURE);

        if (i + sb < N) {
            /*  Compute block row of U */
            cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, sb,
                        N - (i + sb), zone, &A[IDX(i, i, LDA)], LDA,
                        &A[IDX(i, i + sb, LDA)], LDA);

            if (i + sb < M) {
                /* Update trailing submatrix */
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M - (i + sb),
                            N - (i + sb), sb, mzone, &A[IDX(i + sb, i, LDA)], LDA,
                            &A[IDX(i, i + sb, LDA)], LDA, zone,
                            &A[IDX(i + sb, i + sb, LDA)], LDA);
            }
        }
    }
}

sycl::event dgetrf_nopiv_sycl_buffer_ld(int N, int lda, double *A, sycl::queue &q) {

    // Buffer creation: dim=1 and size=N*N
    sycl::buffer<double, 1> A_buffer(A, sycl::range<1>(N * N));

    return q.submit([&](sycl::handler &cgh) {
        auto A = A_buffer.get_access<sycl::access::mode::read_write>(cgh);
        cgh.single_task([=]() {
            for (int k = 0; k < N; ++k) {

                // Kernel for pivot
                double pivot = 1.0 / A[IDX(k, k, lda)];
                for (int i = k + 1; i < N; ++i) {
                    A[IDX(i, k, lda)] *= pivot;
                }

                // Kernel for Schur
                for (int i = k + 1; i < N; ++i) {
                    for (int j = k + 1; j < N; ++j) {
                        A[IDX(i, j, lda)] -= A[IDX(i, k, lda)] * A[IDX(k, j, lda)];
                    }
                }
            }
        });
    });
}

void dgetrf_nopiv_sycl(int N, std::vector<double> &A) {
    sycl::queue q;
    dgetrf_nopiv_sycl_buffer_ld(N, N, A.data(), q);
    q.wait();
}
