#include <mkl.h>

#include "dgetrf.h"
#include "matrix.h"
#include "runtime.h"

runtime_seq::runtime_seq(std::string type, int N, int bs, int ibs)
    : runtime(type, N, bs, ibs) {}

void runtime_seq::dgetrf(std::vector<double> &A) {
    if (type == "default" || type == "rectangular") {
        dgetrf_nopiv_rectangular(A);
    } else if (type == "scalar") {
        dgetrf_nopiv_scalar(A);
    } else if (type == "block") {
        dgetrf_nopiv_block(A);
    } else {
        std::cerr << "Invalid LU factorization type: " << type << std::endl;
        exit(EXIT_FAILURE);
    }
}

void runtime_seq::dgetrf_nopiv_scalar(std::vector<double> &A) {
    dgetrf_nopiv_ld(N, N, A.data());
}

void runtime_seq::dgetrf_nopiv_block(std::vector<double> &A) {
    int num_blocks = N / block_size;
    int lda = N;
    for (int k = 0; k < num_blocks; ++k) {

        // step 1 : facto in place
        dgetrf_nopiv_ib(block_size, block_size, innerblock_size,
                        &A[IDX(k * block_size, k * block_size, lda)], lda);

        // step 2:  TRSM for L
        for (int i = k + 1; i < num_blocks; ++i) {
            cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                        block_size, block_size, 1.0,
                        &A[IDX(k * block_size, k * block_size, lda)], lda,
                        &A[IDX(i * block_size, k * block_size, lda)], lda);
        }

        // step 3: TRSM for U
        for (int j = k + 1; j < num_blocks; ++j) {
            cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                        block_size, block_size, 1.0,
                        &A[IDX(k * block_size, k * block_size, lda)], lda,
                        &A[IDX(k * block_size, j * block_size, lda)], lda);
        }

        // step 4: GEMM for schur
        for (int i = k + 1; i < num_blocks; ++i) {
            for (int j = k + 1; j < num_blocks; ++j) {
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, block_size,
                            block_size, block_size, -1.0,
                            &A[IDX(i * block_size, k * block_size, lda)], lda,
                            &A[IDX(k * block_size, j * block_size, lda)], lda, 1.0,
                            &A[IDX(i * block_size, j * block_size, lda)], lda);
            }
        }
    }
}

void runtime_seq::dgetrf_nopiv_rectangular(std::vector<double> &A) {
    int lda = N;
    int num_blocks = N / block_size;
    for (int k = 0; k < num_blocks; ++k) {

        // step 1: facto in place
        dgetrf_nopiv_ib(block_size, block_size, innerblock_size,
                        &A[IDX(k * block_size, k * block_size, lda)], lda);

        if (k < num_blocks - 1) {

            int schur_size = N - (k + 1) * block_size;

            // step 2: TRSM for L
            cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                        schur_size, block_size, 1.0,
                        &A[IDX(k * block_size, k * block_size, lda)], lda,
                        &A[IDX((k + 1) * block_size, k * block_size, lda)], lda);

            // step 3: TRSM for U
            cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                        block_size, schur_size, 1.0,
                        &A[IDX(k * block_size, k * block_size, lda)], lda,
                        &A[IDX(k * block_size, (k + 1) * block_size, lda)], lda);

            // step 4: GEMM for schur
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, schur_size, schur_size,
                        block_size, -1.0,
                        &A[IDX((k + 1) * block_size, k * block_size, lda)], lda,
                        &A[IDX(k * block_size, (k + 1) * block_size, lda)], lda, 1.0,
                        &A[IDX((k + 1) * block_size, (k + 1) * block_size, lda)], lda);
        }
    }
}
