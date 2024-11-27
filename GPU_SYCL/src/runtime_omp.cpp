#include <mkl.h>
#include <omp.h>

#include "dgetrf.h"
#include "matrix.h"
#include "runtime.h"

runtime_omp::runtime_omp(std::string type, int N, int bs, int ibs)
    : runtime(type, N, bs, ibs) {
    int num_blocks = N / block_size;
    deps = std::vector<char>(num_blocks * num_blocks);
}

void runtime_omp::dgetrf(std::vector<double> &A) {
    if (type == "default") {
        dgetrf_nopiv_block_omptask(A);
    } else {
        std::cerr << "Invalid LU factorization type: " << type << std::endl;
        exit(EXIT_FAILURE);
    }
}

void runtime_omp::dgetrf_nopiv_block_omptask(std::vector<double> &A) {
    int num_blocks = N / block_size;
    int lda = N;

#pragma omp parallel
    {
#pragma omp single
        {
            for (int k = 0; k < num_blocks; ++k) {

                // step 1 : facto in place
#pragma omp task default(shared) firstprivate(k) \
    depend(inout : deps[IDX(k, k, num_blocks)])
                {
                    dgetrf_nopiv_ib(block_size, block_size, innerblock_size,
                                    &A[IDX(k * block_size, k * block_size, lda)], lda);
                }
                // step 2:  TRSM for L
                for (int i = k + 1; i < num_blocks; ++i) {
#pragma omp task default(shared) firstprivate(k, i) \
    depend(in : deps[IDX(k, k, num_blocks)]) depend(inout : deps[IDX(i, k, num_blocks)])
                    cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                                CblasNonUnit, block_size, block_size, 1.0,
                                &A[IDX(k * block_size, k * block_size, lda)], lda,
                                &A[IDX(i * block_size, k * block_size, lda)], lda);
                }

                // step 3: TRSM for U
                for (int j = k + 1; j < num_blocks; ++j) {
#pragma omp task default(shared) firstprivate(k, j) \
    depend(in : deps[IDX(k, k, num_blocks)]) depend(inout : deps[IDX(k, j, num_blocks)])
                    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                                CblasUnit, block_size, block_size, 1.0,
                                &A[IDX(k * block_size, k * block_size, lda)], lda,
                                &A[IDX(k * block_size, j * block_size, lda)], lda);
                }

                // step 4: GEMM for schur
                for (int i = k + 1; i < num_blocks; ++i) {
                    for (int j = k + 1; j < num_blocks; ++j) {
#pragma omp task default(shared) firstprivate(k, i, j) \
    depend(in : deps[IDX(i, k, num_blocks)], deps[IDX(k, j, num_blocks)]) \
    depend(inout : deps[IDX(i, j, num_blocks)])
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, block_size,
                                    block_size, block_size, -1.0,
                                    &A[IDX(i * block_size, k * block_size, lda)], lda,
                                    &A[IDX(k * block_size, j * block_size, lda)], lda,
                                    1.0, &A[IDX(i * block_size, j * block_size, lda)],
                                    lda);
                    }
                }
            }
        }
    }
}
