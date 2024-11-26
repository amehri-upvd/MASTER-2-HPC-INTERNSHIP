#include <oneapi/mkl/blas.hpp>

#include "dgetrf.h"
#include "matrix.h"
#include "runtime.h"

runtime_sycl::runtime_sycl(std::string type, int N, int bs, int ibs)
    : runtime(type, N, bs, ibs) {
    int num_blocks = N / block_size;
    deps = std::vector<sycl::event>(num_blocks * num_blocks);
    std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>()
              << q.get_device().get_info<sycl::info::device::max_compute_units>()
              << std::endl;
    Adev = sycl::malloc_device<double>(N * N, q);
}

runtime_sycl::~runtime_sycl() {
    // Free allocated memory on the device
    sycl::free(Adev, q);
}

void runtime_sycl::prefetch(std::vector<double> &A) {
    // Copy memory from host to device
    q.memcpy(static_cast<void *>(Adev), static_cast<void *>(A.data()),
             sizeof(double) * N * N)
        .wait();
}

void runtime_sycl::dgetrf(std::vector<double> &A) { dgetrf_nopiv_block_sycl(); }

void runtime_sycl::post(std::vector<double> &A) {
    // Copy memory from device back to host
    q.memcpy(static_cast<void *>(A.data()), static_cast<void *>(Adev),
             sizeof(double) * N * N)
        .wait();
}

void runtime_sycl::dgetrf_nopiv_block_sycl() {
    int num_blocks = N / block_size;
    int lda = N;

    for (int k = 0; k < num_blocks; ++k) {
        deps[IDX(k, k, num_blocks)] = dgetrf_nopiv_sycl_ld(
            block_size, N, &Adev[IDX(k * block_size, k * block_size, lda)],
            deps[IDX(k, k, num_blocks)]);

        for (int i = k + 1; i < num_blocks; ++i) {
            deps[IDX(i, k, num_blocks)] = oneapi::mkl::blas::column_major::trsm(
                q, oneapi::mkl::side::R, oneapi::mkl::uplo::U, oneapi::mkl::transpose::N,
                oneapi::mkl::diag::N, block_size, block_size, 1.0,
                &Adev[IDX(k * block_size, k * block_size, lda)], lda,
                &Adev[IDX(i * block_size, k * block_size, lda)], lda,
                {deps[IDX(k, k, num_blocks)], deps[IDX(i, k, num_blocks)]});
        }
        for (int j = k + 1; j < num_blocks; ++j) {

            deps[IDX(k, j, num_blocks)] = oneapi::mkl::blas::column_major::trsm(
                q, oneapi::mkl::side::L, oneapi::mkl::uplo::L, oneapi::mkl::transpose::N,
                oneapi::mkl::diag::U, block_size, block_size, 1.0,
                &Adev[IDX(k * block_size, k * block_size, lda)], lda,
                &Adev[IDX(k * block_size, j * block_size, lda)], lda,
                {deps[IDX(k, k, num_blocks)], deps[IDX(k, j, num_blocks)]});
        }

        for (int i = k + 1; i < num_blocks; ++i) {
            for (int j = k + 1; j < num_blocks; ++j) {
                deps[IDX(i, j, num_blocks)] = oneapi::mkl::blas::column_major::gemm(
                    q, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N, block_size,
                    block_size, block_size, -1.0,
                    &Adev[IDX(i * block_size, k * block_size, lda)], lda,
                    &Adev[IDX(k * block_size, j * block_size, lda)], lda, 1.0,
                    &Adev[IDX(i * block_size, j * block_size, lda)], lda,
                    {deps[IDX(i, k, num_blocks)], deps[IDX(k, j, num_blocks)],
                     deps[IDX(i, j, num_blocks)]});
            }
        }
    }
    q.wait();
}

sycl::event runtime_sycl::dgetrf_nopiv_sycl_ld(int N, int lda, double *A,
                                               sycl::event dep) {
    return q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dep);
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
