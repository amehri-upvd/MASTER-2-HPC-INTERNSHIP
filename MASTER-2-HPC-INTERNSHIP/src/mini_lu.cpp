#include "dgetrf.h"
#include "dgetrf_util.h"
#include "matrix.h"
#include "runtime.h"
#include "tclap/CmdLine.h"

#include <cstdlib>
#include <iostream>
#include <math.h>
#include <mkl.h>
#include <sycl/sycl.hpp>
#include <vector>

void check_input(int argc, char **argv, int &N, int &block_size, int &innerblock_size,
                 unsigned &seed, std::string &runtime, std::string &type, bool &check,
                 int &nruns, bool &csv, bool &noheader, bool &onlyheader) {
    try {

        TCLAP::CmdLine cmd("Mini LU factorization", ' ', "0.1");
        TCLAP::ValueArg<int> nArg("n", "N", "matrix size", true, 0, "integer");
        TCLAP::ValueArg<int> bArg("b", "block_size", "size of block", false, -1,
                                  "integer");
        TCLAP::ValueArg<int> ibArg("B", "innerblock_size",
                                   "size of innerblock (default: 32)", false, 32,
                                   "integer");
        TCLAP::ValueArg<int> lArg("l", "nruns", "number of consecutive runs", false, 1,
                                  "integer");
        TCLAP::ValueArg<unsigned int> sArg("s", "seed", "seed for random numbers", false,
                                           static_cast<unsigned int>(time(nullptr)),
                                           "unsigned integer");
        TCLAP::ValueArg<std::string> rArg("r", "runtime", "compute runtime", false, "seq",
                                          "string");
        TCLAP::ValueArg<std::string> tArg("t", "type", "type of algorithm", false,
                                          "default", "string");
        TCLAP::SwitchArg cArg("c", "check", "perform numerical check", false);
        TCLAP::SwitchArg csvArg("C", "csv", "print output format in csv", cmd, false);
        TCLAP::SwitchArg noheadArg("H", "no-header", "don't print output header", cmd,
                                   false);
        TCLAP::SwitchArg onlyheadArg("O", "only-header", "print output header and exit",
                                     cmd, false);

        cmd.add(nArg);
        cmd.add(bArg);
        cmd.add(ibArg);
        cmd.add(lArg);
        cmd.add(sArg);
        cmd.add(rArg);
        cmd.add(tArg);
        cmd.add(cArg);

        cmd.parse(argc, argv);

        N = nArg.getValue();
        block_size = bArg.getValue();
        if (block_size == -1)
            block_size = N; // default to N
        innerblock_size = ibArg.getValue();
        nruns = lArg.getValue();
        seed = sArg.getValue();
        runtime = rArg.getValue();
        type = tArg.getValue();
        check = cArg.getValue();
        csv = csvArg.getValue();
        noheader = noheadArg.getValue();
        onlyheader = onlyheadArg.getValue();

    } catch (TCLAP::ArgException &e) { // catch any exceptions
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }

    if (block_size == 0 || N % block_size != 0) {
        fprintf(stderr, "error: invalid block size, %d must divide N=%d\n", block_size,
                N);
        exit(EXIT_FAILURE);
    }
}

void print_header(int runtime_length, int type_length, bool check_results, bool csv,
                  bool noheader, bool onlyheader) {
    if (!noheader) {
        if (!csv) {
            printf("%*s %*s %10s %4s %4s %10s %12s %12s %12s", runtime_length, "runtime",
                   type_length, "type", "n", "bs", "ibs", "seed", "tfacto", "tinit",
                   "gflops");
        } else {
            printf("%s,%s,%s,%s,%s,%s,%s,%s,%s", "runtime", "type", "n", "bs", "ibs",
                   "seed", "tfacto", "tinit", "gflops");
        }
        if (!csv) {
            if (check_results) {
                printf(" %12s %12s %5s", "tcheck", "err", "check");
            }
        } else {
            printf(",%s,%s,%s", "tcheck", "err", "check");
        }
        printf("\n");
    }

    if (onlyheader) {
        exit(EXIT_SUCCESS);
    }
}

void print_results(int runtime_length, int type_length, bool check_results, bool csv,
                   std::string runtime, std::string type, int N, int block_size,
                   int innerblock_size, unsigned seed, double facto_time,
                   double init_time, double gflops, double check_time, double err_val,
                   bool err_bool) {
    if (!csv) {
        printf("%*s %*s %10d %4d %4d %10u %12e %12e %12e", runtime_length, runtime.data(),
               type_length, type.data(), N, block_size, innerblock_size, seed, facto_time,
               init_time, gflops);
        if (check_results) {
            printf(" %12e %12e %5s", check_time, err_val, err_bool ? "OK" : "KO");
        }
    } else {
        printf("%s,%s,%d,%d,%d,%u,%e,%e,%e", runtime.data(), type.data(), N, block_size,
               innerblock_size, seed, facto_time, init_time, gflops);
        if (check_results) {
            printf(",%e,%e,%s", check_time, err_val, err_bool ? "OK" : "KO");
        }
    }
    printf("\n");
}

int main(int argc, char **argv) {

    int return_code = EXIT_SUCCESS;
    int N, block_size, innerblock_size, nruns;
    std::string facto_type, facto_runtime;
    unsigned seed;
    double start_time, end_time;
    bool check_results, csv, noheader, onlyheader;
    check_input(argc, argv, N, block_size, innerblock_size, seed, facto_runtime,
                facto_type, check_results, nruns, csv, noheader, onlyheader);

    std::vector<double> A(N * N);

    runtime *rt;
    if (facto_runtime == "seq") {
        rt = new runtime_seq(facto_type, N, block_size, innerblock_size);
    } else if (facto_runtime == "sycl") {
        rt = new runtime_sycl(facto_type, N, block_size, innerblock_size);
    } else if (facto_runtime == "omp") {
        rt = new runtime_omp(facto_type, N, block_size, innerblock_size);
    } else {
        std::cerr << "Invalid compute runtime: " << facto_runtime << std::endl;
        exit(EXIT_FAILURE);
    }

    int facto_runtime_length =
        std::max(facto_runtime.length(), std::string("runtime").length());
    int facto_type_length = std::max(facto_type.length(), std::string("type").length());

    print_header(facto_runtime_length, facto_type_length, check_results, csv, noheader,
                 onlyheader);

    for (int run = 0; run < nruns; ++run) {
        start_time = dsecnd();
        matrix_init_sym_diag_dom(N, A, seed);
        end_time = dsecnd();
        double init_time = end_time - start_time;

        rt->prefetch(A);

        start_time = dsecnd();
        rt->dgetrf(A);
        end_time = dsecnd();
        double facto_time = end_time - start_time;

        rt->post(A);

        double check_time = 0;
        double err_val = -1;
        bool err_bool = false;
        if (check_results) {
            start_time = dsecnd();
            err_bool = dgetrf_check(N, A, seed, err_val);
            end_time = dsecnd();
            check_time = end_time - start_time;
            if (!err_bool) // return failure on any failed check
                return_code = EXIT_FAILURE;
        }

        double gflops = dgetrf_flops(N) * 1.e-9 / facto_time;
        print_results(facto_runtime_length, facto_type_length, check_results, csv,
                      facto_runtime, facto_type, N, block_size, innerblock_size, seed,
                      facto_time, init_time, gflops, check_time, err_val, err_bool);

        seed += run * seed + 42; // use a new seed for each run
    }

    delete rt;
    return return_code;
}
