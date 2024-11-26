#include "../include/matrix.h"
#include "../include/dgetrf.h"
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <random>
#include <vector>

void matrix_print(int &N, std::vector<double> &A_input) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << A_input[IDX(i, j, N)] << " ";
        }
        std::cout << std::endl;
    }
}

void matrix_init_sym_diag_dom(int N, std::vector<double> &A, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    // column major
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j <= i; ++j) {
            double valeur = dis(gen);
            if (i == j)
                A[IDX(i, j, N)] = N;
            else
                A[IDX(i, j, N)] = A[IDX(j, i, N)] = valeur;
        }
    }
}
