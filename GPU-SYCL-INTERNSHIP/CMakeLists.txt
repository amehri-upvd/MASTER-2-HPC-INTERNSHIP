# CMake 3.23.5 is the minimum recommended for IntelLLVM on Linux
cmake_minimum_required(VERSION 3.23.5)

set(CMAKE_CXX_COMPILER "icpx")

project(mini_lu CXX)

find_package(IntelSYCL REQUIRED)

add_library(options_flags INTERFACE)
target_compile_options(options_flags INTERFACE -O2 -Wall -g -ferror-limit=4 -fsycl -qmkl=sequential -qopenmp)
target_link_options(options_flags INTERFACE -O2 -Wall -g -fsycl -qmkl=sequential -qopenmp)
target_include_directories(options_flags INTERFACE include ext/tclap/include)

add_library(mini_lu_lib SHARED src/matrix.cpp src/dgetrf.cpp src/dgetrf_util.cpp
  src/runtime_seq.cpp src/runtime_omp.cpp src/runtime_sycl.cpp)
target_link_libraries(mini_lu_lib PRIVATE options_flags)

add_executable(mini_lu src/mini_lu.cpp)
target_link_libraries(mini_lu mini_lu_lib options_flags)

enable_testing()
set(test_options -s 42 -n 512 -b 128 -B 32 -c -l 4)
add_test(NAME test-seq-scalar COMMAND mini_lu ${test_options} -r seq -t scalar)
add_test(NAME test-seq-block COMMAND mini_lu ${test_options} -r seq -t block)
add_test(NAME test-seq-rect COMMAND mini_lu ${test_options} -r seq -t rectangular)
add_test(NAME test-omp COMMAND mini_lu ${test_options} -r omp)
add_test(NAME test-sycl-scalar COMMAND mini_lu ${test_options} -r sycl -t scalar)
add_test(NAME test-sycl COMMAND mini_lu ${test_options} -r sycl)

install(TARGETS mini_lu)
