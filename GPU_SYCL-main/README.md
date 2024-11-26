<<<<<<< HEAD
# GPU SYCL

=======
# Mini LU

The goal of this program is to study the performance of different versions of LU
factorizations: scalar, block, rectangular, SYCL, etc. The implementations are
done in place according to [this algorithm](https://4m053.pages.math.cnrs.fr/tps/direct/lu).

## Implementations
We currently have 3 different runtimes: sequential, OpenMP task and SYCL (for
both CPU and GPU). Different algorithms are available and runtimes can have
different implementations. The following table describes what is available and
should be kept up to date:

| Runtimes | Types        | Target    | Description                                              | Default |
|----------|--------------|-----------|----------------------------------------------------------|---------|
| seq      | scalar       | CPU       | scalar                                                   |         |
|          | block        | CPU       | square block                                             |         |
|          | rectangular  | CPU       | block with trsm and gemm fused                           | yes     |
| sycl     | scalar       | CPU+GPU   | scalar and sequential                                    |         |
|          | single\_ib   | *CPU*+GPU | blocked, inner-blocked gertrf, sequential inner ger/scal | yes     |
|          | parallel\_ib | CPU+*GPU* | blocked, inner-blocked gertrf, parallel inner ger/scal   |         |
| omp      |              | CPU       | blocked, task-based parallelism                          | yes     |

Some additional information:
 - The block algorithm takes advantage of BLAS routines (GEMM, TRSM) and use our
   custom scalar GETRF function without pivoting.
 - The rectangular algorithm is similar to the block version version but merge
   BLAS calls together to only do one GEMM and two TRSM per iteration.
 - The SYCL `single_ib` gives better performance for CPU workload.
 - The SYCL `parallel_ib` gives better performance for GPU workload.

## Running the code
### Compilation

Mini LU currently relies on an external dependency TCLAP, added as a submodule,
to parse input parameters. You can get it with the following:

```bash
git submodule init
git submodule update
```

The program is compiled using the `icpx` compiler from Intel's oneAPI and the
MKL library, which includes BLAS and LAPACK routines.

```bash
source /opt/intel/oneapi/2024.1/oneapi-vars.sh
ml cmake/3.28.3
```

To compile and launch tests do the following after having loaded a oneAPI stack:

```bash
cmake -B build .
cmake --build build -j
ctest -V --test-dir build
```

### Execution

Here is an example of execution, using a matrix of size 1000x1000 with a
block_size of 500 using the sequential rectangular implementation:  
`./mini_lu -n 1000 -b 500 -c -r seq -t rectangular -s 1234`

You can get the full list of all available options using the `-h` or `--help`
option.

## Contributing
### Data Layout
We use matrices of floating-point numbers in double precision. Our matrices are
implemented as C++ vectors of doubles, stored in a single dimension and in
column-major order.

Specifically, for a matrix $A$ of size $N×N$, the element $a_{ij}$ is located at
the vector index `i + j × N`, where `i` is the row index and `j` is the column
index

![column-major order](doc/images/col_major.png){width=20%}

### Setup clang-format git hook

The goal of this is to unify coding style across different editors and users.
The best solution would be to integrate your own editor to use
`git-clang-format` on save, but this git hook can be used instead. Note that
`git-clang-format` will only format changed lines, so that the history (and `git
blame`) isn't polluted with formatting commits.


First, let's create a `pre-commit` hook, **it will have to be repeated each time
you clone the repo**:

```bash
echo 'PATH=/software/compilers/aocc/aocc-compiler-4.2.0/bin:$PATH /software/compilers/aocc/aocc-compiler-4.2.0/bin/git-clang-format' > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

Now, every commit will have to be formated first, git will reject unformatted
commit.

```bash
# attempt to commit an unformatted file:
$ git commit -am "fixing bug"
changed files:
    src/foo.cpp

# commit was rejected, but sources files are now formatted properly:
$ git status -sb -uno
 M src/foo.cpp

# now that it has been formatted, the commit will proceed as normal:
$ git commit -am "fixing bug"
no modified files to format
[branch_foobar 1e1fe1f] fixing bug
 1 file changed, 4 insertions(+), 88 deletions(-)
```

## Issues

### Check fails on small sizes:

We assume the check failed when the error value is exactly 0. This causes issues
with very small sizes:

```
runtime    type          n   bs  ibs       seed       tfacto        tinit       gflops       tcheck          err check
    seq scalar          1    1   32 1714131866 4.918780e-06 7.330440e-06 2.033024e-04 3.440121e-03 0.000000e+00    KO
    seq scalar          2    2   32 1714131867 7.322291e-06 9.893905e-06 6.828464e-04 3.844572e-03 0.000000e+00    KO
    seq scalar          3    3   32 1714131868 5.429145e-06 7.574446e-06 2.947057e-03 3.222953e-03 3.142939e-01    OK
    seq scalar          4    4   32 1714131869 7.126713e-06 8.665258e-06 5.332051e-03 3.495237e-03 5.896081e-02    OK
```
>>>>>>> 1a07ab4 (first commit)
