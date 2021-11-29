# Testing parallelism with OpenMP
Initial algorithm: [determinant calculation through LU decomposition](http://www.rosettacode.org/wiki/LU_decomposition#C)

# Compilation
## Witch gcc
Compile file `main.c` with `gcc` and flags `-fopenmp` `-lm`.
```shell
gcc main.c -lm -fopenmp -o OMP
```
*Tested with gcc 9.3.0 on Ubuntu 20.04*

## With cmake
```shell
cmake .
make
```
*Tested with cmake 3.22.0 and make 4.2.1 on Ubuntu 20.04*



# Usage
1. Unpack `matrices.7z` into root directory
2. Run `OMP` (i.e. `./OMP`)