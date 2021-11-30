#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

typedef double **mat;



void mat_zero(mat x, int n) {
#pragma omp parallel for shared(x)
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) x[i][j] = 0;
}


mat mat_new(int n)
{
    mat x = malloc(sizeof(double*) * n);
    x[0] = malloc(sizeof(double) * n * n);

#pragma omp parallel for shared(x)
    for (int i = 0; i < n; i++) x[i] = x[0] + n * i;
    mat_zero(x, n);

    return x;
}

mat load_matrix(char* filename, int matrix_size) {
    mat x = mat_new(matrix_size);
    FILE* f = fopen(filename, "r+");
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            fscanf(f, "%lf", &(x[i][j]));
        }
    }
    fclose(f);

    return x;
}

void mat_del(mat x) { free(x[0]); free(x); }


mat mat_mul(mat a, mat b, int n)
{
    mat c = mat_new(n);
#pragma omp parallel for shared(c)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                c[i][j] += a[i][k] * b[k][j];
    return c;
}


void mat_pivot(mat a, mat p, int n)
{
#pragma omp parallel for shared(p)
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) { p[i][j] = (i == j); }
//#pragma omp parallel for shared(a,p)
    for (int i = 0; i < n; i++) {
        int max_j = i;
        for (int j = i; j < n; j++) {
            if (fabs(a[j][i]) > fabs(a[max_j][i])) max_j = j;
        }

        if (max_j != i) {
            for (int k = 0; k < n; k++) {
                {
                    double tmp = p[i][k];
                    p[i][k] = p[max_j][k];
                    p[max_j][k] = tmp;
                };
            }
        }
    }
}

void mat_LU(mat A, mat L, mat U, mat P, int n)
{
//    double tim = omp_get_wtime();
    mat_zero(L, n);
    mat_zero(U, n);
//    printf("Zero calculated in %f.\n", omp_get_wtime() - tim);

//    tim = omp_get_wtime();
    mat_pivot(A, P, n);
//    printf("Pivot calculated in %f.\n", omp_get_wtime() - tim);

//    tim = omp_get_wtime();
    mat Aprime = mat_mul(P, A, n);
//    printf("Mat Mul calculated in %f.\n", omp_get_wtime() - tim);

    int i, j;

//    tim = omp_get_wtime();

    for (i = 0; i < n; i++){
        for (j = 0; j < n; j++){
            double s;
            if (j <= i) {
                s = 0;
#pragma omp parallel for shared(L,U) reduction(+:s)
                for (int k = 0; k < j; k++) {
                    s+= L[j][k] * U[k][i];
                }
                U[j][i] = Aprime[j][i] - s;
            }
            if (j >= i) {
                s = 0;
#pragma omp parallel for shared(L,U) reduction(+:s)
                for (int k = 0; k < i; k++) {
                    s += L[j][k] * U[k][i];
                }
                L[j][i] = (Aprime[j][i] - s) / U[i][i];
            }
        }
    }
//    printf("Big Cycle calculated in %f.\n", omp_get_wtime() - tim);
    mat_del(Aprime);
}


void calculate_det(int p, mat A, int n){
//    printf("Initialized \n");
    double tim = omp_get_wtime();
    mat L, P, U;

    L = mat_new(n);
    U = mat_new(n);
    P = mat_new(n);

//    printf("Params initialized in %f. \n", omp_get_wtime() - tim);

    omp_set_num_threads(p);
    omp_set_nested(1);
//    tim = omp_get_wtime();
//    double tim1 = tim;
    mat_LU(A, L, U, P, n);
//    printf("LU calculated in %f. \n", omp_get_wtime() - tim1);

    int swaps = 0;
    double L_diag = 1;
    double U_diag = 1;

    int i;
//    tim1 = omp_get_wtime();
#pragma omp parallel for private(i) reduction(*:L_diag) reduction(*:U_diag) reduction(+:swaps)
    for (i = 0; i < n; i++){
        if (P[i][i] == 0) {
            swaps += 1;
        }
        L_diag *= L[i][i];
        U_diag *= U[i][i];
    }
//    printf("LU calculated in %f. \n", omp_get_wtime() - tim1);
//    printf("Threads: %d\nDeterminant: %f\nTime: %f\n\n", p, pow(-1.0, swaps)*L_diag*U_diag, omp_get_wtime() - tim);
    printf("%d\t%f\t%f\t%d\n", p, pow(-1.0, swaps)*L_diag*U_diag, omp_get_wtime() - tim, n);
}

int main()
{
    printf("Threads\tDeterminant\tTime\tMatrixSize\n");

    int n;
    n = 200;
    mat B = load_matrix("matrix-200", n);
    for (int i = 0; i < 10; i++) {
        for (int j = 1; j < 17; j++){
            calculate_det(j, B, n);
        }
    }

    n = 5000;
    mat A = load_matrix("matrix-5000", n);
    for (int i = 0; i < 10; i++) {
        for (int j = 1; j < 17; j++){
            calculate_det(j, A, n);
        }
    }
    return 0;
}
