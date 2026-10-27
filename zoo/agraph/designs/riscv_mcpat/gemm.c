// Double-precision GEMM C[m][n] = A[m][k] * B[k][n]; only the multiply is the gem5 region of interest.
#include <stdio.h>
#include <stdlib.h>

#include <gem5/m5ops.h>

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s m k n\n", argv[0]);
        return 1;
    }
    int m = atoi(argv[1]), k = atoi(argv[2]), n = atoi(argv[3]);
    double *a = malloc(sizeof(double) * m * k);
    double *b = malloc(sizeof(double) * k * n);
    double *c = calloc((size_t)m * n, sizeof(double));
    for (int i = 0; i < m * k; i++)
        a[i] = (i % 7) * 0.5;
    for (int i = 0; i < k * n; i++)
        b[i] = (i % 5) * 0.25;

    m5_work_begin(0, 0);
    for (int i = 0; i < m; i++)
        for (int p = 0; p < k; p++) {
            double aip = a[i * k + p];
            for (int j = 0; j < n; j++)
                c[i * n + j] += aip * b[p * n + j];
        }
    m5_work_end(0, 0);

    double sum = 0;
    for (int i = 0; i < m * n; i++)
        sum += c[i];
    printf("gemm %dx%dx%d checksum %.6f\n", m, k, n, sum);
    return 0;
}
