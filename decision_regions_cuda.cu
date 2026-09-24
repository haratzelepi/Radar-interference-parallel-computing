#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <time.h>

#define PI 3.14159265358979323846

#define THREADS_PER_BLOCK 32
#define REPEATS 5

static double get_time_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

__global__ void generate_psk(int M, cuDoubleComplex *X_array)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 0; i < M; i++) {
            double theta = 2.0 * PI * i / M;
            X_array[i] = make_cuDoubleComplex(cos(theta), sin(theta));
        }
    }
}

__global__ void generate_grid(cuDoubleComplex *z, int Nx, int Ny, double x_min, double x_max, double y_min, double y_max)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int Npoints = Nx * Ny;

    if (k < Npoints) {
        double dx = (x_max - x_min) / (Nx - 1);
        double dy = (y_max - y_min) / (Ny - 1);

        int ix = k / Ny;
        int iy = k % Ny;

        double x = x_min + ix * dx;
        double y = y_min + iy * dy;

        z[k] = make_cuDoubleComplex(x, y);
    }
}

__device__ int detector_ML(cuDoubleComplex y, const cuDoubleComplex *X_array, int M, double S, double INR)
{
    double d;
    double dmin = INFINITY;
    int matched_index = 0;

    double sqrtS = sqrt(S);
    double sqrtINR = sqrt(INR);

    for (int i = 0; i < M; i++) {
        double diff_real = cuCreal(y) - sqrtS * cuCreal(X_array[i]);
        double diff_imag = cuCimag(y) - sqrtS * cuCimag(X_array[i]);

        double distance = hypot(diff_real, diff_imag);

        double z = 2.0 * sqrtINR * distance;
        double log_I0 = log(cyl_bessel_i0(z));

        d = distance * distance - log_I0;

        if (d < dmin) {
            dmin = d;
            matched_index = i;
        }
    }

    return matched_index;
}

__global__ void compute_ML_regions(const cuDoubleComplex *z, int Npoints, const cuDoubleComplex *X_array, int M, double S, double INR, int *matched_ML)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < Npoints) {
        matched_ML[k] = detector_ML(z[k], X_array, M, S, INR);
    }
}

int main(void)
{
    // system parameters
    int M = 16;

    double S_dB = 10.0;
    double S = pow(10.0, S_dB / 10.0);

    double INR_dB = 15.0;
    double INR = pow(10.0, INR_dB / 10.0);

    int grid_sizes[] = {200, 500, 1000, 2000, 3000};
    int num_grids = sizeof(grid_sizes) / sizeof(grid_sizes[0]);

    double x_min = -10.0;
    double x_max = 10.0;

    double y_min = -10.0;
    double y_max = 10.0;

    cuDoubleComplex *d_X_array;

    CUDA_CHECK(cudaMalloc((void **)&d_X_array, (size_t)M * sizeof(cuDoubleComplex)));

    generate_psk<<<1, 1>>>(M, d_X_array);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const char *filename = "decision_regions_timing.csv";
    FILE *fp = fopen(filename, "a+");

    if (fp == NULL) {
        fprintf(stderr, "Could not open benchmark output file\n");

        CUDA_CHECK(cudaFree(d_X_array));

        return EXIT_FAILURE;
    }

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);

    if (file_size == 0) {
        fprintf(fp, "Implementation,Modulation,M,SNR_dB,INR_dB,Threads,Nx,Ny,Npoints,MLTime_s\n");
    }

    for (int g = 0; g < num_grids; g++) {

        int Nx = grid_sizes[g];
        int Ny = grid_sizes[g];

        int Npoints = Nx * Ny;
        int num_blocks = (Npoints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        cuDoubleComplex *d_z;
        int *d_matched_ML;
        int *matched_ML = (int *)malloc((size_t)Npoints * sizeof(int));

        if (matched_ML == NULL) {
            fprintf(stderr, "Host memory allocation failed for grid %d x %d\n", Nx, Ny);
            fclose(fp);
            CUDA_CHECK(cudaFree(d_X_array));
            return EXIT_FAILURE;
        }

        CUDA_CHECK(cudaMalloc((void **)&d_z, (size_t)Npoints * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc((void **)&d_matched_ML, (size_t)Npoints * sizeof(int)));

        generate_grid<<<num_blocks, THREADS_PER_BLOCK>>>(d_z, Nx, Ny, x_min, x_max, y_min, y_max);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // warm-up
        compute_ML_regions<<<num_blocks, THREADS_PER_BLOCK>>>(d_z, Npoints, d_X_array, M, S, INR, d_matched_ML);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        printf("\n");
        printf("Implementation: CUDA\n");
        printf("Modulation: %dPSK\n", M);
        printf("Threads per Block: %d\n", THREADS_PER_BLOCK);
        printf("GRID: %d x %d = %d points\n", Nx, Ny, Npoints);
        printf("INR = %.1f dB\n", INR_dB);
        
        double total_time = 0.0;

        for (int rep = 0; rep < REPEATS; rep++) {
            double start = get_time_seconds();

            compute_ML_regions<<<num_blocks, THREADS_PER_BLOCK>>>(d_z, Npoints, d_X_array, M, S, INR, d_matched_ML);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(matched_ML, d_matched_ML, (size_t)Npoints * sizeof(int), cudaMemcpyDeviceToHost));

            double end = get_time_seconds();
            total_time += end - start;
        }

        double average_time = total_time / REPEATS;

        printf("========================================\n");
        printf("ML computation: %.9f s\n", average_time);

        fprintf(fp, "CUDA,PSK,%d,%.2f,%.2f,%d,%d,%d,%d,%.9f\n", M, S_dB, INR_dB, THREADS_PER_BLOCK, Nx, Ny, Npoints, average_time);
        fflush(fp);

        CUDA_CHECK(cudaFree(d_z));
        CUDA_CHECK(cudaFree(d_matched_ML));
        free(matched_ML);
    }

    fclose(fp);

    CUDA_CHECK(cudaFree(d_X_array));

    printf("\nCUDA ML BENCHMARK COMPLETE\n");
    printf("Results appended to: %s\n", filename);

    return EXIT_SUCCESS;
}