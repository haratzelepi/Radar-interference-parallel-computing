#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <curand_kernel.h>

#define PI 3.14159265358979323846

#define THREADS_PER_BLOCK 32
#define NUM_RUNS 5

void cuda_check(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(err));
        exit(1);
    }
}

#define CUDA_CHECK(call) cuda_check((call), __FILE__, __LINE__)

__device__ cuDoubleComplex generate_noise(curandStatePhilox4_32_10_t *state)
{
    double u1 = curand_uniform_double(state);
    double u2 = curand_uniform_double(state);

    double r = sqrt(-log(u1));
    double theta = 2.0 * PI * u2;

    return make_cuDoubleComplex(r * cos(theta), r * sin(theta));
}

__device__ cuDoubleComplex generate_Y(cuDoubleComplex X, double S, double INR, curandStatePhilox4_32_10_t *state)
{
    double theta = 2.0 * PI * curand_uniform_double(state);

    cuDoubleComplex Z = generate_noise(state);

    double sqrtS = sqrt(S);
    double sqrtINR = sqrt(INR);

    double real_Y = sqrtS * cuCreal(X) + sqrtINR * cos(theta) + cuCreal(Z);
    double imag_Y = sqrtS * cuCimag(X) + sqrtINR * sin(theta) + cuCimag(Z);

    return make_cuDoubleComplex(real_Y, imag_Y);
}

__global__ void generate_psk(int M, cuDoubleComplex *X_array)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 0; i < M; i++) {
            double theta = 2.0 * PI * i / M;
            X_array[i] = make_cuDoubleComplex(cos(theta), sin(theta));
        }
    }
}

// STABLE log(I0)
__device__ double log_I0_stable(double z)
{
    if (z < 50.0) {
        return log(cyl_bessel_i0(z));
    }

    double inv_z = 1.0 / z;
    double correction = 1.0 + inv_z / 8.0 + 9.0 * inv_z * inv_z / 128.0 + 225.0 * inv_z * inv_z * inv_z / 3072.0;

    return z - 0.5 * log(2.0 * PI * z) + log(correction);
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
        double log_I0 = log_I0_stable(z);

        d = distance * distance - log_I0;

        if (d < dmin) {
            dmin = d;
            matched_index = i;
        }
    }

    return matched_index;
}

__device__ int detector_TIN(cuDoubleComplex y, const cuDoubleComplex *X_array, int M, double S)
{
    double d;
    double dmin = INFINITY;
    int matched_index = 0;

    double sqrtS = sqrt(S);

    for (int i = 0; i < M; i++) {
        double diff_real = cuCreal(y) - sqrtS * cuCreal(X_array[i]);
        double diff_imag = cuCimag(y) - sqrtS * cuCimag(X_array[i]);

        double distance = hypot(diff_real, diff_imag);

        d = distance * distance;

        if (d < dmin) {
            dmin = d;
            matched_index = i;
        }
    }

    return matched_index;
}

__device__ int detector_IC(cuDoubleComplex y, const cuDoubleComplex *X_array, int M, double S, double INR)
{
    double d;
    double dmin = INFINITY;
    int matched_index = 0;

    double sqrtS = sqrt(S);
    double sqrtINR = sqrt(INR);

    for (int i = 0; i < M; i++) {
        double diff_real = cuCreal(y) - sqrtS * cuCreal(X_array[i]);
        double diff_imag = cuCimag(y) - sqrtS * cuCimag(X_array[i]);

        double distance = hypot(diff_real, diff_imag) - sqrtINR;

        d = distance * distance;

        if (d < dmin) {
            dmin = d;
            matched_index = i;
        }
    }

    return matched_index;
}

// RNG INITIALIZATION
// One independent RNG state for every CUDA thread
__global__ void init_rng(curandStatePhilox4_32_10_t *states, unsigned long long seed)
{
    unsigned long long global_tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    curand_init(seed, global_tid, 0ULL, &states[global_tid]);
}

// One block = one INR point
// Threads of the block share N_samples
__global__ void SER_simulate(const cuDoubleComplex *X_array, double *SER, curandStatePhilox4_32_10_t *states, int M, long long N_samples, double S, double INR_dB_min, double INR_dB_step)
{
    int INR_index = blockIdx.x;
    int tid = threadIdx.x;

    unsigned long long global_tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ double INR;
    __shared__ unsigned long long errors[THREADS_PER_BLOCK];

    if (tid == 0) {
        double INR_dB = INR_dB_min + INR_index * INR_dB_step;
        INR = pow(10.0, INR_dB / 10.0);
    }

    __syncthreads();

    curandStatePhilox4_32_10_t state = states[global_tid];

    unsigned long long local_errors = 0;

    for (long long k = tid; k < N_samples; k += blockDim.x) {
        int send_id = (int)(curand(&state) % M);

        cuDoubleComplex X_send = X_array[send_id];

        cuDoubleComplex Y = generate_Y(X_send, S, INR, &state);

        int matched_id = detector_ML(Y, X_array, M, S, INR);

        if (matched_id != send_id) {
            local_errors++;
        }
    }

    states[global_tid] = state;

    errors[tid] = local_errors;

    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            errors[tid] += errors[tid + offset];
        }

        __syncthreads();
    }

    if (tid == 0) {
        SER[INR_index] = (double)errors[0] / (double)N_samples;
    }
}

struct timespec t_start, t_end;

int main(void)
{
    // system parameters

    int M = 16;
    const long long N_samples = (long long)1e2;

    double S_dB = 10.0;
    double S = pow(10.0, S_dB / 10.0);

    double INR_dB_min = -20.0;
    double INR_dB_max = 60.0;

    int INR_points = 1000;

    double INR_dB_step = (INR_dB_max - INR_dB_min) / (INR_points - 1);

    const char *implementation = "CUDA";

    double *SER = (double *)malloc((size_t)INR_points * sizeof(double));

    if (SER == NULL) {
        printf("Memory allocation error\n");
        return 1;
    }

    cuDoubleComplex *d_Xarray;
    double *d_SER;
    curandStatePhilox4_32_10_t *d_states;

    size_t total_threads = (size_t)INR_points * THREADS_PER_BLOCK;

    CUDA_CHECK(cudaMalloc((void **)&d_Xarray, (size_t)M * sizeof(cuDoubleComplex)));

    CUDA_CHECK(cudaMalloc((void **)&d_SER, (size_t)INR_points * sizeof(double)));

    CUDA_CHECK(cudaMalloc((void **)&d_states, total_threads * sizeof(curandStatePhilox4_32_10_t)));

    generate_psk<<<1, 1>>>(M, d_Xarray);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    init_rng<<<INR_points, THREADS_PER_BLOCK>>>(d_states, 1ULL);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    double total_time = 0.0;

    for (int run = 0; run < NUM_RUNS; run++) {

        // start timer
        clock_gettime(CLOCK_MONOTONIC, &t_start);

        SER_simulate<<<INR_points, THREADS_PER_BLOCK>>>(d_Xarray, d_SER, d_states, M, N_samples, S, INR_dB_min, INR_dB_step);

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(SER, d_SER, (size_t)INR_points * sizeof(double), cudaMemcpyDeviceToHost));

        // stop timer
        clock_gettime(CLOCK_MONOTONIC, &t_end);

        total_time += (double)(t_end.tv_sec - t_start.tv_sec) + (double)(t_end.tv_nsec - t_start.tv_nsec) / 1e9;
    }

    double average_time = total_time / NUM_RUNS;

    printf("\nSER CUDA BENCHMARK\n");
    printf("Implementation    : %s\n", implementation);
    printf("M                 : %d\n", M);
    printf("N_samples         : %lld\n", N_samples);
    printf("INR_points        : %d\n", INR_points);
    printf("Blocks            : %d\n", INR_points);
    printf("Threads per block : %d\n", THREADS_PER_BLOCK);
    printf("Total CUDA threads: %zu\n", total_threads);
    printf("Average time      : %.6f seconds (%d runs)\n", average_time, NUM_RUNS);

    FILE *fp = fopen("SER_timings.csv", "a+");

    if (fp == NULL) {
        printf("Error opening timing file\n");
        CUDA_CHECK(cudaFree(d_Xarray));
        CUDA_CHECK(cudaFree(d_SER));
        CUDA_CHECK(cudaFree(d_states));
        free(SER);
        return 1;
    }

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);

    if (file_size == 0)
        fprintf(fp, "Implementation,M,N_samples,INR_points,Threads,Blocks,Threads_per_block,Time_seconds\n");

    fprintf(fp, "%s,%d,%lld,%d,%zu,%d,%d,%.9f\n",
        implementation, M, N_samples, INR_points, total_threads, INR_points, THREADS_PER_BLOCK, average_time);

    fclose(fp);

    printf("Results saved to SER_timings.csv\n");

    CUDA_CHECK(cudaFree(d_Xarray));

    CUDA_CHECK(cudaFree(d_SER));

    CUDA_CHECK(cudaFree(d_states));

    free(SER);

    return 0;
}