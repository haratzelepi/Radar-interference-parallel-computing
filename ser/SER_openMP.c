#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <omp.h>

#include "constellation.h"
#include "detector.h"

#define PI 3.14159265358979323846

#define NUM_THREADS 16
#define NUM_RUNS 5

double complex generate_noise(unsigned int *seed)
{
    double u1 = (double)rand_r(seed) / ((double)RAND_MAX + 1.0);
    double u2 = (double)rand_r(seed) / ((double)RAND_MAX + 1.0);

    if (u1 == 0.0)
        u1 = 1.0 / ((double)RAND_MAX + 1.0);

    double r = sqrt(-log(u1));
    double theta = 2.0 * PI * u2;

    return r * cos(theta) + I * r * sin(theta);
}

double complex generate_Y(double complex X, double S, double INR, unsigned int *seed)
{
    double theta = 2.0 * PI * ((double)rand_r(seed) / ((double)RAND_MAX + 1.0));
    double complex E = cexp(I * theta);
    double complex Z = generate_noise(seed);

    double complex Y = sqrt(S) * X + sqrt(INR) * E + Z;
    return Y;
   
}

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

    const char *implementation = "openMP";

    double *SER = malloc(sizeof(*SER) * INR_points);

    omp_set_num_threads(NUM_THREADS);

    if (SER == NULL) {
        printf("Memory allocation error\n");
        return 1;
    }

    // send constellation
    double complex *X_array = generate_psk(M);

    if (X_array == NULL) {
        printf("Error generating constellation\n");
        free(SER);
        return 1;
    }

    double INR_dB_step = (INR_dB_max - INR_dB_min) / (INR_points - 1);

    double total_time = 0.0;

    for (int run = 0; run < NUM_RUNS; run++) {

        // start timer
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            unsigned int seed = 1 + thread_id;

            #pragma omp for schedule(static)
            for (int i = 0; i < INR_points; i++) {

                double INR_dB = INR_dB_min + i * INR_dB_step;
                double INR = pow(10.0, INR_dB / 10.0);

                long long N_errors = 0;

                for (long long k = 0; k < N_samples; k++) {

                    int send_id = rand_r(&seed) % M;
                    double complex X_send = X_array[send_id];

                    double complex Y = generate_Y(X_send, S, INR, &seed);

                    int matched_id = detector_ML(Y, X_array, M, S, INR);

                    if (matched_id != send_id)
                        N_errors++;
                }

                SER[i] = (double)N_errors / (double)N_samples;
            }
        }

        // stop timer
        clock_gettime(CLOCK_MONOTONIC, &end);

        total_time += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    }

    double average_time = total_time / NUM_RUNS;

    printf("\nSER BENCHMARK \n");
    printf("Implementation : %s\n", implementation);
    printf("M              : %d\n", M);
    printf("N_samples      : %lld\n", N_samples);
    printf("INR_points     : %d\n", INR_points);
    printf("Threads        : %d\n", NUM_THREADS);
    printf("Average time   : %.6f seconds (%d runs)\n", average_time, NUM_RUNS);

    FILE *fp = fopen("SER_timings.csv", "a+");

    if (fp == NULL) {
        printf("Error opening timing file\n");
        free(X_array);
        free(SER);
        return 1;
    }

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);

    if (file_size == 0)
        fprintf(fp, "Implementation,M,N_samples,INR_points,Threads,Blocks,Threads_per_block,Time_seconds\n");

    fprintf(fp, "%s,%d,%lld,%d,%d,%d,%d,%.9f\n",
            implementation, M, N_samples, INR_points, NUM_THREADS, 0, 0, average_time);

    fclose(fp);

    printf("Results saved to SER_timings.csv\n");

    free(X_array);
    free(SER);

    return 0;
}