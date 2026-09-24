#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <omp.h>

#include "constellation.h"
#include "decision_regions.h"

#define NUM_THREADS 16 // change manually for thread sweep
#define REPEATS 5 

static double get_time_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static FILE *open_benchmark_file(const char *filename)
{
    FILE *fp = fopen(filename, "a+");

    if (fp == NULL)
        return NULL;

    fseek(fp, 0, SEEK_END);

    long file_size = ftell(fp);

    if (file_size == 0) {
        fprintf(fp, "Implementation,Modulation,M,SNR_dB,INR_dB,Threads,Nx,Ny,Npoints,MLTime_s\n");
    }

    return fp;
}

int main(void)
{
    // system parameters
    int M = 16;
    const char *modulation_name = "PSK";

    double S_dB = 10.0;
    double S = pow(10.0, S_dB / 10.0);

    double INR_dB = 15.0;
    double INR = pow(10.0, INR_dB / 10.0);

    omp_set_num_threads(NUM_THREADS);

    int grid_sizes[] = {
        200,
        500,
        1000,
        2000,
        3000
    };

    int num_grids = sizeof(grid_sizes) / sizeof(grid_sizes[0]);

    double x_min = -10.0;
    double x_max = 10.0;

    double y_min = -10.0;
    double y_max = 10.0;

    // constellation
    double complex *X_array = generate_psk(M);

    if (X_array == NULL) {
        fprintf(stderr, "Error generating constellation\n");
        return EXIT_FAILURE;
    }

    const char *benchmark_filename = "decision_regions_timing.csv";

    FILE *benchmark_fp = open_benchmark_file(benchmark_filename);

    if (benchmark_fp == NULL) {
        fprintf(stderr, "Could not open benchmark output file\n");
        free(X_array);
        return EXIT_FAILURE;
    }

    //loop over grid sizes
    for (int g = 0; g < num_grids; g++) {

        int Nx = grid_sizes[g];
        int Ny = grid_sizes[g];
        int Npoints = Nx * Ny;

        printf("\n");
        printf("Implementation: C-openMP\n");
        printf("Modulation: %d-%s\n", M, modulation_name);
        printf("Threads: %d\n", NUM_THREADS);
        printf("GRID: %d x %d = %d points\n", Nx, Ny, Npoints);
        printf("INR = %.1f dB\n", INR_dB);
        printf("========================================\n");

        double complex *z = malloc(sizeof(*z) * (size_t)Npoints);
        int *matched_ML = malloc(sizeof(*matched_ML) * (size_t)Npoints);

        if (z == NULL || matched_ML == NULL) {
            fprintf(stderr, "Memory allocation failed for grid %d x %d\n", Nx, Ny);

            free(z);
            free(matched_ML);

            fclose(benchmark_fp);
            free(X_array);

            return EXIT_FAILURE;
        }

        generate_grid(z, Nx, Ny, x_min, x_max, y_min, y_max);

        //warm-up not timed
        compute_ML_regions_openMP(z, Npoints, X_array, M, S, INR, matched_ML);

        // ML timing
        double ML_times[REPEATS];

        for (int rep = 0; rep < REPEATS; rep++) {

            double start = get_time_seconds();

            compute_ML_regions_openMP(z, Npoints, X_array, M, S, INR, matched_ML);

            double end = get_time_seconds();

            ML_times[rep] = end - start;
        }

        double time_ML = 0.0;
        for (int rep = 0; rep < REPEATS; rep++)
            time_ML += ML_times[rep];
        time_ML /= REPEATS;

        printf("ML computation: %.9f s\n", time_ML);


        fprintf(
            benchmark_fp,
            "C-openMP,%s,%d,%.2f,%.2f,%d,%d,%d,%d,%.9f\n",
            modulation_name,
            M,
            S_dB,
            INR_dB,
            NUM_THREADS,
            Nx,
            Ny,
            Npoints,
            time_ML
        );

        fflush(benchmark_fp);

        free(z);
        free(matched_ML);
    }

    fclose(benchmark_fp);
    free(X_array);

    printf("\n");
    printf("ML BENCHMARK COMPLETE\n");
    printf("Results saved to:\n%s\n", benchmark_filename);

    return EXIT_SUCCESS;
}