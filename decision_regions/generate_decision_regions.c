#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <string.h>

#include "constellation.h"
#include "decision_regions.h"

int main(void)
{
    int M = 16;

    double S_dB = 10.0;
    double S = pow(10.0, S_dB / 10.0);

    //GRID PARAMETERS: Change this manually for better quality
    int Nx = 500;
    int Ny = 500;

    double x_min = -10.0;
    double x_max =  10.0;

    double y_min = -10.0;
    double y_max =  10.0;

    int Npoints = Nx * Ny;

    double complex *X_array =
        generate_psk(M);

    if (X_array == NULL) {

        fprintf(
            stderr,
            "Error generating constellation\n"
        );

        return EXIT_FAILURE;
    }

    //INR: Change this manually
    double INR_dB = 15.0;

    double INR = pow(10.0, INR_dB / 10.0);

    char filename[200];

    snprintf(
        filename,
        sizeof(filename),
        "decision_regions_%dPSK_SNR10dB_INR%.1fdB_ML%dp.csv",M,INR_dB,Nx
    );

    double complex *z =
        malloc(sizeof(*z) * Npoints);

    int *matched_ML = malloc(sizeof(*matched_ML) * Npoints);


    if (z == NULL ||
        matched_ML == NULL) {

        fprintf(
            stderr,
            "Memory allocation failed\n"
        );

        free(X_array);
        free(z);
        free(matched_ML);

        return EXIT_FAILURE;
    }

    generate_grid(
        z,
        Nx,
        Ny,
        x_min,
        x_max,
        y_min,
        y_max
    );

    printf(
        "\n%d-PSK: "
        "SNR = %.1f dB, "
        "INR = %.1f dB "
        "(linear INR = %.6f)\n",

        M,
        S_dB,
        INR_dB,
        INR
    );

    compute_ML_regions_serial(
        z,
        Npoints,
        X_array,
        M,
        S,
        INR,
        matched_ML
    );

    FILE *fp = fopen(filename, "w");

    if (fp == NULL) {
        fprintf(stderr, "Could not open %s\n", filename);

        free(X_array);
        free(z);
        free(matched_ML);

        return EXIT_FAILURE;
    }

    fprintf(fp, "x,y,matched_ML\n");

    for (int k = 0; k < Npoints; k++) {
        fprintf(fp, "%.17g,%.17g,%d\n", creal(z[k]), cimag(z[k]), matched_ML[k]);
    }

    fclose(fp);

    printf("Saved: %s\n", filename);

        free(X_array);
        free(z);
        free(matched_ML);

        return EXIT_SUCCESS;
}