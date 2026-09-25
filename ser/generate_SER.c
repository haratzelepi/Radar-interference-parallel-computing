#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#include "constellation.h"
#include "detector.h"

#define PI 3.14159265358979323846

double complex generate_noise(void)
{
    double u1 = (double)rand() / ((double)RAND_MAX + 1.0);
    double u2 = (double)rand() / ((double)RAND_MAX + 1.0);

    if (u1 == 0.0)
        u1 = 1.0 / ((double)RAND_MAX + 1.0);

    double r = sqrt(-log(u1));
    double theta = 2.0 * PI * u2;

    return r * cos(theta) + I * r * sin(theta);
}

double complex generate_Y(double complex X, double S, double INR)
{
    double theta = 2.0 * PI * ((double)rand() / ((double)RAND_MAX + 1.0));
    double complex E = cexp(I * theta);
    double complex Z = generate_noise();

    return sqrt(S) * X + sqrt(INR) * E + Z;
}

int main(void)
{
    int M = 16;

    // Change manually for high-quality validation
    const long long N_samples = (long long)1e2;

    double S_dB = 10.0;
    double S = pow(10.0, S_dB / 10.0);

    double INR_dB_min = -20.0;
    double INR_dB_max = 60.0;

    int INR_points = 1000;

    double complex *X_array = generate_psk(M);

    if (X_array == NULL) {
        fprintf(stderr, "Error generating constellation\n");
        return EXIT_FAILURE;
    }

    char filename[200];

    snprintf(filename, sizeof(filename),
             "SER_validation_%dPSK_%lldsamples_%dpoints.csv",
             M, N_samples, INR_points);

    FILE *fp = fopen(filename, "w");

    if (fp == NULL) {
        fprintf(stderr, "Could not open %s\n", filename);
        free(X_array);
        return EXIT_FAILURE;
    }

    fprintf(fp, "INR_dB,INR_SNR_dB,SER_TIN,SER_ML,SER_IC\n");

    srand(1);

    double INR_dB_step = (INR_dB_max - INR_dB_min) / (INR_points - 1);

    for (int i = 0; i < INR_points; i++) {

        double INR_dB = INR_dB_min + i * INR_dB_step;
        double INR = pow(10.0, INR_dB / 10.0);

        long long errors_TIN = 0;
        long long errors_ML = 0;
        long long errors_IC = 0;

        for (long long k = 0; k < N_samples; k++) {

            int send_id = rand() % M;
            double complex X_send = X_array[send_id];

            double complex Y = generate_Y(X_send, S, INR);

            int matched_TIN = detector_TIN(Y, X_array, M, S);
            int matched_ML = detector_ML(Y, X_array, M, S, INR);
            int matched_IC = detector_IC(Y, X_array, M, S, INR);

            if (matched_TIN != send_id)
                errors_TIN++;

            if (matched_ML != send_id)
                errors_ML++;

            if (matched_IC != send_id)
                errors_IC++;
        }

        double SER_TIN = (double)errors_TIN / (double)N_samples;
        double SER_ML = (double)errors_ML / (double)N_samples;
        double SER_IC = (double)errors_IC / (double)N_samples;

        double INR_SNR_dB = INR_dB - S_dB;

        fprintf(fp, "%.9f,%.9f,%.12g,%.12g,%.12g\n",
                INR_dB, INR_SNR_dB, SER_TIN, SER_ML, SER_IC);
    }

    fclose(fp);

    printf("\nSER validation complete\n");
    printf("M          : %d-PSK\n", M);
    printf("SNR        : %.1f dB\n", S_dB);
    printf("N_samples  : %lld\n", N_samples);
    printf("INR_points : %d\n", INR_points);
    printf("Saved      : %s\n", filename);

    free(X_array);

    return EXIT_SUCCESS;
}