#ifndef DECISION_REGIONS_H
#define DECISION_REGIONS_H 

#include <complex.h>

void generate_grid(
    double complex *z,
    int Nx,
    int Ny,
    double x_min,
    double x_max,
    double y_min,
    double y_max
);

void compute_TIN_regions_serial(
    const double complex *z,
    int Npoints,
    const double complex *X_array,
    int M,
    double S,
    int *matched_TIN
);


void compute_IC_regions_serial(
    const double complex *z,
    int Npoints,
    const double complex *X_array,
    int M,
    double S,
    double INR,
    int *matched_IC
);

void compute_ML_regions_serial(
    const double complex *z,
    int Npoints,
    const double complex *X_array,
    int M,
    double S,
    double INR,
    int *matched_ML
);

void compute_ML_regions_openMP(
    const double complex *z,
    int Npoints,
    const double complex *X_array,
    int M,
    double S,
    double INR,
    int *matched_ML
);

#endif