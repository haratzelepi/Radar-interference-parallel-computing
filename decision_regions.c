#include <complex.h>
#include <math.h>
#include <omp.h>

#include "detector.h"

void generate_grid(
    double complex *z,
    int Nx,
    int Ny,
    double x_min,
    double x_max,
    double y_min,
    double y_max
)
{
    double dx = (x_max - x_min) / (Nx - 1);
    double dy = (y_max - y_min) / (Ny - 1);


    for (int ix = 0; ix < Nx; ix++) {

        double x = x_min + ix * dx;

        for (int iy = 0; iy < Ny; iy++) {

            double y = y_min + iy * dy;

            int k = ix * Ny + iy;

            z[k] = x + I * y;
        }
    }
}


void compute_TIN_regions_serial(
    const double complex *z,
    int Npoints,
    const double complex *X_array,
    int M,
    double S,
    int *matched_TIN
)
{
    for (int k = 0; k < Npoints; k++) {

        matched_TIN[k] =
            detector_TIN(
                z[k],
                X_array,
                M,
                S
            );
    }
}


void compute_IC_regions_serial(
    const double complex *z,
    int Npoints,
    const double complex *X_array,
    int M,
    double S,
    double INR,
    int *matched_IC
)
{
    for (int k = 0; k < Npoints; k++) {

        matched_IC[k] =
            detector_IC(
                z[k],
                X_array,
                M,
                S,
                INR
            );
    }
}


void compute_ML_regions_serial(
    const double complex *z,
    int Npoints,
    const double complex *X_array,
    int M,
    double S,
    double INR,
    int *matched_ML
)
{
    for (int k = 0; k < Npoints; k++) {

        matched_ML[k] =
            detector_ML(
                z[k],
                X_array,
                M,
                S,
                INR
            );
    }
}

void compute_ML_regions_openMP(
    const double complex *z,
    int Npoints,
    const double complex *X_array,
    int M,
    double S,
    double INR,
    int *matched_ML
)
{
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < Npoints; k++) {

        matched_ML[k] =
            detector_ML(
                z[k],
                X_array,
                M,
                S,
                INR
            );
    }
}