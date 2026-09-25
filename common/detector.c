#include <math.h>
#include <complex.h>
#include <gsl/gsl_sf_bessel.h>

int detector_TIN(double complex y,
                 const double complex *X_array,
                 int M,
                 double S)
{
    double d;
    double dmin = INFINITY;
    int matched_index = 0;

    double sqrtS = sqrt(S);

    for (int i = 0; i < M; i++) {

        double distance = cabs(y - sqrtS * X_array[i]);

        d = distance * distance;

        if (d < dmin) {
            dmin = d;
            matched_index = i;
        }
    }

    return matched_index;
}

int detector_IC(double complex y,
                 const double complex *X_array,
                 int M,
                 double S,
                 double INR)
{
    double d;
    double dmin = INFINITY;
    int matched_index = 0;

    double sqrtS = sqrt(S);
    double sqrtINR = sqrt(INR);

    for (int i = 0; i < M; i++) {

        double distance = cabs(y - sqrtS * X_array[i]) - sqrtINR;

        d = distance * distance;

        if (d < dmin) {
            dmin = d;
            matched_index = i;
        }
    }

    return matched_index;
}

int detector_ML(double complex y,
                 const double complex *X_array,
                 int M,
                 double S,
                 double INR)
{
    double d;
    double dmin = INFINITY;
    int matched_index = 0;

    double sqrtS = sqrt(S);
    double sqrtINR = sqrt(INR);

    for (int i = 0; i < M; i++) {

        double distance = cabs(y - sqrtS * X_array[i]);

        double z = 2.0 * sqrtINR * distance;

        double log_I0 = z + log(gsl_sf_bessel_I0_scaled(z));

        d = distance * distance - log_I0;

        if (d < dmin) {
            dmin = d;
            matched_index = i;
        }
    }

    return matched_index;
}


