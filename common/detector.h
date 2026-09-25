#ifndef DETECTOR_H
#define DETECTOR_H

#include <complex.h>

int detector_TIN(double complex y,
                 const double complex *X_array,
                 int M,
                 double S);

int detector_IC(double complex y,
                const double complex *X_array,
                 int M,
                 double S,
                 double INR);

int detector_ML(double complex y,
                 const double complex *X_array,
                 int M,
                 double S,
                 double INR);

#endif