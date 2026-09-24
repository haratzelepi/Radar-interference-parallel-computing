#include <math.h>
#include <stdlib.h>
#include <complex.h>

#define PI 3.14159265358979323846

double complex *generate_pam(int M){

    double Es = (M*M-1)/3.0;
    double complex *X_array = malloc(M*sizeof(*X_array));
    
    for (int i=0; i<M; i++){

        X_array[i] = (2.0*i-M+1) /sqrt(Es);

    }

    return X_array;
}

double complex *generate_psk(int M){

    double complex *X_array = malloc(sizeof(*X_array)*M);

    for (int i=0; i<M; i++){
       
        double theta = 2.0*PI*i/M;
        X_array[i] = cos(theta) + I*sin(theta);
    }

    return X_array;
}

double complex *generate_qam(int M){

    double complex *X_array = malloc(M*sizeof(*X_array));
    double Es = 2*(M-1)/3.0;
    int L = sqrt(M);
    int i =0;

    if (L * L != M)
    return NULL;

    for (int p=0; p<L; p++){
        for (int q=0; q<L; q++){
            
            X_array[i] = ((2*p-L+1)+I*(2*q-L+1))/sqrt(Es);

            i++;

            }
        }      

    return X_array;

}

