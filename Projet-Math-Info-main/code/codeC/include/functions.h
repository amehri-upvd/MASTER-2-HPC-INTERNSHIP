#ifndef FUNCTIONS_H
#define FUNCTIONS_H  

#include <stdlib.h>
#include <stdio.h>
#include <math.h>


//les function principale pour la resolution 
void decompLU_Tridiag(double **a,double **l,double **u,int n);
void decompLU(double** A, double** L, double** U, int n);
double *resoudres(double **l,double **u,double *b,int n);
void maillage(double * x,int n);
double s(double x);
void *createB(double * B,double * x);

#endif