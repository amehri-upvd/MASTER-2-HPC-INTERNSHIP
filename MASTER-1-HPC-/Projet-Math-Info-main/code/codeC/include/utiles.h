#ifndef UTILES_H
#define UTILES_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

//les fonctions utiles au deroulement du programme main
void *createB(double * B,double * x);
double **multMat(double **a, double **b,int n);
double *multMatVect(double **a, double *v,int n);
int EgaliteTestMat(double **a, double **b,int n);
int EgaliteTestvect(double *a, double *b,int n);
void printMat(double **a,int n);
void printVect(double *x,int n);

#endif

