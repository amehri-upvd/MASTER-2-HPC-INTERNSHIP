#include "../include/utiles.h"

//multiplication de deux matrices 
double **multMat(double **a, double **b,int n){
   
    double **c;
    c = (double**)malloc(n*sizeof(double*));
        for(int i=0;i<n;i++) {
            c[i] = (double*)malloc(n*sizeof(double));
        }

    for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }           
    }
    return c;
}
//multiplication matrice et vecteur
double *multMatVect(double **a, double *v,int n){
    double *c;
    c = (double*)malloc(n*sizeof(double));

    for(int i=0;i<n;i++) {
        for(int j=0;j<n;j++) {
            c[i] += a[i][j]*v[j];
        }
    }
    return c;
}
//tester egalité entre deux matrices
int EgaliteTestMat(double **a, double **b,int n){
    for (int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(floor(abs(a[i][j]-b[i][j]))!= 0) return 0;//a!=b
        }
    }
    return 1;//a==b
}
//tester egalité entre une matrice et un vecteur
int EgaliteTestvect(double *a, double *b,int n){
    for (int i = 0; i < n; i++) {
        if(floor(abs(a[i]-b[i]))!=0) {;return 0;}//a!=b
    }
    return 1;//a==b
}
//les fonctions d'affichage
void printMat(double **a,int n) {
  int i,j;

  for(i=0;i<n;i++) {
    for(j=0;j<n;j++) {
        printf("%0.2lf \t",a[i][j]);
    };
    printf("\n");
  }
}

void printVect(double *x,int n){
    for(int i=0;i<n;i++) {
        printf("%lf \t",x[i]);
    }
    printf("\n");
}