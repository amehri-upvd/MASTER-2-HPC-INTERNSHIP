#include "../include/functions.h"

// parametres de discretisation
#define N  5   // nombre de points de discretisation
#define H  1.0/(N+1)   // le pas d'espace uniforme h = 1/(n+1)
// conditions de Dirichlet
#define ALPHA 0  //condition limite u(O)
#define BETA 0  // condition limite u(1)

void decompLU_Tridiag(double **a,double **l,double **u,int n){
   
    u[0][0]=a[0][0];

    //remplir la sous diagonale de L et la diagonale de U

    for(int i=1;i<n;i++){
        
        l[i][i-1]=a[i][i-1]/u[i-1][i-1];

        u[i][i]=a[i][i] - l[i][i-1]*a[i-1][i];
    }

    //remplir la diagonale de L

    for(int i=0;i<n;i++){

        l[i][i]= 1;
    }

    //remplir la sur diagonale de U
    for(int i=1;i<n;i++){

        u[i-1][i]=a[i-1][i];
    }
}

void decompLU(double** A, double** L, double** U, int n) {
    for (int j = 0; j < n; j++) {
        L[j][j] = 1;
        for (int i = 0; i <= j; i++) {
            double s1 = 0;
            for (int k = 0; k < i; k++) {
                s1 += U[k][j] * L[i][k];
            }
            U[i][j] = A[i][j] - s1;
        }
        for (int i = j+1; i < n; i++) {
            double s2 = 0;
            for (int k = 0; k < j; k++) {
                s2 += U[k][j] * L[i][k];
            }
            L[i][j] = (A[i][j] - s2) / U[j][j];
        }
    }
}

double  *resoudres(double **l,double **u,double *b,int n) {
    double *y,*x;
    //LUX=B et Y=UX resolution par descente (Y) et remonté (X)
    x=(double*)malloc(sizeof(double)*n);
    y=(double*)malloc(sizeof(double)*n);

    y[0]=b[0];
    for(int i=1;i<n;i++) {
        y[i]=b[i]-l[i][i-1]*y[i-1];
    }
    
    x[n-1]=y[n-1]/u[n-1][n-1];
    for(int i=n-2;i>=0;i--) {
        x[i]=(y[i]-u[i][i+1]*x[i+1])/u[i][i];
    }
    return x; 
}

//fonction maillage 
void maillage(double * x,int n)
{ 
    // Créer un tableau pour stocker les points du maillage
    for (int i = 0; i <= n+1; i++)
    {
        x[i] =i * H;
    }
}

//La fonction s en entrée xi et retourne "s(xi) = 4π²sin(2πxi)" sachant que xi = H * i
double s(double x) {
    return 4 * M_PI * M_PI * sin(2 * M_PI * x);
}

//fonction qui retourne le vecteur B
void *createB(double * B,double * x) {
  
  int i;

  B[0] = (s(x[0]) + (ALPHA / H*H)) ;
  B[N-1] = (s(x[N-1]) + (BETA / H*H)) ;
    for(i = 1; i < N-1 ; i++) {
        B[i] = s(x[i]);
    }
  
}