#include "./include/functions.h"
#include "./include/utiles.h"

// parametres de discretisation
#define N  5   // nombre de points de discretisation
#define H  1.0/(N+1)   // le pas d'espace uniforme h = 1/(n+1)
// conditions de Dirichlet
#define ALPHA 0  //condition limite u(O)
#define BETA 0  // condition limite u(1)

int main(){
    double **a, **l, **u, *B ,*x,*xi;
    int i;

    a = (double**)malloc(N*sizeof(double*));
    l = (double**)malloc(N*sizeof(double*));
    u = (double**)malloc(N*sizeof(double*));
        for(i=0;i<N;i++) {
            a[i] = (double*)malloc(N*sizeof(double));
            l[i] = (double*)malloc(N*sizeof(double));
            u[i] = (double*)malloc(N*sizeof(double));
        }
    B= (double*)malloc(N*sizeof(double));
    xi = (double*)malloc(sizeof(double)*N);
  
  //definir A 
  for(i=1;i<N;i++) {
    a[i-1][i]=-1/(H*H);
    a[i][i-1]=-1/(H*H);
    a[i-1][i-1]=2/(H*H);
    a[N-1][N-1]=2/(H*H);
  }
  
  printf("Matrice A :\n");
  printMat(a,N);

  //LU 
  //decompLU(a,l,u,N);
  
  //LU TRIDIAGGONALE
  decompLU_Tridiag(a,l,u,N);
  
  //test unitaire de la decomposition LU
  if(!EgaliteTestMat(multMat(l,u,N),a,N)){
    printf("Erreur lors de la decomposition LU : A != L*U\n");
    return EXIT_FAILURE;
  }

  printf("Matrice L :\n");
  printMat(l,N);
 
  printf("Matrice U :\n");
  printMat(u,N);
  
  maillage(xi,N);
  createB(B,xi);
  
  printf("Matrice B :\n");
  printVect(B,N);
  
  //resoudre le systeme:
  x=resoudres(l,u,B,N);
  
  //test unitaire de la solution :
  if(!EgaliteTestvect(multMatVect(a,x,N),B,N)){
    printf("Erreur de solution : AX!= B\n");
    return EXIT_FAILURE;
  }
  
  printf("solution X:\n");
  printVect(x,N);

  return 0;
}
