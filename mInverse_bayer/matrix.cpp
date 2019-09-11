// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2018, Thibaud Briand <thibaud.briand@enpc.fr>
// Copyright (C) 2015, Javier Sánchez Pérez <jsanchez@dis.ulpgc.es>
// All rights reserved.

#include "matrix.h"
#include <math.h>

//Multiplication of a square matrix and a vector
void Axb(double *A, double *b, double *p, int n)
{
  for(int i=0; i<n; i++)
  {
    double sum=0;
    for(int j=0; j<n; j++)
      sum+=A[i*n+j]*b[j];

    p[i]=sum;
  }
}

//Multiplication of the transpose of a matrix and a vector
//p should be initialized to zero outside
void Atb(double *A, double *b, double *p, int n, int m)
{
  for(int i=0; i<m; i++) {
    double sum = 0;
    for(int j=0; j<n; j++)
      sum +=A[j*m+i]*b[j];
    p[i] += sum;
  }
}

//Multiplication of the transpose of a matrix, a vector and a scalar
//p should be initialized to zero outside
void sAtb(double s, double *A, double *b, double *p, int n, int m)
{
  for(int i=0; i<m; i++) {
    double sum = 0;
    for(int j=0; j<n; j++)
      sum += A[j*m+i]*b[j];
    p[i] += s*sum;
  }
}

//Multiplication of the transpose of a matrix and itself
//B should be initialized to zero outside
void AtA(double *A, double *B, int n, int m)
{
  for(int i=0; i<m; i++)
    for(int j=0; j<m; j++) {
      double sum = 0;
      for(int k=0; k<n; k++)
        sum += A[k*m+i]*A[k*m+j];
      B[i*m+j] += sum;
    }
}

//Multiplication of the transpose of a matrix and itself with a scalar
//B should be initialized to zero outside
void sAtA(double s, double *A, double *B, int n, int m)
{
  for(int i=0; i<m; i++)
    for(int j=0; j<m; j++) {
      double sum = 0;
      for(int k=0; k<n; k++)
        sum += A[k*m+i]*A[k*m+j];
      B[i*m+j] += s*sum;
    }
}

//Multiplication with a scalar
//B should be initialized to zero outside
void sA(double s, double *A, double *B, int m)
{
  for(int i=0; i<m*m; i++)
    B[i] += s*A[i];
}

//Function to compute the inverse of a matrix
//through Gaussian elimination
int inverse(
  double *A,   //input matrix
  double *A_1, //output matrix
  int N        //matrix dimension
)
{
  double *PASO=new double[2*N*N];

  double max,paso,mul;
  int i,j,i_max,k;

  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
      PASO[i*2*N+j]=A[i*N+j];
      PASO[i*2*N+j+N]=0.;
    }
  }
  for(i=0;i<N;i++)
      PASO[i*2*N+i+N]=1.;

  for(i=0;i<N;i++){
    max=fabs(PASO[i*2*N+i]);
    i_max=i;
    for(j=i;j<N;j++){
       if(fabs(PASO[j*2*N+i])>max){
         i_max=j; max=fabs(PASO[j*2*N+i]);
       }
    }

    if(max<10e-30){
      delete []PASO;
      return -1;
    }
    if(i_max>i){
      for(k=0;k<2*N;k++){
        paso=PASO[i*2*N+k];
        PASO[i*2*N+k]=PASO[i_max*2*N+k];
        PASO[i_max*2*N+k]=paso;
      }
    }

    for(j=i+1;j<N;j++){
      mul=-PASO[j*2*N+i]/PASO[i*2*N+i];
      for(k=i;k<2*N;k++) PASO[j*2*N+k]+=mul*PASO[i*2*N+k];
    }
  }

  if(fabs(PASO[(N-1)*2*N+N-1])<10e-30){
      delete []PASO;
      return -1;
  }

  for(i=N-1;i>0;i--){
    for(j=i-1;j>=0;j--){
      mul=-PASO[j*2*N+i]/PASO[i*2*N+i];
      for(k=i;k<2*N;k++) PASO[j*2*N+k]+=mul*PASO[i*2*N+k];
    }
  }
  for(i=0;i<N;i++)
    for(j=N;j<2*N;j++)
      PASO[i*2*N+j]/=PASO[i*2*N+i];

  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      A_1[i*N+j]=PASO[i*2*N+j+N];

  delete []PASO;

  return 0;
}
