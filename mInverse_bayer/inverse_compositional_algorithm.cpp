// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2018, Thibaud Briand <thibaud.briand@enpc.fr>
// Copyright (C) 2015, Javier Sánchez Pérez <jsanchez@dis.ulpgc.es>
// All rights reserved.

/**
  *
  *  This code implements the 'modified inverse compositional algorithm'.
  *
  *  The 'inverse compositional algorithm' was proposed in
  *     [1] S. Baker, and I. Matthews. (2004). Lucas-kanade 20 years on: A
  *         unifying framework. International Journal of Computer Vision,
  *         56(3), 221-255.
  *     [2] S. Baker, R. Gross, I. Matthews, and T. Ishikawa. (2004).
  *         Lucas-kanade 20 years on: A unifying framework: Part 2.
  *         International Journal of Computer Vision, 56(3), 221-255.
  *
  *  This implementation is for color images. It calculates the global
  *  transform between two images. It uses robust error functions and a
  *  coarse-to-fine strategy for computing large displacements
  *
**/

#include <stdlib.h>
#include <cmath>
#include <stdio.h>

#include "bicubic_interpolation.h"
#include "inverse_compositional_algorithm.h"
#include "matrix.h"
#include "mask.h"
#include "transformation.h"
#include "zoom.h"

/**
 *
 *  Derivative of robust error functions
 *
 */
double rhop(
  double t2,     //squared difference of both images
  double lambda, //robust threshold
  int    type    //choice of the robust error function
)
{
  double result=0.0;
  double lambda2=lambda*lambda;
  switch(type)
  {
    case QUADRATIC:
      result=1;
      break;
    default:
    case TRUNCATED_QUADRATIC:
      if(t2<lambda2) result=1.0;
      else result=0.0;
      break;
    case GEMAN_MCCLURE:
      result=lambda2/((lambda2+t2)*(lambda2+t2));
      break;
    case LORENTZIAN:
      result=1/(lambda2+t2);
      break;
    case CHARBONNIER:
      result=1.0/(sqrt(t2+lambda2));
      break;
  }
  return result;
}

/**
 *
 *  Function to compute DI^t*J
 *  from the gradient of the image and the Jacobian
 *
 */
static void steepest_descent_images
(
  double *Ix,  //x derivate of the image
  double *Iy,  //y derivate of the image
  double *J,   //Jacobian matrix
  double *G,   //output DI^t*J
  int nparams, //number of parameters
  int nx,      //number of columns
  int ny,      //number of rows
  int nz       //number of channels
)
{
  int k=0;

  for(int p=0; p<nx*ny; p++)
      for(int c=0; c<nz; c++)
        for(int n=0; n<nparams; n++)
          G[k++]=Ix[p*nz+c]*J[2*p*nparams+n]+
                   Iy[p*nz+c]*J[2*p*nparams+n+nparams];
}

/**
 *
 *  Function to compute the Hessian matrix
 *  the Hessian is equal to G^t*G
 *
 */
static void hessian
(
  double *G,   //the steepest descent image
  double *H,   //output Hessian matrix
  int nparams, //number of parameters
  int nx,      //number of columns
  int ny,      //number of rows
  int nz       //number of channels
)
{
  //initialize the hessian to zero
  for(int k=0; k<nparams*nparams; k++)
    H[k] = 0;

  //calculate the hessian in a neighbor window
  for(int p=0; p<nx*ny; p++) {
        if ( std::isfinite(G[p*nz*nparams]) ) //Discarded if NAN
          AtA(&(G[p*nz*nparams]), H, nz, nparams);
  }
}

#ifdef PRECOMPUTATION_GTG
/**
 *
 *  Function to compute G^T G
 *  from G
 *
 */
static void precomputation_hessian
(
  double *G,   //input G
  double *GTG, //output G^T G
  int nparams, //number of parameters
  int nx,      //number of columns
  int ny,      //number of rows
  int nz       //number of channels
)
{
  for(int k=0; k<nparams*nparams*nx*ny; k++)
    GTG[k] = 0;

  //calculate the hessian in a neighbor window
  for(int p=0; p<nx*ny; p++)
        AtA(&(G[p*nz*nparams]), &(GTG[p*nparams*nparams]), nz, nparams);
}

/**
 *
 *  Function to compute the Hessian matrix
 *  the Hessian is equal to rho'*(G^T G)
 *
 */
static void compute_hessian(
  double *GTG, //precomputed matrix G^T G
  double *rho, //robust weights
  double *H,   //output Hessian matrix
  int nparams, //number of parameters
  int nx,      //number of rows
  int ny       //number of columns
)
{
  //initialize the hessian to zero
  for(int k=0; k<nparams*nparams; k++)
    H[k] = 0;

  //calculate the hessian in a neighbor window
  for(int p=0; p<nx*ny; p++) {
      //Discarded if NAN
      if ( std::isfinite(rho[p]) && std::isfinite(GTG[p*nparams*nparams]))
        sA(rho[p], &(GTG[p*nparams*nparams]), H, nparams);
  }
}
#else
/**
 *
 *  Function to compute the Hessian matrix with robust error functions
 *  the Hessian is equal to rho'*G^t*G
 *
 */
static void hessian
(
  double *G,   //the steepest descent image
  double *rho, //robust weights
  double *H,   //output Hessian matrix
  int nparams, //number of parameters
  int nx,      //number of columns
  int ny,      //number of rows
  int nz       //number of channels
)
{
  //initialize the hessian to zero
  for(int k=0; k<nparams*nparams; k++)
    H[k] = 0;

  //calculate the hessian in a neighbor window
  for(int p=0; p<nx*ny; p++) {
      //Discarded if NAN
      if ( std::isfinite(rho[p]) && std::isfinite(G[p*nz*nparams]))
        sAtA(rho[p], &(G[p*nz*nparams]), H, nz, nparams);
  }
}
#endif

/**
 *
 *  Function to compute the inverse of the Hessian
 *
 */
static void inverse_hessian
(
  double *H,   //input Hessian
  double *H_1, //output inverse Hessian
  int nparams  //number of parameters
)
{
  if(inverse(H, H_1, nparams)==-1)
    //if the matrix is not invertible, set parameters to 0
    for(int i=0; i<nparams*nparams; i++) H_1[i]=0;
}

/**
 *
 *  Function to compute I2(W(x;p))-I1(x)
 *
 */
static void difference_image
(
  double *I,  //second warped image I2(x'(x;p))
  double *Iw, //first image I1(x)
  double *DI, //output difference array
  int nx,     //number of columns
  int ny,     //number of rows
  int nz      //number of channels
)
{
  for(int i=0; i<nx*ny*nz; i++)
    DI[i]=Iw[i]-I[i];
}

/**
 *
 *  Function to store the values of p'((I2(x'(x;p))-I1(x))^2)
 *
 */
static void robust_error_function
(
  double *DI,    //input difference array
  double *rho,   //output robust weights
  double lambda, //threshold used in the robust functions
  int    type,   //choice of robust error function
  int nx,        //number of columns
  int ny,        //number of rows
  int nz         //number of channels
)
{
    for(int p=0; p<nx*ny; p++) {
      if ( DI[p*nz+0] == NAN)
        rho[p] = NAN; // Already discarded for I2
      else {
        double norm=0.0;
        for(int c=0;c<nz;c++) norm+=DI[p*nz+c]*DI[p*nz+c];
        rho[p]=rhop(norm,lambda,type);
      }
    }
}

/**
 *
 *  Function to compute b=Sum(G^t * DI)
 *
 */
static void independent_vector
(
  double *G,   //the steepest descent image
  double *DI,  //I2(x'(x;p))-I1(x)
  double *b,   //output independent vector
  int nparams, //number of parameters
  int nx,      //number of columns
  int ny,      //number of rows
  int nz       //number of channels
)
{
  //initialize the vector to zero
  for(int k=0; k<nparams; k++)
    b[k]=0;

  for(int p=0; p<nx*ny; p++) {
      //Discard if NAN
      if ( std::isfinite(G[p*nparams*nz]) && std::isfinite(DI[p*nz]) )
        Atb(&(G[p*nparams*nz]), &(DI[p*nz]), b, nz, nparams);
    }
}

/**
 *
 *  Function to compute b=Sum(rho'*G^t * DI)
 *  with robust error functions
 *
 */
static void independent_vector
(
  double *G,   //the steepest descent image
  double *DI,  //I2(x'(x;p))-I1(x)
  double *rho, //robust function
  double *b,   //output independent vector
  int nparams, //number of parameters
  int nx,      //number of columns
  int ny,      //number of rows
  int nz       //number of channels
)
{
  //initialize the vector to zero
  for(int k=0; k<nparams; k++)
    b[k]=0;

  for(int p=0; p<nx*ny; p++) {
      //Discard if NAN
      if ( std::isfinite(G[p*nparams*nz]) && std::isfinite(DI[p*nz]) )
        sAtb(rho[p], &(G[p*nparams*nz]), &(DI[p*nz]), b, nz, nparams);
   }
}

/**
 *
 *  Function to solve for dp
 *
 */
static double parametric_solve
(
  double *H_1, //inverse Hessian
  double *b,   //independent vector
  double *dp,  //output parameters increment
  int nparams  //number of parameters
)
{
  double error=0.0;
  Axb(H_1, b, dp, nparams);
  for(int i=0; i<nparams; i++) error+=dp[i]*dp[i];
  return sqrt(error);
}

/**
  *
  *  Inverse compositional algorithm
  *  Quadratic version - L2 norm
  *
  *
**/
void inverse_compositional_algorithm(
  double *I1,        //first image
  double *I2,        //second image
  double *p,         //parameters of the transform (output)
  int nparams,       //number of parameters of the transform
  int nx,            //number of columns of the image
  int ny,            //number of rows of the image
  int nz,            //number of channels of the images
  double TOL,        //Tolerance used for the convergence in the iterations
  int nanifoutside,  //parameter for discarding boundary pixels
  int delta,         //distance to the boundary
  int type_gradient, //type of gradient
  int verbose        //enable verbose mode
)
{
  int size1=nx*ny*nz;        //size of the image with channels
  int size2=size1*nparams;   //size of the image with transform parameters
  int size3=nparams*nparams; //size for the Hessian
  int size4=2*nx*ny*nparams; //size for the Jacobian

  double *Ix =new double[size1];   //x derivate of the first image
  double *Iy =new double[size1];   //y derivate of the first image
  double *J  =new double[size4];   //jacobian matrix for all points
  double *G  =new double[size2];   //steepest descent images
  double *H  =new double[size3];   //Hessian matrix
  double *H_1=new double[size3];   //inverse Hessian matrix

  //Evaluate the gradient of I1
  //Do not prefilter if central differences are used
  if ( type_gradient )
    gradient_robust(I1, Ix, Iy, nx, ny, nz, type_gradient);
  else
    gradient(I1, Ix, Iy, nx, ny, nz);

  //Discard boundary pixels
  if ( nanifoutside && delta) {
    for (int i = 0; i < ny; i++) {
        for( int j = 0; j < nx; j++) {
            if ( i < delta || i > ny-1-delta || j < delta || j > nx - 1 - delta) {
                for (int index_color = 0; index_color < nz; index_color++) {
                    int k = (i * nx + j) * nz + index_color;
                    Ix[k] = NAN;
                    Iy[k] = NAN;
                }
            }
        }
    }
  }

  //Prefiltering of the images before the loop
  if ( type_gradient ) {
    prefiltering_robust(I1, nx, ny, nz, type_gradient);
    prefiltering_robust(I2, nx, ny, nz, type_gradient);
  }

  //Evaluate the Jacobian
  jacobian(J, nparams, nx, ny);

  //Compute the steepest descent images
  steepest_descent_images(Ix, Iy, J, G, nparams, nx, ny, nz);

  //Compute the Hessian matrix
  hessian(G, H, nparams, nx, ny, nz);
  inverse_hessian(H, H_1, nparams);

  //delete allocated memory
  delete []Ix;
  delete []Iy;
  delete []J;

  // memory allocation for the iteration
  double *Iw =new double[size1];   //warp of the second image/
  double *DI =new double[size1];   //error image (I2(w)-I1)
  double *dp =new double[nparams]; //incremental solution
  double *b  =new double[nparams]; //steepest descent images

  //Iterate
  double error=1E10;
  int niter=0;

  do{
    //Warp image I2
    bicubic_interpolation(I2, Iw, p, nparams, nx, ny, nz, delta, nanifoutside);

    //Compute the error image (I1-I2w)
    difference_image(I1, Iw, DI, nx, ny, nz);

    //Compute the independent vector
    independent_vector(G, DI, b, nparams, nx, ny, nz);

    //Solve equation and compute increment of the motion
    error=parametric_solve(H_1, b, dp, nparams);

    //Update the warp x'(x;p) := x'(x;p) * x'(x;dp)^-1
    update_transform(p, dp, nparams);

    if(verbose)
    {
      printf("|Dp|=%f: p=(",error);
      for(int i=0;i<nparams-1;i++)
        printf("%f ",p[i]);
      printf("%f)\n",p[nparams-1]);
    }
    niter++;
  }
  while(error>TOL && niter<MAX_ITER);

  //delete allocated memory
  delete []G;
  delete []Iw;
  delete []dp;
  delete []b;
  delete []H;
  delete []H_1;
}

/**
  *
  *  Inverse compositional algorithm
  *  Version with robust error functions
  *
**/
void robust_inverse_compositional_algorithm(
  double *I1,        //first image
  double *I2,        //second image
  double *p,         //parameters of the transform (output)
  int nparams,       //number of parameters of the transform
  int nx,            //number of columns of the image
  int ny,            //number of rows of the image
  int nz,            //number of channels of the images
  double TOL,        //Tolerance used for the convergence in the iterations
  int    robust,     //robust error function
  double lambda,     //parameter of robust error function
  int nanifoutside,  //parameter for discarding boundary pixels
  int delta,         //distance to the boundary
  int type_gradient, //type of gradient
  int verbose        //enable verbose mode
)
{
  int size0=nx*ny;           //size of the image
  int size1=nx*ny*nz;        //size of the image with channels
  int size2=size1*nparams;   //size of the image with transform parameters
  int size3=nparams*nparams; //size for the Hessian
  int size4=2*nx*ny*nparams; //size for the Jacobian
  int size5=size0*size3;     //size for G^T G

  double *Ix =new double[size1];   //x derivate of the first image
  double *Iy =new double[size1];   //y derivate of the first image
  double *J  =new double[size4];   //jacobian matrix for all points
  double *G=new double[size2];     //steepest descent images

  //Evaluate the gradient of I1
  //Do not prefilter if central differences are used
  if ( type_gradient )
    gradient_robust(I1, Ix, Iy, nx, ny, nz, type_gradient);
  else
    gradient(I1, Ix, Iy, nx, ny, nz);

  //Discard boundary pixels
  if ( nanifoutside && delta) {
    for (int i = 0; i < ny; i++) {
        for( int j = 0; j < nx; j++) {
            if ( i < delta || i > ny-1-delta || j < delta || j > nx - 1 - delta) {
                for (int index_color = 0; index_color < nz; index_color++) {
                    int k = (i * nx + j) * nz + index_color;
                    Ix[k] = NAN;
                    Iy[k] = NAN;
                }
            }
        }
    }
  }

  //Prefiltering of the images before the loop
  if ( type_gradient ) {
    prefiltering_robust(I1, nx, ny, nz, type_gradient);
    prefiltering_robust(I2, nx, ny, nz, type_gradient);
  }

  //Evaluate the Jacobian
  jacobian(J, nparams, nx, ny);

  //Compute the steepest descent images
  steepest_descent_images(Ix, Iy, J, G, nparams, nx, ny, nz);

  #ifdef PRECOMPUTATION_GTG
    //Precompute G^T G
    double *GTG=new double[size5];   //G^T G
    precomputation_hessian(G, GTG, nparams, nx, ny, nz);
  #endif

  //delete allocated memory
  delete []Ix;
  delete []Iy;
  delete []J;

  //memory allocation for the iteration
  double *Iw =new double[size1];   //warp of the second image/
  double *DI =new double[size1];   //error image (I2(w)-I1)
  double *dp =new double[nparams]; //incremental solution
  double *b  =new double[nparams]; //steepest descent images
  double *H  =new double[size3];   //Hessian matrix
  double *H_1=new double[size3];   //inverse Hessian matrix
  double *rho=new double[size0];   //robust function

  //Iterate
  double error=1E10;
  int niter=0;
  double lambda_it;

  if(lambda>0) lambda_it=lambda;
  else lambda_it=LAMBDA_0;

  do{
    //Warp image I2
    bicubic_interpolation(I2, Iw, p, nparams, nx, ny, nz, delta, nanifoutside);

    //Compute the error image (I1-I2w)
    difference_image(I1, Iw, DI, nx, ny, nz);

    //compute robustifiction function
    robust_error_function(DI, rho, lambda_it, robust, nx, ny, nz);
    if(lambda<=0 && lambda_it>LAMBDA_N)
    {
      lambda_it*=LAMBDA_RATIO;
      if(lambda_it<LAMBDA_N) lambda_it=LAMBDA_N;
    }

    //Compute the independent vector
    independent_vector(G, DI, rho, b, nparams, nx, ny, nz);

    //Compute the Hessian matrix
    #ifdef PRECOMPUTATION_GTG
        compute_hessian(GTG, rho, H, nparams, nx, ny);
    #else
        hessian(G, rho, H, nparams, nx, ny, nz);
    #endif

    inverse_hessian(H, H_1, nparams);

    //Solve equation and compute increment of the motion
    error=parametric_solve(H_1, b, dp, nparams);

    //Update the warp x'(x;p) := x'(x;p) * x'(x;dp)^-1
    update_transform(p, dp, nparams);

    if(verbose)
    {
      printf("|Dp|=%f: p=(",error);
      for(int i=0;i<nparams-1;i++)
        printf("%f ",p[i]);
      printf("%f), lambda=%f\n",p[nparams-1],lambda_it);
    }
    niter++;
  }
  while(error>TOL && niter<MAX_ITER);

  //delete allocated memory
  delete []DI;
  delete []Iw;
  delete []G;
  delete []dp;
  delete []b;
  delete []H;
  delete []H_1;
  delete []rho;
  #ifdef PRECOMPUTATION_GTG
    delete []GTG;
  #endif
}

/**
  *
  *  Multiscale approach for computing the optical flow
  *
**/
void pyramidal_inverse_compositional_algorithm(
    double *I1,        //first image
    double *I2,        //second image
    double *p,         //parameters of the transform
    int nparams,       //number of parameters
    int nxx,           //image width
    int nyy,           //image height
    int nzz,           //number of color channels in image
    int nscales,       //number of scales
    double nu,         //downsampling factor
    double TOL,        //stopping criterion threshold
    int robust,        //robust error function
    double lambda,     //parameter of robust error function
    int first_scale,   //number of the first scale
    int nanifoutside,  //parameter for discarding boundary pixels
    int delta,         //distance to the boundary
    int type_gradient, //type of gradient
    bool   verbose     //switch on messages
)
{
    int size=nxx*nyy*nzz;

    double **I1s=new double*[nscales];
    double **I2s=new double*[nscales];
    double **ps =new double*[nscales];

    int *nx=new int[nscales];
    int *ny=new int[nscales];

    I1s[0]=new double[size];
    I2s[0]=new double[size];

    //copy the input images
    for(int i=0;i<size;i++)
    {
      I1s[0][i]=I1[i];
      I2s[0][i]=I2[i];
    }

    ps[0]=p;
    nx[0]=nxx;
    ny[0]=nyy;

    //create the scales
    for(int s=1; s<nscales; s++)
    {
      zoom_size(nx[s-1], ny[s-1], nx[s], ny[s], nu);

      const int size=nx[s]*ny[s];

      I1s[s]=new double[size*nzz];
      I2s[s]=new double[size*nzz];
      ps[s] =new double[nparams];

      //zoom the images from the previous scale
      zoom_out(I1s[s-1], I1s[s], nx[s-1], ny[s-1], nzz, nu);
      zoom_out(I2s[s-1], I2s[s], nx[s-1], ny[s-1], nzz, nu);
    }

    //delete allocated memory for unused scales
    for (int i=0; i<first_scale; i++) {
        delete []I1s[i];
        delete []I2s[i];
    }

    //initialization of the transformation parameters at the coarsest scale
    for(int i=0; i<nparams; i++)
        ps[nscales-1][i]=0.0;

    //pyramidal approach for computing the transformation
    for(int s=nscales-1; s>= first_scale; s--)
    {
      if(verbose) printf("Scale: %d ",s);

      //incremental refinement for this scale
      if(robust==QUADRATIC)
      {
        if(verbose) printf("(L2 norm)\n");

        inverse_compositional_algorithm(
          I1s[s], I2s[s], ps[s], nparams, nx[s],
          ny[s], nzz, TOL, nanifoutside, delta, type_gradient, verbose
        );
      }
      else
      {
        if(verbose) printf("(Robust error function %d)\n",robust);

        robust_inverse_compositional_algorithm(
          I1s[s], I2s[s], ps[s], nparams, nx[s],
          ny[s], nzz, TOL, robust, lambda, nanifoutside, delta, type_gradient, verbose);
      }

      //if it is not the finer scale, then upsample the parameters
      if(s) {
        zoom_in_parameters(
          ps[s], ps[s-1], nparams, nx[s], ny[s], nx[s-1], ny[s-1]);
        delete []ps [s];
      }

      //delete allocated memory
      delete []I1s[s];
      delete []I2s[s];
    }

    //Upsample the parameters
    if ( first_scale > 1 )
    {
        zoom_in_parameters(
          ps[first_scale - 1], ps[0], nparams,
          nx[first_scale-1], ny[first_scale - 1], nx[0], ny[0]);
    }

    for(int i=1; i<first_scale; i++)
        delete []ps [i];
    delete []I1s;
    delete []I2s;
    delete []ps;
    delete []nx;
    delete []ny;
}
