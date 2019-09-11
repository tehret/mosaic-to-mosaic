// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2018, Thibaud Briand <thibaud.briand@enpc.fr>
// Copyright (C) 2015, Javier Sánchez Pérez <jsanchez@ulpgc.es>
// Copyright (C) 2014, Nelson Monzón López  <nmonzon@ctim.es>
// All rights reserved.

#include "mask.h"
#include <math.h>
#include <stdio.h>
#include <assert.h>

/** Macro to get the number of elements in a static array */
#define NUMEL(x)    (sizeof(x)/sizeof(*(x)))

struct gradientStruct
{
    /** prefilter **/
    double *k;
    /** differentiator */
    double *d;
    /** size **/
    int size;
};

/* Definition of the gradient estimators */
//Central
static double kCentral[3] = {0.0, 1.0, 0.0};
static double dCentral[3] = {-0.5, 0.0, 0.5};

//Hypomode
static double kHypomode[2] = {0.5, 0.5};
static double dHypomode[2] = {-1 , 1};

//Farid 3x3
static double kFarid3[3] = {0.229879, 0.540242, 0.229879};
static double dFarid3[3] = {-0.455271, 0.0, 0.455271};

//Farid 5x5
static double kFarid5[5] = {0.037659, 0.249153, 0.426375, 0.249153, 0.037659};
static double dFarid5[5] = {-0.109604, -0.276691, 0, 0.276691, 0.109604};

//Gaussian sigma = 0.3
static double kGaussian3[3] = {0.003865, 0.999990, 0.003865};
static double dGaussian3[3] = {-0.707110, 0.0, 0.707110};

//Gaussian sigma = 0.6
static double kGaussian5[5] = {0.003645, 0.235160, 0.943070, 0.235160, 0.003645};
static double dGaussian5[5] = {-0.021915,-0.706770, 0, 0.706770, 0.021915};

//Store the gradients in a table
static gradientStruct gradientTable[] = {
    {kCentral,   dCentral,   3},
    {kHypomode,  dHypomode,  2},
    {kFarid3,    dFarid3,    3},
    {kFarid5,    dFarid5,    5},
    {kGaussian3, dGaussian3, 3},
    {kGaussian5, dGaussian5, 5}
};


/**
 *
 * Compute the gradient with central differences
 *
 */
void
gradient (double *input,        //input image
          double *dx,           //computed x derivative
          double *dy,           //computed y derivative
          int nx,               //image width
          int ny,               //image height
          int nz                //number of color channels in the image
  )
{
  int nx_rgb = nx * nz;

  for (int index_color = 0; index_color < nz; index_color++)
    {
      //gradient in the center body of the image
      for (int i = 1; i < ny - 1; i++)
        {
          for (int j = 1; j < nx - 1; j++)
            {
              int k = (i * nx + j) * nz + index_color;

              dx[k] = 0.5 * (input[k + nz] - input[k - nz]);
              dy[k] = 0.5 * (input[k + nx_rgb] - input[k - nx_rgb]);
            }
        }

      //gradient in the first and last rows
      for (int j = 1; j < nx - 1; j++)
        {
          int index = j * nz + index_color;

          dx[index] = 0.5 * (input[index + nz] - input[index - nz]);
          dy[index] = 0.5 * (input[index + nx_rgb] - input[index]);

          int k = ((ny - 1) * nx + j) * nz + index_color;

          dx[k] = 0.5 * (input[k + nz] - input[k - nz]);
          dy[k] = 0.5 * (input[k] - input[k - nx_rgb]);
        }

      //gradient in the first and last columns
      for (int i = 1; i < ny - 1; i++)
        {
          int p = (i * nx_rgb) + index_color;

          dx[p] = 0.5 * (input[p + nz] - input[p]);
          dy[p] = 0.5 * (input[p + nx_rgb] - input[p - nx_rgb]);

          int k = ((i + 1) * nx - 1) * nz + index_color;

          dx[k] = 0.5 * (input[k] - input[k - nz]);
          dy[k] = 0.5 * (input[k + nx_rgb] - input[k - nx_rgb]);
        }

      //calculate the gradient in the corners
      dx[index_color] = 0.5 * (input[index_color + nz] - input[index_color]);
      dy[index_color] =
        0.5 * (input[nx_rgb + index_color] - input[index_color]);

      int corner_up_right = (nx - 1) * nz + index_color;

      dx[corner_up_right] =
        0.5 * (input[corner_up_right] - input[corner_up_right - nz]);
      dy[corner_up_right] =
        0.5 * (input[(2 * nx_rgb) + index_color - nz] -
               input[corner_up_right]);

      int corner_down_left = ((ny - 1) * nx) * nz + index_color;

      dx[corner_down_left] =
        0.5 * (input[corner_down_left + nz] - input[corner_down_left]);
      dy[corner_down_left] =
        0.5 * (input[corner_down_left] -
               input[(ny - 2) * nx_rgb + index_color]);

      int corner_down_right = ny * nx_rgb - nz + index_color;

      dx[corner_down_right] =
        0.5 * (input[corner_down_right] - input[corner_down_right - nz]);
      dy[corner_down_right] =
        0.5 * (input[corner_down_right] -
               input[(ny - 1) * nx_rgb - nz + index_color]);
    }
}

/**
 *
 * Convolution of the rows of an image with a kernel
 *
 */
static void
convolution_rows (
  double *I,      //input/output image
  int xdim,       //image width
  int ydim,       //image height
  int zdim,       //number of color channels in the image
  double *kernel, //kernel
  int kdim,       //kernel length
  int bc          //boundary condition
  )
{
  int i, j, k;

  // kernel is of the form [ lpadding values, center, rpadding values ]
  int kcenter = (kdim - 1)/2;
  int lpadding = kcenter;
  int rpadding = kdim/2;

  // buffer taking into account the boundary condition
  int Bdim = lpadding + xdim + rpadding;
  double *B = new double[Bdim];

  // position of the boundaries in the buffer
  int bdx = xdim + lpadding;

  //Loop for every channel
  for(int index_color = 0; index_color < zdim; index_color++){

    //convolution of each line of the input image
    for (k = 0; k < ydim; k++) {
      // construct buffer for the line k
      for (i = lpadding; i < bdx; i++)
        B[i] = I[(k * xdim + i - lpadding) * zdim + index_color];
      switch (bc)
        {
        case 0: //Dirichlet boundary conditions
          for (i = 0; i < lpadding; i++)
            B[i] = 0;
          for (j = bdx; j < Bdim; j++)
            B[j] = 0;
          break;
        case 1: //Reflecting boundary conditions (wsym)
          for (i = 0; i < lpadding; i++)
              B[i] = I[(k * xdim + lpadding - i ) * zdim + index_color];
          for (j = bdx; j < Bdim; j++)
              B[j] = I[(k * xdim + xdim + bdx - j - 2) * zdim + index_color ];
          break;
        case 2: //Periodic boundary conditions
          for (i = 0; i < lpadding; i++)
              B[i] = I[(k * xdim + xdim - lpadding + i) * zdim + index_color];
          for (j = bdx; j < Bdim; j++)
              B[j] = I[(k * xdim + j - bdx) * zdim + index_color];
          break;
        }

      // convolution of the line k
      for (i = lpadding; i < bdx; i++) {
          double sum = 0;
          for (int j = 0; j < kdim; j++)
            sum += B[i-lpadding+j]*kernel[j];

          // update I
          I[(k * xdim + i - lpadding) * zdim + index_color] = sum;
      }
    }
  }

  delete[]B;
}

/**
 *
 * Convolution of the columns of an image with a kernel
 *
 */
static void
convolution_columns (
  double *I,      //input/output image
  int xdim,       //image width
  int ydim,       //image height
  int zdim,       //number of color channels in the image
  double *kernel, //kernel
  int kdim,       //kernel length
  int bc          //boundary condition
  )
{
  int i, j, k;

  // kernel is of the form [ lpadding values, center, rpadding values ]
  int kcenter = (kdim - 1)/2;
  int lpadding = kcenter;
  int rpadding = kdim/2;

  // buffer taking into account the boundary condition
  int Bdim = lpadding + ydim + rpadding;
  double *B = new double[Bdim];

  // position of the boundaries in the buffer
  int bdy = ydim + lpadding;

  //Loop for every channel
  for(int index_color = 0; index_color < zdim; index_color++){

    //convolution of each column of the input image
    for (k = 0; k < xdim; k++) {
      // construct buffer for the column k
      for (i = lpadding; i < bdy; i++)
        B[i] = I[((i - lpadding) * xdim + k) * zdim + index_color];
      switch (bc)
        {
        case 0: //Dirichlet boundary conditions
          for (i = 0; i < lpadding; i++)
            B[i] = 0;
          for (j = bdy; j < Bdim; j++)
            B[j] = 0;
          break;
        case 1: //Reflecting boundary conditions
          for (i = 0; i < lpadding; i++)
              B[i] = I[((lpadding - i) * xdim + k ) * zdim + index_color];
          for (j = bdy; j < Bdim; j++)
              B[j] = I[((bdy + ydim - j - 2) * xdim + k) * zdim + index_color ];
          break;
        case 2: //Periodic boundary conditions
          for (i = 0; i < lpadding; i++)
              B[i] = I[((ydim - lpadding + i) * xdim + k) * zdim + index_color];
          for (j = bdy; j < Bdim; j++)
              B[j] = I[((j - bdy) * xdim + k) * zdim + index_color];
          break;
        }

      // convolution of the line k
      for (i = lpadding; i < bdy; i++) {
          double sum = 0;
          for (int j = 0; j < kdim; j++)
            sum += B[i-lpadding+j]*kernel[j];

          // update I
          I[((i - lpadding) * xdim + k) * zdim + index_color] = sum;
      }
    }
  }

  delete[]B;
}

/**
 *
 * Compute the gradient estimator
 * dx = d * k^t * I
 * dy = k * d^t * I
 * where * denotes the convolution operator
 *
 */
void
gradient_robust (double *input, //input image
          double *dx,           //computed x derivative
          double *dy,           //computed y derivative
          int nx,               //image width
          int ny,               //image height
          int nz,               //number of color channels in the image
          int gradientType      //type of gradient
  )
{
  //kernel definition
  assert(gradientType < (int) NUMEL(gradientTable));
  double *kernel = gradientTable[gradientType].k;
  double *differentiator = gradientTable[gradientType].d;
  int nkernel = gradientTable[gradientType].size;
  int bc = 1;

  //initialization
  for(int i = 0; i < nx*ny*nz; i++)
    dx[i] = dy[i] = input[i];

  //x derivative computation
    //convolution of each column (k^t * I)
    convolution_columns(dx, nx, ny , nz, kernel, nkernel, bc);

    //convolution of each line (d * (k^t * I))
    convolution_rows(dx, nx, ny , nz, differentiator, nkernel, bc);

  //y derivative computation
    //convolution of each column (d^t * I)
    convolution_columns(dy, nx, ny , nz, differentiator, nkernel, bc);

    //convolution of each line (k * (d^t * dx))
    convolution_rows(dy, nx, ny , nz, kernel, nkernel, bc);
}

/**
 *
 * Prefiltering of an image compatible with the gradient
 * I <-- k * k^t * I
 * where * denotes the convolution operator
 *
 */
void
prefiltering_robust (
          double *I,            //input/output image
          int nx,               //image width
          int ny,               //image height
          int nz,               //number of color channels in the image
          int gradientType      //type of gradient
)
{
  // kernel definition
  assert(gradientType < (int) NUMEL(gradientTable));
  double *kernel = gradientTable[gradientType].k;
  int nkernel = gradientTable[gradientType].size;
  int bc = 1;

  //convolution of each line of the input image
  convolution_rows(I, nx, ny , nz, kernel, nkernel, bc);

  //convolution of each column of the input image
  convolution_columns(I, nx, ny , nz, kernel, nkernel, bc);
}

/**
 *
 * Convolution with a Gaussian kernel
 *
 */
void
gaussian (
  double *I,    //input/output image
  int xdim,     //image width
  int ydim,     //image height
  int zdim,     //number of color channels in the image
  double sigma, //Gaussian sigma
  int bc,       //boundary condition
  int precision //defines the size of the window
)
{
  int i, j, k;

  double den = 2 * sigma * sigma;
  int size = (int) (precision * sigma) + 1;
  int bdx = xdim + size;
  int bdy = ydim + size;

  if (bc && size > xdim){
      printf("GaussianSmooth: sigma too large for this bc\n");
      throw 1;
  }

  //compute the coefficients of the 1D convolution kernel
  double *B = new double[size];
  for (int i = 0; i < size; i++)
    B[i] = 1 / (sigma * sqrt (2.0 * 3.1415926)) * exp (-i * i / den);

  double norm = 0;

  //normalize the 1D convolution kernel
  for (int i = 0; i < size; i++)
    norm += B[i];

  norm *= 2;
  norm -= B[0];

  for (int i = 0; i < size; i++)
    B[i] /= norm;

  double *R = new double[size + xdim + size];
  double *T = new double[size + ydim + size];

  //Loop for every channel
  for(int index_color = 0; index_color < zdim; index_color++){

  //convolution of each line of the input image
   for (k = 0; k < ydim; k++)
    {
      for (i = size; i < bdx; i++)
        R[i] = I[(k * xdim + i - size) * zdim + index_color];
      switch (bc)
        {
        case 0: //Dirichlet boundary conditions

          for (i = 0, j = bdx; i < size; i++, j++)
            R[i] = R[j] = 0;
          break;
        case 1: //Reflecting boundary conditions
          for (i = 0, j = bdx; i < size; i++, j++)
            {
              R[i] = I[(k * xdim + size - i ) * zdim + index_color];
              R[j] = I[(k * xdim + xdim - i - 1) * zdim + index_color ];
            }
          break;
        case 2: //Periodic boundary conditions
          for (i = 0, j = bdx; i < size; i++, j++)
            {
              R[i] = I[(k * xdim + xdim - size + i) * zdim + index_color];
              R[j] = I[(k * xdim + i) * zdim + index_color];
            }
          break;
        }

      for (i = size; i < bdx; i++)
        {

          double sum = B[0] * R[i];

          for (int j = 1; j < size; j++)
            sum += B[j] * (R[i - j] + R[i + j]);

          I[(k * xdim + i - size) * zdim + index_color] = sum;

        }
    }

  //convolution of each column of the input image
  for (k = 0; k < xdim; k++)
    {
      for (i = size; i < bdy; i++)
        T[i] = I[((i - size) * xdim + k) * zdim + index_color];

      switch (bc)
        {
        case 0: // Dirichlet boundary conditions
          for (i = 0, j = bdy; i < size; i++, j++)
            T[i] = T[j] = 0;
          break;
        case 1: // Reflecting boundary conditions
          for (i = 0, j = bdy; i < size; i++, j++)
            {
              T[i] = I[((size - i) * xdim + k) * zdim + index_color];
              T[j] = I[((ydim - i - 1) * xdim + k) * zdim + index_color];
            }
          break;
        case 2: // Periodic boundary conditions
          for (i = 0, j = bdx; i < size; i++, j++)
            {
              T[i] = I[((ydim - size + i) * xdim + k) * zdim + index_color];
              T[j] = I[(i * xdim + k) * zdim + index_color];
            }
          break;
        }

      for (i = size; i < bdy; i++)
        {
          double sum = B[0] * T[i];

          for (j = 1; j < size; j++)
            sum += B[j] * (T[i - j] + T[i + j]);

          I[((i - size) * xdim + k) * zdim + index_color] = sum;
        }
    }
  }

  delete[]B;
  delete[]R;
  delete[]T;
}
