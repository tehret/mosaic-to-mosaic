// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2018, Thibaud Briand <thibaud.briand@enpc.fr>
// Copyright (C) 2015, Javier Sánchez Pérez <jsanchez@dis.ulpgc.es>
// Copyright (C) 2014, Nelson Monzón López  <nmonzon@ctim.es>
// All rights reserved.

#ifndef MASK_H
#define MASK_H

/**
 *
 * Compute the gradient with central differences
 *
 */
void gradient(
  double *input,  //input image
  double *dx,     //computed x derivative
  double *dy,     //computed y derivative
  int nx,         //image width
  int ny,         //image height
  int nz          //number of color channels in the image
);

/**
 *
 * Compute the gradient estimator
 * dx = d * k^t * I
 * dy = k * d^t * I
 * where * denotes the convolution operator
 *
 */
void gradient_robust (double *input,        //input image
          double *dx,           //computed x derivative
          double *dy,           //computed y derivative
          int nx,               //image width
          int ny,               //image height
          int nz,               //number of color channels in the image
          int gradientType      //type of gradient
);

/**
 *
 * Prefiltering of an image compatible with the gradient
 * I <-- k * k^t * I
 * where * denotes the convolution operator
 *
 */
void prefiltering_robust (
          double *I,            //input/output image
          int nx,               //image width
          int ny,               //image height
          int nz,               //number of color channels in the image
          int gradientType      //type of gradient
);

/**
 *
 * Convolution with a Gaussian kernel
 *
 */
void gaussian (
  double *I,        //input/output image
  int xdim,         //image width
  int ydim,         //image height
  int zdim,         //number of color channels in the image
  double sigma,     //Gaussian sigma
  int bc = 1,       //boundary condition
  int precision = 5 //defines the size of the window
);

#endif
