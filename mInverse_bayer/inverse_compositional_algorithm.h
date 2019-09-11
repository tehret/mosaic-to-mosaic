// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2018, Thibaud Briand <thibaud.briand@enpc.fr>
// Copyright (C) 2015, Javier Sánchez Pérez <jsanchez@dis.ulpgc.es>
// All rights reserved.

#ifndef INVERSE_COMPOSITIONAL_ALGORITHM
#define INVERSE_COMPOSITIONAL_ALGORITHM

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

#define QUADRATIC 0
#define TRUNCATED_QUADRATIC 1
#define GEMAN_MCCLURE 2
#define LORENTZIAN 3
#define CHARBONNIER 4

#define MAX_ITER 30
#define LAMBDA_0 80
#define LAMBDA_N 5
#define LAMBDA_RATIO 0.90

#define PRECOMPUTATION_GTG //comment to remove the precomputation of GTG

/**
 *
 *  Derivative of robust error functions
 *
 */
double rhop(
  double t2,    //squared difference of both images
  double sigma, //robust threshold
  int    type   //choice of the robust error function
);

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
  double TOL,        //tolerance used for the convergence in the iterations
  int nanifoutside,  //parameter for discarding boundary pixels
  int delta,         //distance to the boundary
  int type_gradient, //type of gradient
  int verbose=0      //enable verbose mode
);

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
  int verbose=0      //enable verbose mode
);

/**
  *
  *  Multiscale approach for computing the optical flow
  *
**/
void pyramidal_inverse_compositional_algorithm(
    double *I1,       //first image
    double *I2,       //second image
    double *p,        //parameters of the transform
    int    nparams,   //number of parameters
    int    nxx,       //image width
    int    nyy,       //image height
    int    nzz,       //number of color channels in image
    int    nscales,   //number of scales
    double nu,        //downsampling factor
    double TOL,       //stopping criterion threshold
    int    robust,    //robust error function
    double lambda,    //parameter of robust error function
    int first_scale,  //number of the first scale
   int nanifoutside,  //parameter for discarding boundary pixels
   int delta,         //distance to the boundary
   int type_gradient, //type of gradient
    bool   verbose    //switch on messages
);

#endif
