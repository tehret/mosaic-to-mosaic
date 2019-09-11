// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2018, Thibaud Briand <thibaud.briand@enpc.fr>
// Copyright (C) 2015, Javier Sánchez Pérez <jsanchez@dis.ulpgc.es>
// All rights reserved.

#ifndef TRANSFORMATION_H
#define TRANSFORMATION_H

//types of transformations
#define TRANSLATION_TRANSFORM 2
#define EUCLIDEAN_TRANSFORM   3
#define SIMILARITY_TRANSFORM  4
#define AFFINITY_TRANSFORM    6
#define HOMOGRAPHY_TRANSFORM  8

/**
 *
 *  Function to compute the Jacobian matrix
 *  These parametrizations of the Jacobian are taken from the book of Zselinski
 *  (chapter 6 and 9)
 *
 */
void jacobian
(
  double *J,   //computed Jacobian
  int nparams, //number of parameters
  int nx,      //number of columns of the image
  int ny       //number of rows of the image
);

/**
 *
 *  Function to update the current transform with the computed increment
 *  x'(x;p) = x'(x;p) o x'(x;dp)^-1
 *
 */
void update_transform
(
  double *p,  //output accumulated transform
  double *dp, //computed increment
  int nparams //number of parameters
);

/**
 *
 *  Function to transform a 2D point (x,y) through a parametric model
 *
 */
void project
(
  int x,      //x component of the 2D point
  int y,      //y component of the 2D point
  double *p,  //parameters of the transformation
  double &xp, //x component of the transformed point
  double &yp, //y component of the transformed point
  int nparams //number of parameters
);

/**
 *
 *  Function to convert a parametric model to its matrix representation
 *
 */
void params2matrix
(
  double *p,      //input parametric model
  double *matrix, //output matrix
  int nparams     //number of parameters
);

#endif
