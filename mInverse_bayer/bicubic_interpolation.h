// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2018, Thibaud Briand <thibaud.briand@enpc.fr>
// Copyright (C) 2015, Javier Sánchez Pérez <jsanchez@ulpgc.es>
// Copyright (C) 2014, Nelson Monzón López <nmonzon@ctim.es>
// All rights reserved.

#ifndef BICUBIC_INTERPOLATION_H
#define BICUBIC_INTERPOLATION_H

double
bicubic_interpolation(
  double *input, //image to be interpolated
  double uu,     //x component of the vector field
  double vv,     //y component of the vector field
  int nx,        //width of the image
  int ny,        //height of the image
  int nz,        //number of channels of the image
  int k          //actual channel
);

/**
  *
  * Compute the bicubic interpolation of an image from a parametric trasform
  *
**/
void bicubic_interpolation(
  double *input,   //image to be warped
  double *output,  //warped output image with bicubic interpolation
  double *params,  //x component of the vector field
  int nparams,     //number of parameters of the transform
  int nx,          //width of the image
  int ny,          //height of the image
  int nz,          //number of channels of the image
  int delta,       //distance to the boundary
  int nanifoutside //parameter for discarding boudary pixels
);

#endif
