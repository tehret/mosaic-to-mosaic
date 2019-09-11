// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2018, Thibaud Briand <thibaud.briand@enpc.fr>
// Copyright (C) 2015, Javier Sánchez Pérez <jsanchez@ulpgc.es>
// Copyright (C) 2014, Nelson Monzón López <nmonzon@ctim.es>
// All rights reserved.

#include "bicubic_interpolation.h"
#include "transformation.h"
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

/**
  *
  * Neumann boundary condition test
  *
**/
static int
neumann_bc(
  int x, //index
  int nx //size
)
{
  if (x<0)
      x = 0;
  else if (x >= nx)
      x = nx - 1;
  return x;
}

/**
  *
  * Bicubic interpolation in one dimension
  *
**/
static double
cubic_interpolation(
  double v[4], //interpolation points
  double x     //point to be interpolated
)
{
  return v[1] + 0.5 * x * (v[2] - v[0]
                           + x * (2.0 * v[0] - 5.0 * v[1] + 4.0 * v[2] - v[3]
                                  + x * (3.0 * (v[1] - v[2]) + v[3] - v[0])));
}

/**
  *
  * Bicubic interpolation in two dimension
  *
**/
static double
bicubic_interpolation(
  double p[4][4], //array containing the interpolation points
  double x,       //x position to be interpolated
  double y        //y position to be interpolated
)
{
  double v[4];
  v[0] = cubic_interpolation (p[0], y);
  v[1] = cubic_interpolation (p[1], y);
  v[2] = cubic_interpolation (p[2], y);
  v[3] = cubic_interpolation (p[3], y);
  return cubic_interpolation (v, x);
}

/**
  *
  * Compute the bicubic interpolation of a point in an image.
  * Detects if the point goes outside the image domain
  *
**/
double
bicubic_interpolation(
  double *input, //image to be interpolated
  double uu,     //x component of the vector field
  double vv,     //y component of the vector field
  int nx,        //width of the image
  int ny,        //height of the image
  int nz,        //number of channels of the image
  int k          //actual channel
)
{
    int sx = (uu < 0) ? -1 : 1;
    int sy = (vv < 0) ? -1 : 1;

    int x, y, mx, my, dx, dy, ddx, ddy;

    x = neumann_bc ((int) uu, nx);
    y = neumann_bc ((int) vv, ny);
    mx = neumann_bc ((int) uu - sx, nx);
    my = neumann_bc ((int) vv - sx, ny);
    dx = neumann_bc ((int) uu + sx, nx);
    dy = neumann_bc ((int) vv + sy, ny);
    ddx = neumann_bc ((int) uu + 2 * sx, nx);
    ddy = neumann_bc ((int) vv + 2 * sy, ny);

    //obtain the interpolation points of the image
    double p11 = input[(mx  + nx * my) * nz + k];
    double p12 = input[(x   + nx * my) * nz + k];
    double p13 = input[(dx  + nx * my) * nz + k];
    double p14 = input[(ddx + nx * my) * nz + k];

    double p21 = input[(mx  + nx * y) * nz + k];
    double p22 = input[(x   + nx * y) * nz + k];
    double p23 = input[(dx  + nx * y) * nz + k];
    double p24 = input[(ddx + nx * y) * nz + k];

    double p31 = input[(mx  + nx * dy) * nz + k];
    double p32 = input[(x   + nx * dy) * nz + k];
    double p33 = input[(dx  + nx * dy) * nz + k];
    double p34 = input[(ddx + nx * dy) * nz + k];

    double p41 = input[(mx  + nx * ddy) * nz + k];
    double p42 = input[(x   + nx * ddy) * nz + k];
    double p43 = input[(dx  + nx * ddy) * nz + k];
    double p44 = input[(ddx + nx * ddy) * nz + k];

    //create array
    double pol[4][4] = {
    {p11, p21, p31, p41}, {p12, p22, p32, p42},
    {p13, p23, p33, p43}, {p14, p24, p34, p44}
    };

    //return interpolation
    return bicubic_interpolation (pol, (double) uu - x, (double) vv - y);
}

/**
  *
  * Compute the bicubic interpolation of an image from a parametric transform
  *
**/
void bicubic_interpolation(
  double *input,   //image to be warped
  double *output,  //warped output image with bicubic interpolation
  double *params,  //parameters of the transformation
  int nparams,     //number of parameters of the transform
  int nx,          //width of the image
  int ny,          //height of the image
  int nz,          //number of channels of the image
  int delta,       //distance to the boundary
  int nanifoutside //parameter for discarding boudary pixels
)
{
  double out_value = ( nanifoutside ) ? NAN : 0;

  for (int i=0; i<ny; i++)
    for (int j=0; j<nx; j++) {
      int p=i*nx+j;
      double x, y;

      //transform coordinates using the parametric model
      project(j, i, params, x, y, nparams);

      bool out = false;
      if ( x < delta || x > nx-1-delta
         || y < delta || y > ny-1-delta )
          out = true;

      if ( out ) {
          for(int k=0; k<nz; k++)
              output[p*nz+k]=out_value;
      }
      else {
          //obtain the bicubic interpolation at position (uu, vv)
          for(int k=0; k<nz; k++)
              output[p*nz+k]=bicubic_interpolation(
                  input, x, y, nx, ny, nz, k);
      }
    }
}
