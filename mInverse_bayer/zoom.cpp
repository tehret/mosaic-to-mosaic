// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2018, Thibaud Briand <thibaud.briand@enpc.fr>
// Copyright (C) 2015, Javier Sánchez Pérez <jsanchez@dis.ulpgc.es>
// Copyright (C) 2014, Nelson Monzón López <nmonzon@ctim.es>
// All rights reserved.

#include <math.h>
#include <stdio.h>

#include "zoom.h"
#include "mask.h"
#include "bicubic_interpolation.h"
#include "transformation.h"

#define ZOOM_SIGMA_ZERO 0.6

/**
  *
  * Compute the size of a zoomed image from the zoom factor
  *
**/
void zoom_size
(
  int nx,       //width of the orignal image
  int ny,       //height of the orignal image
  int &nxx,     //width of the zoomed image
  int &nyy,     //height of the zoomed image
  double factor //zoom factor between 0 and 1
)
{
  nxx = (int) ((double) nx * factor + 0.5);
  nyy = (int) ((double) ny * factor + 0.5);
}

/**
  *
  * Function to downsample the image
  *
**/
void zoom_out
(
  double *I,    //input image
  double *Iout, //output image
  int nx,       //image width
  int ny,       //image height
  int nz,       // number of color channels in image
  double factor //zoom factor between 0 and 1
)
{
  int nxx, nyy, original_size =nx*ny*nz;
  double ifactor = 1.0/factor;
  double *Is=new double[original_size];

  for (int i=0; i<original_size; i++)
    Is[i]=I[i];

  //calculate the size of the zoomed image
  zoom_size(nx, ny, nxx, nyy, factor);

  //compute the Gaussian sigma for smoothing
  double sigma=ZOOM_SIGMA_ZERO*sqrt(ifactor*ifactor-1.0);

  //pre-smooth the image
  gaussian(Is, nx, ny, nz, sigma);

  if ( (int) ifactor == ifactor) {
      for (int i1=0; i1<nyy; i1++)
        for (int j1=0; j1<nxx; j1++)
        {
            int i2= i1*ifactor;
            int j2= j1*ifactor;
            for(int index_color=0; index_color<nz; index_color++)
            Iout[(i1*nxx+j1)*nz+index_color]=Is[(i2*nx+j2)*nz+index_color];
        }
    }
    else {
    // re-sample the image using bicubic interpolation
        for (int i1=0; i1<nyy; i1++)
        for (int j1=0; j1<nxx; j1++)
        {
            double i2=(double)i1*ifactor;
            double j2=(double)j1*ifactor;
            for(int index_color=0; index_color<nz; index_color++)
            Iout[(i1*nxx+j1)*nz+index_color]=
            bicubic_interpolation(Is, j2, i2, nx, ny, nz, index_color);
        }
    }

  delete []Is;
}

/**
  *
  * Function to upsample the parameters of the transformation
  *
**/
void zoom_in_parameters
(
  double *p,    //input parameters
  double *pout, //output parameters
  int nparams,  //number of parameters
  int nx,       //width of the original image
  int ny,       //height of the original image
  int nxx,      //width of the zoomed image
  int nyy       //height of the zoomed image
)
{
  //compute the zoom factor
  double factorx=((double)nxx/nx);
  double factory=((double)nyy/ny);
  double nu=(factorx>factory)?factorx:factory;

  switch(nparams) {
    default: case TRANSLATION_TRANSFORM: //p=(tx, ty)
      pout[0]=p[0]*nu;
      pout[1]=p[1]*nu;
      break;
    case EUCLIDEAN_TRANSFORM: //p=(tx, ty, tita)
      pout[0]=p[0]*nu;
      pout[1]=p[1]*nu;
      pout[2]=p[2];
      break;
    case SIMILARITY_TRANSFORM: //p=(tx, ty, a, b)
      pout[0]=p[0]*nu;
      pout[1]=p[1]*nu;
      pout[2]=p[2];
      pout[3]=p[3];
      break;
    case AFFINITY_TRANSFORM: //p=(tx, ty, a00, a01, a10, a11)
      pout[0]=p[0]*nu;
      pout[1]=p[1]*nu;
      pout[2]=p[2];
      pout[3]=p[3];
      pout[4]=p[4];
      pout[5]=p[5];
      break;
    case HOMOGRAPHY_TRANSFORM: //p=(h00, h01,..., h21)
      pout[0]=p[0];
      pout[1]=p[1];
      pout[2]=p[2]*nu;
      pout[3]=p[3];
      pout[4]=p[4];
      pout[5]=p[5]*nu;
      pout[6]=p[6]/nu;
      pout[7]=p[7]/nu;
      break;
  }
}
