// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2018, Thibaud Briand <thibaud.briand@enpc.fr>
// Copyright (C) 2015, Javier Sánchez Pérez <jsanchez@dis.ulpgc.es>
// Copyright (C) 2014, Nelson Monzón López <nmonzon@ctim.es>
// All rights reserved.

#ifndef ZOOM_H
#define ZOOM_H

/**
  *
  * Compute the size of a zoomed image from the zoom factor
  *
**/
void zoom_size 
(
  int nx,   //width of the orignal image
  int ny,   //height of the orignal image          
  int &nxx, //width of the zoomed image
  int &nyy, //height of the zoomed image
  double factor = 0.5 //zoom factor between 0 and 1
);

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
  double factor = 0.5 //zoom factor between 0 and 1
);

/**
  *
  * Function to upsample the parameters of the transformation
  *
**/
void zoom_in_parameters 
(
  double *p,    //input image
  double *pout, //output image   
  int nparams,  //number of parameters
  int nx,       //width of the original image
  int ny,       //height of the original image
  int nxx,      //width of the zoomed image
  int nyy       //height of the zoomed image
);

#endif
