// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2018, Thibaud Briand <thibaud.briand@enpc.fr>
// Copyright (C) 2015, Javier Sánchez Pérez <jsanchez@ulpgc.es>
// All rights reserved.

#ifndef FILE_H
#define FILE_H

/**
 *
 *  Functions to read images using the iio library
 *  It allocates memory for the image and returns true if it
 *  correctly reads the image
 *
 */
bool read_image
(
  char *fname,  //file name
  double **f,   //output image
  int &nx,      //number of columns of the image
  int &ny,      //number of rows of the image
  int &nz       //number of channels of the image
);

bool read_image
(
  char *fname,  //file name
  float **f,    //output image
  int &nx,      //number of columns of the image
  int &ny,      //number of rows of the image
  int &nz       //number of channels of the image
);

/**
 *
 *  Functions to save images using the iio library
 *
 */
void save_image
(
  char *fname,  //file name
  double *f,    //output image
  int nx,       //number of columns of the image
  int ny,       //number of rows of the image
  int nz        //number of channels of the image
);


void save_normalize_image
(
  char *fname,  //file name
  double *f,    //output image
  int nx,       //number of columns of the image
  int ny,       //number of rows of the image
  int nz        //number of channels of the image
);


void save_image
(
  char *fname,  //file name
  float *f,     //output image
  int nx,       //number of columns of the image
  int ny,       //number of rows of the image
  int nz        //number of channels of the image
);

/**
 *
 *  Function to save flow fields using the iio library
 *
 */
void save_flow
(
  char *file, //file name
  double *u,  //x component of the optical flow
  double *v,  //y component of the optical flow
  int nx,     //number of columns
  int ny      //number of rows
);

/**
 *
 *  Function to read the parameters in ascii format
 *  It reads a header with: nparams nx ny
 *  Then it reads the parameters for each pixel
 *
 */
void read
(
  char *file,   //input file name
  double **p,   //parameters to be read
  int &nparams, //number of parameters
  int &nx,      //number of columns
  int &ny       //number of rows
);

/**
 *
 *  Function to save the parameters in ascii format
 *  It creates a header with: nparams nx ny
 *  Then it stores the parameters for each pixel
 *
 */
void save
(
  char *file,  //output file name
  double *p,   //parameters to be saved
  int nparams, //number of parameters
  int nx,      //number of columns
  int ny       //number of rows
);

/**
 *
 *  Function to read the parameters in ascii format
 *  It reads a header with: nparams nx ny
 *  Then it reads the parameters
 *
 */
void read
(
  const char *file, //input file name
  double **p,       //parameters to be read
  int &nparams      //number of parameters
);

/**
 *
 *  Function to save the parameters in ascii format
 *  It creates a header with: nparams nx ny
 *  Then it stores the parameters
 *
 */
void save
(
  const char *file, //output file name
  double *p,        //parameters to be saved
  int nparams       //number of parameters
);

#endif
