// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2018, Thibaud Briand <thibaud.briand@enpc.fr>
// Copyright (C) 2015, Javier Sánchez Pérez <jsanchez@dis.ulpgc.es>
// All rights reserved.

#include <cmath>
#include <stdio.h>
#include <stdlib.h>

#include "bicubic_interpolation.h"
#include "file.h"
#include "inverse_compositional_algorithm.h"
#include "transformation.h"

/*********************************************************************
 * NOTE:                                                             *
 * This file generates some information for the online demo in IPOL  *
 * It can be removed for using the program outside the demo system   *
 *                                                                   *
 *********************************************************************/

/**
 *
 *  Function to project a point with two transformations and
 *  compute the distance between them
 *
 */
double distance(
  double *m1, //first transformation
  double *m2, //second transformation
  double x,   //x point coordinate
  double y    //y point coordinate
)
{
  double x1=99999,y1=99999,z1=m1[6]*x+m1[7]*y+1;
  double x2=99999,y2=99999,z2=m2[6]*x+m2[7]*y+1;
  if(z1*z1>1E-10 && z2*z2>1E-10)
  {
    x1=(m1[0]*x+m1[1]*y+m1[2])/z1;
    y1=(m1[3]*x+m1[4]*y+m1[5])/z1;
    x2=(m2[0]*x+m2[1]*y+m2[2])/z2;
    y2=(m2[3]*x+m2[4]*y+m2[5])/z2;
  }
  double d=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2);
  return sqrt(d);
}

/**
 *
 *  Function to write some output information for the demo
 *
 */
void print_output(
  double *I1,  //first image
  double *I2,  //second image
  double *p,   //parametric model
  double *p2,  //parametric model
  int nparams, //number of parameters
  int nparams2,//number of parameters
  int nx,      //number of columns
  int ny,      //number of rows
  int nz       //number of channels
)
{
  double *Iw=new double[nx*ny*nz];
  double *DI=new double[nx*ny*nz];
  double *DIrmse= new double[nx*ny];
  double *EPE=new double[nx*ny];

  //compute the interpolated image I2(x')
  bicubic_interpolation(I2, Iw, p, nparams, nx, ny, nz, 0, 1);
  char outfile[50]="output_estimated.tiff";
  save_image(outfile, Iw, nx, ny, nz);
  char outfile2[50]="output_estimated.png";
  save_image(outfile2, Iw, nx, ny, nz);

  //compute the difference image I2(x') - I1(x) and the RMSE
  double sum=0.0, rmse=9999;
  int size = 0;
  for (int i=0; i<nx*ny*nz; i++) {
      DI[i] = Iw[i] - I1[i];
      if ( std::isfinite(DI[i]) ) {
          sum += DI[i]*DI[i];
          size++;
      }
  }
  if (size > 0)
      rmse = sqrt(sum/size);
  printf("RMSE(I1(x),I2(x'))=%lf\n",rmse);
  char diff_image[50]="diff_image.tiff";
  save_image(diff_image, DI, nx, ny, nz);
  char diff_image2[50]="diff_image.png";
  save_normalize_image(diff_image2, DI, nx, ny, nz);

  //compute the pointwise rmse of the difference image
  for(int p=0; p<nx*ny; p++) {
    double norm = 0.0;
    for(int c=0; c<nz; c++) norm += DI[p*nz+c]*DI[p*nz+c];
    DIrmse[p] = sqrt(norm/nz);
  }
  char diff_image_rmse[50]="diff_image_rmse.tiff";
  save_image(diff_image_rmse, DIrmse, nx, ny, 1);
  char diff_image_rmse2[50]="diff_image_rmse.png";
  save_normalize_image(diff_image_rmse2, DIrmse, nx, ny, 1);

  //computing the EPE field
  double m1[9], m2[9], mean = 0.0;
  params2matrix(p, m1, nparams);
  if( p2!=NULL ) {
    params2matrix(p2, m2, nparams2);
    for (int i=0; i<ny; i++) {
        for (int j=0; j<nx; j++) {
            double tmp = distance(m1, m2, j, i);
            EPE[i * nx + j] = tmp;
            mean += tmp;
        }
    }
    mean /= (nx*ny);
    printf("EPE=%lf\n", mean);
    char epefile[50]="epe.tiff";
    save_image(epefile, EPE, nx, ny, 1);
    char epefile2[50]="epe.png";
    save_normalize_image(epefile2, EPE, nx, ny, 1);
  }
  else
    printf("EPE=-N/A-\n");

  //writing matrices
  printf("Computed transform=%0.14lg %0.14lg %0.14lg %0.14lg %0.14lg %0.14lg "
         "%0.14lg %0.14lg %0.14lg\n",
         m1[0],m1[1],m1[2],m1[3],m1[4],m1[5],m1[6],m1[7],m1[8]);
  if(p2!=NULL)
    printf("Original transform=%0.14lg %0.14lg %0.14lg %0.14lg %0.14lg %0.14lg "
           "%0.14lg %0.14lg %0.14lg\n",
           m2[0],m2[1],m2[2],m2[3],m2[4],m2[5],m2[6],m2[7],m2[8]);
  else
    printf("Original transform=- - - - - - - - -\n");

  delete []Iw;
  delete []DI;
  delete []EPE;
  delete []DIrmse;
}

int main(int c, char *v[])
{
  if(c < 4 || c > 5) {
      printf("<Usage>: %s image1 image2 transform1 [transform2]\n", v[0]);
      return(EXIT_FAILURE);
  }

  int nx, ny, nz, nx1, ny1, nz1;

  char  *image1= v[1];
  char  *image2= v[2];
  const char  *transform1= v[3];
  const char  *transform2= c > 4 ? v[4] : "-";

  //read the input images
  double *I1, *I2;
  bool correct1=read_image(image1, &I1, nx, ny, nz);
  bool correct2=read_image(image2, &I2, nx1, ny1, nz1);

  if ( ! (correct1 && correct2 && nx == nx1 && ny == ny1 && nz == nz1) )
  {
      printf("Images should have the same size\n");
      return(EXIT_FAILURE);
  }

  int n1 = 0, n2 = 0;
  double *p1=NULL, *p2=NULL;
  read(transform1, &p1, n1);
  read(transform2, &p2, n2);

  print_output(I1, I2, p1, p2, n1, n2, nx, ny, nz);

  //free memory
  free (I1);
  free (I2);
  delete []p1;
  delete []p2;

  return(EXIT_SUCCESS);
}
