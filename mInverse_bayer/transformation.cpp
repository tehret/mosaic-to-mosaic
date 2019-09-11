// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2018, Thibaud Briand <thibaud.briand@enpc.fr>
// Copyright (C) 2015, Javier Sánchez Pérez <jsanchez@dis.ulpgc.es>
// All rights reserved.

#include "transformation.h"
#include <math.h>

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
)
{
  switch(nparams)
  {
    default: case TRANSLATION_TRANSFORM:  //p=(tx, ty)
      for(int i=0; i<nx*ny; i++)
      {
        int c=2*i*nparams;
        J[c]  =1.0; J[c+1]=0.0;
        J[c+2]=0.0; J[c+3]=1.0;
      }
      break;
    case EUCLIDEAN_TRANSFORM:  //p=(tx, ty, theta)
      for(int i=0; i<ny; i++)
        for(int j=0; j<nx; j++)
        {
          int c=2*(i*nx+j)*nparams;
          J[c]  =1.0; J[c+1]=0.0; J[c+2]=-i;
          J[c+3]=0.0; J[c+4]=1.0; J[c+5]= j;
        }
      break;
    case SIMILARITY_TRANSFORM: //p=(tx, ty, a, b)
      for(int i=0; i<ny; i++)
        for(int j=0; j<nx; j++)
        {
          int c=2*(i*nx+j)*nparams;
          J[c]  =1.0; J[c+1]=0.0; J[c+2]=j; J[c+3]=-i;
          J[c+4]=0.0; J[c+5]=1.0; J[c+6]=i; J[c+7]= j;
        }
      break;
    case AFFINITY_TRANSFORM:  //p=(tx, ty, a00, a01, a10, a11)
      for(int i=0; i<ny; i++)
        for(int j=0; j<nx; j++)
        {
          int c=2*(i*nx+j)*nparams;
          J[c]  =1.0;J[c+1]=0.0;J[c+2]=  j;J[c+3]=  i;J[c+ 4]=0.0;J[c+ 5]=0.0;
          J[c+6]=0.0;J[c+7]=1.0;J[c+8]=0.0;J[c+9]=0.0;J[c+10]=  j;J[c+11]=  i;
        }
      break;
    case HOMOGRAPHY_TRANSFORM: //p=(h00, h01,..., h21)
      for(int i=0; i<ny; i++)
        for(int j=0; j<nx; j++)
        {
          int c=2*(i*nx+j)*nparams;
          J[c]  =  j; J[c+1]=  i; J[c+2]=1.0;  J[c+3]=0.0;
          J[c+4]=0.0; J[c+5]=0.0; J[c+6]=-j*j; J[c+7]=-j*i;

          J[c+8]=0.0; J[c+9] =0.0; J[c+10]= 0.0; J[c+11]=   j;
          J[c+12]= i; J[c+13]=1.0; J[c+14]=-j*i; J[c+15]=-i*i;
        }
      break;
  }
}

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
)
{
  switch(nparams)
  {
    default: case TRANSLATION_TRANSFORM: //p=(tx, ty)+++
      for(int i = 0; i < nparams; i++)
        p[i]-=dp[i];
      break;
    case EUCLIDEAN_TRANSFORM: //p=(tx, ty, tita)
    {
      double a=cos(dp[2]);
      double b=sin(dp[2]);
      double c=dp[0];
      double d=dp[1];
      double ap=cos(p[2]);
      double bp=sin(p[2]);
      double cp=p[0];
      double dp=p[1];
      double cost=a*ap+b*bp;
      double sint=a*bp-b*ap;
      p[0]=cp-bp*(b*c-a*d)-ap*(a*c+b*d);
      p[1]=dp-bp*(a*c+b*d)+ap*(b*c-a*d);
      p[2]=atan2(sint,cost);
    }
    break;
    case SIMILARITY_TRANSFORM: //p=(tx, ty, a, b)
    {
      double a=dp[2];
      double b=dp[3];
      double c=dp[0];
      double d=dp[1];
      double det=(2*a+a*a+b*b+1);
      if(det*det>1E-10)
      {
        double ap=p[2];
        double bp=p[3];
        double cp=p[0];
        double dp=p[1];

        p[0]=cp-bp*(-d-a*d+b*c)/det+(ap+1)*(-c-a*c-b*d)/det;
        p[1]=dp+bp*(-c-a*c-b*d)/det+(ap+1)*(-d-a*d+b*c)/det;
        p[2]=b*bp/det+(a+1)*(ap+1)/det-1;
        p[3]=-b*(ap+1)/det+bp*(a+1)/det ;
      }
    }
    break;
    case AFFINITY_TRANSFORM: //p=(tx, ty, a00, a01, a10, a11)
    {
      double a=dp[2];
      double b=dp[3];
      double c=dp[0];
      double d=dp[4];
      double e=dp[5];
      double f=dp[1];
      double det=(a-b*d+e+a*e+1);
      if(det*det>1E-10)
      {
        double ap=p[2];
        double bp=p[3];
        double cp=p[0];
        double dp=p[4];
        double ep=p[5];
        double fp=p[1];

        p[0]=cp+(-f*bp-a*f*bp+c*d*bp)/det+(ap+1)*(-c+b*f-c*e)/det;
        p[1]=fp+dp*(-c+b*f-c*e)/det+(-f+c*d-a*f-f*ep-a*f*ep+d*d*ep)/det;
        p[2]=((1+ap)*(1+e)-d*bp)/det-1;
        p[3]=(bp+a*bp-b-b*ap)/det;
        p[4]=(dp*(1+e)-d-d*ep)/det;
        p[5]=(a+ep+a*ep+1-b*dp)/det-1;
      }
    }
    break;
    case HOMOGRAPHY_TRANSFORM:   //p=(h00, h01,..., h21)
    {
      double a=dp[0];
      double b=dp[1];
      double c=dp[2];
      double d=dp[3];
      double e=dp[4];
      double f=dp[5];
      double g=dp[6];
      double h=dp[7];
      double ap=p[0];
      double bp=p[1];
      double cp=p[2];
      double dp=p[3];
      double ep=p[4];
      double fp=p[5];
      double gp=p[6];
      double hp=p[7];

      double det=f*hp+a*f*hp-c*d*hp+gp*(c-b*f+c*e)-a+b*d-e-a*e-1;
      if(det*det>1E-10)
      {
        p[0]=((d*bp-f*g*bp)+cp*(g-d*h+g*e)+(ap+1)*(f*h-e-1))/det-1;
        p[1]=(h*cp+a*h*cp-b*g*cp-bp-a*bp+c*g*bp+b-c*h+b*ap-c*h*ap)/det;
        p[2]=(f*bp+a*f*bp-c*d*bp+(ap+1)*(c-b*f+c*e)+cp*(-a+b*d-e-a*e-1))/det;
        p[3]=(fp*(g-d*h+g*e)+d-f*g+d*ep-f*g*ep+dp*(f*h-e-1))/det;
        p[4]=(b*dp-c*h*dp+h*fp+a*h*fp-b*g*fp-a+c*g-ep-a*ep+c*g*ep-1)/det-1;
        p[5]=(dp*(c-b*f+c*e)+f+a*f-c*d+f*ep+a*f*ep-c*d*ep+fp*(-a+b*d-e-a*e-1))/det;
        p[6]=(d*hp-f*g*hp+g-d*h+g*e+gp*(f*h-e-1))/det;
        p[7]=(h+a*h-b*g+b*gp-c*h*gp-hp-a*hp+c*g*hp)/det;
      }
    }
    break;
  }
}

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
)
{
  switch(nparams) {
    default: case TRANSLATION_TRANSFORM: //p=(tx, ty)
      xp=x+p[0];
      yp=y+p[1];
      break;
    case EUCLIDEAN_TRANSFORM:   //p=(tx, ty, tita)
      xp=cos(p[2])*x-sin(p[2])*y+p[0];
      yp=sin(p[2])*x+cos(p[2])*y+p[1];
      break;
    case SIMILARITY_TRANSFORM:  //p=(tx, ty, a, b)
      xp=(1+p[2])*x-p[3]*y+p[0];
      yp=p[3]*x+(1+p[2])*y+p[1];
      break;
    case AFFINITY_TRANSFORM:    //p=(tx, ty, a00, a01, a10, a11)
      xp=(1+p[2])*x+p[3]*y+p[0];
      yp=p[4]*x+(1+p[5])*y+p[1];
      break;
    case HOMOGRAPHY_TRANSFORM:  //p=(h00, h01,..., h21)
      double d=p[6]*x+p[7]*y+1;
      xp=((1+p[0])*x+p[1]*y+p[2])/d;
      yp=(p[3]*x+(1+p[4])*y+p[5])/d;
      break;
  }
}

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
)
{
  matrix[0]=matrix[4]=matrix[8]=1;
  matrix[1]=matrix[2]=matrix[3]=0;
  matrix[5]=matrix[6]=matrix[7]=0;
  switch(nparams) {
    default: case TRANSLATION_TRANSFORM: //p=(tx, ty)
      matrix[2]=p[0];
      matrix[5]=p[1];
      break;
    case EUCLIDEAN_TRANSFORM:   //p=(tx, ty, tita)
      matrix[0]=cos(p[2]);
      matrix[1]=-sin(p[2]);
      matrix[2]=p[0];
      matrix[3]=sin(p[2]);
      matrix[4]=cos(p[2]);
      matrix[5]=p[1];
     break;
    case SIMILARITY_TRANSFORM:  //p=(tx, ty, a, b)
      matrix[0]=1+p[2];
      matrix[1]=-p[3];
      matrix[2]=p[0];
      matrix[3]=-matrix[1];
      matrix[4]=matrix[0];
      matrix[5]=p[1];
      break;
    case AFFINITY_TRANSFORM:    //p=(tx, ty, a00, a01, a10, a11)
      matrix[0]=1+p[2];
      matrix[1]=p[3];
      matrix[2]=p[0];
      matrix[3]=p[4];
      matrix[4]=1+p[5];
      matrix[5]=p[1];
      break;
    case HOMOGRAPHY_TRANSFORM:  //p=(h00, h01,..., h21)
      matrix[0]=1+p[0];
      matrix[1]=p[1];
      matrix[2]=p[2];
      matrix[3]=p[3];
      matrix[4]=1+p[4];
      matrix[5]=p[5];
      matrix[6]=p[6];
      matrix[7]=p[7];
      break;
  }
}
