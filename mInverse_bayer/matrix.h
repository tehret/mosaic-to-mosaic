// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2018, Thibaud Briand <thibaud.briand@enpc.fr>
// Copyright (C) 2015, Javier Sánchez Pérez <jsanchez@dis.ulpgc.es>
// All rights reserved.

#ifndef MATRIX_H
#define MATRIX_H

//Multiplication of a square matrix and a vector
void Axb(double *A, double *b, double *p, int n);

//Multiplication of the transpose of a matrix and a vector
//p should be initialized to zero outside
void Atb(double *A, double *b, double *p, int n, int m);

//Multiplication of the transpose of a matrix, a vector and a scalar
//p should be initialized to zero outside
void sAtb(double s, double *A, double *b, double *p, int n, int m);

//Multiplication of the transpose of a matrix and itself
//B should be initialized to zero outside
void AtA(double *A, double *B, int n, int m);

//Multiplication of the transpose of a matrix and itself with a scalar
//B should be initialized to zero outside
void sAtA(double s, double *A, double *B, int n, int m);

//Multiplication with a scalar
//B should be initialized to zero outside
void sA(double s, double *A, double *B, int m);

//Function to compute the inverse of a matrix
//through Gaussian elimination
int inverse(double *A, double *A_1, int N = 3);

#endif
