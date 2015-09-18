/* Copyright (c) 2015 The University of Edinburgh. */

/* 
* This software was developed as part of the                       
* EC FP7 funded project Adept (Project ID: 610490)                 
* www.adept-project.eu                                            
*/

/* Licensed under the Apache License, Version 2.0 (the "License"); */
/* you may not use this file except in compliance with the License. */
/* You may obtain a copy of the License at */

/*     http://www.apache.org/licenses/LICENSE-2.0 */

/* Unless required by applicable law or agreed to in writing, software */
/* distributed under the License is distributed on an "AS IS" BASIS, */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. */
/* See the License for the specific language governing permissions and */
/* limitations under the License. */

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include "level1.h"
#include "utils.h"
#include "matrix_utils.h"

#include <omp.h>

/*
 * Vector dot product, integers
 *
 * result = result + v1_i * v2_i
 *
 * Input: size of the vectors (in number of elements)
 * Output: dot product
 *
 */
int int_dot_product(unsigned int size){

  int i;

  /* create two vectors */
  int *v1 = (int *)malloc(size * sizeof(int));
  int *v2 = (int *)malloc(size * sizeof(int));

  /* result variable */
  unsigned int result = 0;

  if(v1 == NULL || v2 == NULL){
    printf("Out Of Memory: could not allocate space for the two arrays.\n");
    return 0;
  }

  struct timespec start, end;

  /* fill vectors with random integer values */
#pragma omp parallel for schedule(static)
  for(i=0; i<size; i++){
    v1[i] = KISS;
    v2[i] = KISS;
  }

  clock_gettime(CLOCK, &start);

  /* perform dot product */

#pragma omp parallel for schedule(static) default(none) shared(v1, v2, size) private(i) reduction(+:result)
  for(i=0; i<size; i++){
    result = result + v1[i] * v2[i];
  }

  clock_gettime(CLOCK, &end);

  /* print result so compiler does not throw it away */
  printf("Dot product result: %d\n", result);

  elapsed_time_hr(start, end, "Integer dot product.");

  free(v1);
  free(v2);

  return 0;

}

/*
 * Vector dot product, floats
 *
 * result = result + v1_i * v2_i
 *
 * Input: size of the vectors (in number of elements)
 * Output: dot product
 *
 */
int float_dot_product(unsigned int size){

  int i;

  /* create three vectors */
  float *v1 = (float *)malloc(size * sizeof(float));
  float *v2 = (float *)malloc(size * sizeof(float));
  float result = 0.0;

  if(v1 == NULL || v2 == NULL){
    printf("Out Of Memory: could not allocate space for the two arrays.\n");
    return 0;
  }

  struct timespec start, end;

  /* fill vectors with random floats */
#pragma omp parallel for schedule(static)
  for(i=0; i<size; i++){
    v1[i] = UNI;
    v2[i] = UNI;
  }

  clock_gettime(CLOCK, &start);

  /* perform dot product */
#pragma omp parallel for schedule(static) default(none) shared(v1, v2, size) private(i) reduction(+:result)
  for(i=0; i<size; i++){
    result = result + v1[i] * v2[i];
  }

  clock_gettime(CLOCK, &end);

  /* print result so compiler does not throw it away */
  printf("Dot product result: %f\n", result);

  elapsed_time_hr(start, end, "Float dot product.");

  free(v1);
  free(v2);

  return 0;

}


/*
 * Vector dot product, doubles
 *
 * result = result + v1_i * v2_i
 *
 * Input: size of the vectors (in number of elements)
 * Output: dot product
 *
 */
int double_dot_product(unsigned int size){

  int i;

  /* create three vectors */
  double *v1 = (double *)malloc(size * sizeof(double));
  double *v2 = (double *)malloc(size * sizeof(double));
  double result = 0.0;

  if(v1 == NULL || v2 == NULL){
    printf("Out Of Memory: could not allocate space for the two arrays.\n");
    return 0;
  }



  struct timespec start, end;

  /* fill vectors with random floats */
#pragma omp parallel for schedule(static)
  for(i=0; i<size; i++){
    v1[i] = VNI;
    v2[i] = VNI;
  }

  clock_gettime(CLOCK, &start);

  /* perform dot product */
#pragma omp parallel for schedule(static) default(none) shared(v1, v2, size) private(i) reduction(+:result)
  for(i=0; i<size; i++){
    result = result + v1[i] * v2[i];
  }
  clock_gettime(CLOCK, &end);

  /* print result so compiler does not throw it away */
  printf("Dot product result: %f\n", result);

  elapsed_time_hr(start, end, "Double dot product.");

  free(v1);
  free(v2);

  return 0;

}


/* Vector scalar multiplication, integers    */
/* v_i = a * v1_i                     */
int int_scalar_mult(unsigned int size){

  int i;

  /* create vector and scalar */
  int *v = (int *)malloc(size * sizeof(int));
  unsigned int a = 0;

  if(v == NULL){
    printf("Out Of Memory: could not allocate space for the array.\n");
    return 0;
  }

  struct timespec start, end;

  /* fill vector with random ints */
#pragma omp parallel for schedule(static)
  for(i=0; i<size; i++){
    v[i] = KISS;
  }

  /* assign random int value */
  a = KISS;

  clock_gettime(CLOCK, &start);

  /* perform scalar product */
#pragma omp parallel for schedule(static) default(none) shared(a,v,size) private(i)
  for(i=0; i<size; i++){
    v[i] = a * v[i];
  }
  clock_gettime(CLOCK, &end);

  /* print result so compiler does not throw it away */
  printf("Scalar product result: %d\n", v[0]);

  elapsed_time_hr(start, end, "Int scalar multiplication.");

  free(v);

  return 0;

}

/* Vector scalar product, floats    */
/* v_i = a * v1_i                     */
int float_scalar_mult(unsigned int size){

  int i;

  /* create vector and scalar */
  float *v = (float *)malloc(size * sizeof(float));
  float a = 0.0;

  if(v == NULL){
    printf("Out Of Memory: could not allocate space for the array.\n");
    return 0;
  }

  struct timespec start, end;

  /* fill vector with random floats */
#pragma omp parallel for schedule(static)
  for(i=0; i<size; i++){
    v[i] = UNI;
  }

  /* assign random float value */
  a = UNI;

  clock_gettime(CLOCK, &start);

  /* perfom scalar product */
#pragma omp parallel for schedule(static) default(none) shared(a,v,size) private(i)
  for(i=0; i<size; i++){
    v[i] = a * v[i];
  }

  clock_gettime(CLOCK, &end);

  /* print result so compiler does not throw it away */
  printf("Scalar product result: %f\n", v[0]);

  elapsed_time_hr(start, end, "Float scalar multiplication.");

  free(v);

  return 0;

}

/* Vector scalar product, doubles    */
/* v_i = a * v1_i                     */
int double_scalar_mult(unsigned int size){

  int i;

  /* create vector and scalar */
  double *v = (double *)malloc(size * sizeof(double));
  double a = 0.0;

  if(v == NULL){
    printf("Out Of Memory: could not allocate space for the array.\n");
    return 0;
  }

  struct timespec start, end;

  /* fill vector with random doubles */
#pragma omp parallel for schedule(static)
  for(i=0; i<size; i++){
    v[i] = VNI;
  }

  /* assign random double value */
  a = VNI;

  clock_gettime(CLOCK, &start);

  /* perfom scalar product */
#pragma omp parallel for schedule(static) default(none) shared(a,v,size) private(i)
  for(i=0; i<size; i++){
    v[i] = a * v[i];
  }

  clock_gettime(CLOCK, &end);

  /* print result so compiler does not throw it away */
  printf("Scalar product result: %f\n", v[0]);

  elapsed_time_hr(start, end, "Double scalar multiplication.");

  free(v);

  return 0;

}

/* compute the Euclidean norm of an int vector      */
/* !!!! naive implementation -- find algorithm that  */
/* !!!! will avoid over/underflow for large vectors  */
int int_norm(unsigned int size){

  int i;

  unsigned int *v = (unsigned int *)malloc(size * sizeof(unsigned int));
  unsigned int sum = 0;
  float norm = 0.0; /* Result is a float */

  if(v == NULL){
    printf("Out Of Memory: could not allocate space for the array.\n");
    return 0;
  }

  struct timespec start, end;

  /* fill vector with random ints */
#pragma omp parallel for schedule(static)
  for(i=0; i<size; i++){
    v[i] = 1+(int)UNI;
  }

  clock_gettime(CLOCK, &start);

#pragma omp parallel for schedule(static) default(none) shared(size, v) private(i) reduction(+:sum)
  for (i=0; i<size; i++){
    sum = sum + (v[i]*v[i]);
  }

  norm = sqrt(sum);

  clock_gettime(CLOCK, &end);

  elapsed_time_hr(start, end, "Int vector norm.");

  /* print result so compiler does not throw it away */
  printf("Norm = %f\n", norm);

  free(v);

  return 0;
}

/* compute the Euclidean norm of a float vector      */
/* !!!! naive implementation -- find algorithm that  */
/* !!!! will avoid over/underflow for large vectors  */
int float_norm(unsigned int size){

  int i;

  float *v = (float *)malloc(size * sizeof(float));
  float sum = 0.0, norm = 0.0;

  if(v == NULL){
    printf("Out Of Memory: could not allocate space for the array.\n");
    return 0;
  }

  struct timespec start, end;

  /* fill vector with random floats */
#pragma omp parallel for schedule(static)
  for(i=0; i<size; i++){
    v[i] = UNI;
  }

  clock_gettime(CLOCK, &start);

#pragma omp parallel for schedule(static) default(none) shared(size, v) private(i) reduction(+:sum)
  for (i=0; i<size; i++){
    sum = sum + (v[i]*v[i]);
  }

  norm = sqrt(sum);

  clock_gettime(CLOCK, &end);

  elapsed_time_hr(start, end, "Float vector norm.");

  /* print result so compiler does not throw it away */
  printf("Norm = %f\n", norm);

  free(v);

  return 0;
}

/* compute the Euclidean norm of a double vector      */
/* !!!! naive implementation -- find algorithm that  */
/* !!!! will avoid over/underflow for large vectors  */
int double_norm(unsigned int size){

  int i;

  double *v = (double *)malloc(size * sizeof(double));
  double sum = 0.0, norm = 0.0;

  if(v == NULL){
    printf("Out Of Memory: could not allocate space for the array.\n");
    return 0;
  }

  struct timespec start, end;

  /* fill vector with random doubles */
#pragma omp parallel for schedule(static)
  for(i=0; i<size; i++){
    v[i] = UNI;
  }

  clock_gettime(CLOCK, &start);

#pragma omp parallel for schedule(static) default(none) shared(size, v) private(i) reduction(+:sum)
  for (i=0; i<size; i++){
    sum = sum + (v[i]*v[i]);
  }

  norm = sqrt(sum);

  clock_gettime(CLOCK, &end);

  elapsed_time_hr(start, end, "Double vector norm.");

  /* print result so compiler does not throw it away */
  printf("Norm = %f\n", norm);

  free(v);

  return 0;
}



/*
 *
 * Compute vector-scalar product
 * AXPY, integers
 *
 * y = a * x + y
 *
 * Naive implementation
 *
 */
int int_axpy(unsigned int size){

  int i, a;
  int *x = (int *)malloc(size * sizeof(int));
  int *y = (int *)malloc(size * sizeof(int));

  if(x == NULL || y == NULL){
    printf("Out Of Memory: could not allocate space for the two arrays.\n");
    return 0;
  }


  a = KISS;

  /* fill x and y vectors with random ints */
#pragma omp parallel for schedule(static)
  for(i=0; i<size; i++){
    x[i] = KISS;
    y[i] = KISS;
  }

  struct timespec start, end;

  clock_gettime(CLOCK, &start);

#pragma omp parallel for schedule(static) default(none) shared(size, a, x, y) private(i)
  for(i=0; i<size; i++){
    y[i] = a * x[i] + y[i];
  }

  clock_gettime(CLOCK, &end);

  elapsed_time_hr(start, end, "Int AXPY.");

  /* print some of the result so compiler does not throw it away */
  printf("APXY result = %d\n", y[0]);

  free(x);
  free(y);

  return 0;
}

/*
 *
 * Compute vector-scalar product
 * AXPY, floats
 *
 * y = a * x + y
 *
 * Naive implementation
 *
 */
int float_axpy(unsigned int size){

  int i;
  float a;
  float *x = (float *)malloc(size * sizeof(float));
  float *y = (float *)malloc(size * sizeof(float));

  if(x == NULL || y == NULL){
    printf("Out Of Memory: could not allocate space for the two arrays.\n");
    return 0;
  }


  a = UNI;

  /* fill x and y vectors with random ints */
#pragma omp parallel for schedule(static)
  for(i=0; i<size; i++){
    x[i] = UNI;
    y[i] = UNI;
  }

  struct timespec start, end;

  clock_gettime(CLOCK, &start);

#pragma omp parallel for schedule(static) default(none) shared(size, a, x, y) private(i)
  for(i=0; i<size; i++){
    y[i] = a * x[i] + y[i];
  }

  clock_gettime(CLOCK, &end);

  elapsed_time_hr(start, end, "Float AXPY.");

  /* print some of the result so compiler does not throw it away */
  printf("APXY result = %f\n", y[0]);

  free(x);
  free(y);

  return 0;
}

/*
 *
 * Compute vector-scalar product
 * AXPY, doubles
 *
 * y = a * x + y
 *
 * Naive implementation
 *
 */
int double_axpy(unsigned int size){

  int i;
  double a;
  double *x = (double *)malloc(size * sizeof(double));
  double *y = (double *)malloc(size * sizeof(double));

  if(x == NULL || y == NULL){
    printf("Out Of Memory: could not allocate space for the two arrays.\n");
    return 0;
  }


  a = VNI;

  /* fill x and y vectors with random ints */
#pragma omp parallel for schedule(static)
  for(i=0; i<size; i++){
    x[i] = VNI;
    y[i] = VNI;
  }

  struct timespec start, end;

  clock_gettime(CLOCK, &start);

#pragma omp parallel for schedule(static) default(none) shared(size, a, x, y) private(i)
  for(i=0; i<size; i++){
    y[i] = a * x[i] + y[i];
  }

  clock_gettime(CLOCK, &end);

  elapsed_time_hr(start, end, "Double AXPY.");

  /* print some of the result so compiler does not throw it away */
  printf("APXY result = %f\n", y[0]);

  free(x);
  free(y);

  return 0;
}

/*
 * Dense Matrix-Vector product, integers
 *
 * y = A * x
 * where A is a square matrix
 *
 * Input:  number of elements in vectors and of rows/cols
 *         in matrix specified as number of ints
 *
 */
int int_dmatvec_product(unsigned int size){

  int i,j;
  int r1,r2;

  /* create two vectors */
  int *x = (int *)malloc(size * sizeof(int));
  int *y = (int *)calloc(size, sizeof(int));

  /* create matrix */
  int **A;
  A = (int **) malloc(size * sizeof(int *));
  for(i=0; i<size; i++){
    A[i] = (int *)malloc(size * sizeof(int));
  }

  if(x == NULL || y == NULL || A == NULL){
    printf("Out Of Memory: could not allocate space for the vectors and matrix.\n");
    return 0;
  }

  struct timespec start, end;

  r1 = KISS;
  r2 = KISS;

  /* fill vector x and matrix A with random integer values */
  for(i=0; i<size; i++){
    x[i] = r1;
    for(j=0; j<size; j++){
      A[i][j] = r2;
    }
  }

  clock_gettime(CLOCK, &start);

  /* perform matrix-vector product */
#pragma omp parallel for schedule(static) default(none) shared(size, A, x, y) private(i, j)
  for(i=0; i<size; i++){
    for(j=0; j<size; j++){
      y[i] = y[i] + A[i][j] * x[j];
    }
  }

  clock_gettime(CLOCK, &end);

  elapsed_time_hr(start, end, "Dense Matrix-Vector product.");

  /* print result so compiler does not throw it away */
  printf("Result vector y[0] = %d\n", y[0]);


  free(x);
  free(y);
  for(i =0; i<size; i++) free(A[i]);
  free(A);

  return 0;

}

/*
 * Dense Matrix-Vector product, floats
 *
 * y = A * x
 * where A is a square matrix
 *
 * Input:  number of elements in vectors and of rows/cols
 *         in matrix specified as number of floats
 *
 */
int float_dmatvec_product(unsigned int size){

  int i,j;
  float r1,r2;

  /* create two vectors */
  float *x = (float *)malloc(size * sizeof(float));
  float *y = (float *)calloc(size, sizeof(float));

  /* create matrix */
  float **A;
  A = (float **) malloc(size * sizeof(float *));
  for(i=0; i<size; i++){
    A[i] = (float *)malloc(size * sizeof(float));
  }

  if(x == NULL || y == NULL || A == NULL){
    printf("Out Of Memory: could not allocate space for the vectors and matrix.\n");
    return 0;
  }

  struct timespec start, end;

  r1 = UNI;
  r2 = UNI;

  /* fill vector x and matrix A with random integer values */
  for(i=0; i<size; i++){
    x[i] = r1;
    for(j=0; j<size; j++){
      A[i][j] = r2;
    }
  }

  clock_gettime(CLOCK, &start);

  /* perform matrix-vector product */
#pragma omp parallel for schedule(static) default(none) shared(size, A, x, y) private(i, j)
  for(i=0; i<size; i++){
    for(j=0; j<size; j++){
      y[i] = y[i] + A[i][j] * x[j];
    }
  }

  clock_gettime(CLOCK, &end);

  elapsed_time_hr(start, end, "Dense Matrix-Vector product.");

  /* print result so compiler does not throw it away */
  printf("Result vector y[0] = %f\n", y[0]);


  free(x);
  free(y);
  for(i =0; i<size; i++) free(A[i]);
  free(A);

  return 0;

}

/*
 * Dense Matrix-Vector product, doubles
 *
 * y = A * x
 * where A is a square matrix
 *
 * Input:  number of elements in vectors and of rows/cols
 *         in matrix specified as number of floats
 *
 */
int double_dmatvec_product(unsigned int size){

  int i,j;
  double r1,r2;

  /* create two vectors */
  double *x = (double *)malloc(size * sizeof(double));
  double *y = (double *)calloc(size, sizeof(double));

  if(x == NULL || y == NULL){
    printf("Out Of Memory: could not allocate space for the vectors.\n");
    return 0;
  }


  /* create matrix */
  double **A;
  A = (double **) malloc(size * sizeof(double *));
  if(A == NULL){
    printf("Out Of Memory: could not allocate space for the matrix.\n");
    return 0;
  }

  for(i=0; i<size; i++){
    A[i] = (double *)malloc(size * sizeof(double));
    if(A[i] == NULL){
      printf("Out Of Memory: could not allocate space for the matrix.\n");
      return 0;
    }
  }

  struct timespec start, end;

  r1 = VNI;
  r2 = VNI;

  /* fill vector x and matrix A with random integer values */
  for(i=0; i<size; i++){
    x[i] = r1;
    for(j=0; j<size; j++){
      A[i][j] = r2;
    }
  }

  clock_gettime(CLOCK, &start);

  /* perform matrix-vector product */
#pragma omp parallel for schedule(static) default(none) shared(size, A, x, y) private(i, j)
  for(i=0; i<size; i++){
    for(j=0; j<size; j++){
      y[i] = y[i] + A[i][j] * x[j];
    }
  }

  clock_gettime(CLOCK, &end);
  elapsed_time_hr(start, end, "Dense Matrix-Vector product.");

  /* print result so compiler does not throw it away */
  printf("Result vector y[0] = %f\n", y[0]);


  free(x);
  free(y);
  for(i =0; i<size; i++) free(A[i]);
  free(A);

  return 0;

}
