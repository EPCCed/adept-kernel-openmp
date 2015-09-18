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
#include <string.h>

#include "level1.h"


/* Level 1 benchmark driver - calls appropriate function */
/* based on command line arguments.                      */
void bench_level1(char *b, unsigned int s, unsigned long r, char *o, char *dt ){

  /* BLAS operations */
  if(strcmp(b, "blas_op") == 0){

    if(strcmp(o, "dot_product") == 0){

      if(strcmp(dt, "int") == 0) int_dot_product(s);
      else if(strcmp(dt, "float") == 0) float_dot_product(s);
      else if(strcmp(dt, "double") == 0) double_dot_product(s);
      else fprintf(stderr, "ERROR: check you are using a valid data type...\n");

    }

    else if(strcmp(o, "scalar_product") == 0){

      if(strcmp(dt, "int") == 0) int_scalar_mult(s);
      else if(strcmp(dt, "float") == 0) float_scalar_mult(s);
      else if(strcmp(dt, "double") == 0) double_scalar_mult(s);
      else fprintf(stderr, "ERROR: check you are using a valid data type...\n");

    }

    else if(strcmp(o, "norm") == 0){

      if(strcmp(dt, "int") == 0) int_norm(s);
      else if(strcmp(dt, "float") == 0) float_norm(s);
      else if(strcmp(dt, "double") == 0) double_norm(s);
      else fprintf(stderr, "ERROR: check you are using a valid data type...\n");

    }

    else if(strcmp(o, "axpy") == 0){

      if(strcmp(dt, "int") == 0) int_axpy(s);
      else if(strcmp(dt, "float") == 0) float_axpy(s);
      else if(strcmp(dt, "double") == 0) double_axpy(s);
      else fprintf(stderr, "ERROR: check you are using a valid data type...\n");

    }

    else if(strcmp(o, "dmv") == 0){

      if(strcmp(dt, "int") == 0) int_dmatvec_product(s);
      else if(strcmp(dt, "float") == 0) float_dmatvec_product(s);
      else if(strcmp(dt, "double") == 0) double_dmatvec_product(s);
      else fprintf(stderr, "ERROR: check you are using a valid data type...\n");

    }

  }

  /* Stencil codes */
  else if (strcmp(b, "stencil") == 0){

    /* o is set to "dot_product" by default. Use this to check for a default */
    if( strcmp(o, "27") == 0 || strcmp(o, "dot_product") == 0){
      if(strcmp(dt, "double") == 0) double_stencil27(s);
      else if (strcmp(dt, "float") == 0) float_stencil27(s);
      else {
        fprintf(stderr, "ERROR: check you are using a valid data type...\n");
      }
    }

    else if(strcmp(o, "19") == 0){
      if(strcmp(dt, "double") == 0) double_stencil19(s);
      else if (strcmp(dt, "float") == 0) float_stencil19(s);
      else {
        fprintf(stderr, "ERROR: check you are using a valid data type...\n");
      }
    }


    else if(strcmp(o, "9") == 0){
      if(strcmp(dt, "double") == 0) double_stencil9(s);
      else if (strcmp(dt, "float") == 0) float_stencil9(s);
      else {
        fprintf(stderr, "ERROR: check you are using a valid data type...\n");
      }
    }


    else if(strcmp(o, "5") == 0){
      if(strcmp(dt, "double") == 0) double_stencil5(s);
      else if (strcmp(dt, "float") == 0) float_stencil5(s);
      else {
        fprintf(stderr, "ERROR: check you are using a valid data type...\n");
      }
    }

    else fprintf(stderr, "ERROR: check you are using a valid operation type...\n");

  }

  else if (strcmp(b, "fileparse") == 0){
    fileparse(s);
  }

  else if (strcmp(b, "cg") == 0) {
    conjugate_gradient(s);
  }


  else fprintf(stderr, "ERROR: check you are using a valid benchmark...\n");


}
