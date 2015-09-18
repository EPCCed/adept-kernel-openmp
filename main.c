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

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <limits.h>

#include "level1.h"
#include "utils.h"

void usage();

void settable(UL i1,UL i2,UL i3,UL i4,UL i5, UL i6){
  int i; z=i1;w=i2,jsr=i3; jcong=i4; a=i5; b=i6;
  for(i=0;i<256;i=i+1) t[i]=KISS;
}

int main(int argc, char **argv){

  int c;

  /* Init RNG here */
  settable(12345,65435,34221,12345,9983651,95746118);

  char *bench = "blas_op";
  unsigned int size = 200;
  unsigned long rep = 1;
  char *op  = "dot_product";
  char *dt = "double";

  static struct option option_list[] =
    { {"bench", required_argument, NULL, 'b'},
      {"size", required_argument, NULL, 's'},
      {"reps", required_argument, NULL, 'r'},
      {"op", required_argument, NULL, 'o'},
      {"dtype", required_argument, NULL, 'd'},
      {"help", no_argument, NULL, 'h'},
      {0, 0, 0, 0}
    };

  while((c = getopt_long(argc, argv, "b:s:r:o:d:h", option_list, NULL)) != -1){
    switch(c){
    case 'b':
      bench = optarg;
      printf("Benchmark is %s.\n", bench);
      break;
    case 's':
      size = atoi(optarg);
      printf("Size is %d.\n", size);
      break;
    case 'r':
      rep = atol(optarg);
      printf("Number of repetitions %lu.\n", rep);
      break;
    case 'o':
      op = optarg;
      printf("Operation %s\n", op);
      break;
    case 'd':
      dt = optarg;
      printf("Data type is %s\n", dt);
      break;
    case 'h':
      usage();
      return 0;
    default:
      printf("Undefined.\n");
      return 0;
    }
  }

  bench_level1(bench, size, rep, op, dt);

  return 0;

}

void usage(){

  printf("Usage for OpenMP KERNEL benchmarks:\n\n");
  printf("\t -b, --bench NAME \t NAME of the benchmark - possible values are blas_op, stencil, fileparse and cg.\n");
  printf("\t -s, --size N \t\t N number of elements, default is 200.\n"
		 "\t\t\t\t --> for the fileparse benchmark this is the number of rows.\n"
		 "\t\t\t\t --> for CG it is the number of rows/columns of the matrix.\n"
		 "\t\t\t\t --> for stencil, size dictates to size of the work buffer.\n"
		 "\t\t\t\t     It is size^2 for 5 and 9 point stencils, and size^3 for 19 and 27 point stencils.\n"
		 "\t\t\t\t --> for the BLAS operations it is the vector length or size of the matrix. \n");
  printf("\t -r, --reps N \t\t N number of repetitions. Default value is 1.\n"
		 "\t\t\t\t --> for the BLAS operations spmv and spgemm, this can be used to run the operation multiple times.\n");
  printf("\t -o, --op TYPE \t\t TYPE of operation.\n");
  printf("\t\t\t\t --> for BLAS benchmark: \"dot_product\", \"scalar_mult\", \"norm\", \"axpy\" and \"dmv\". \n"
		 "\t\t\t\t     Default is \"dot_product\".\n");
  printf("\t\t\t\t --> for stencil benchmark: \"27\", \"19\", \"9\" and \"5\". \n"
		 "\t\t\t\t     Default is \"27\".\n");
  printf("\t -d, --dtype DATATYPE \t DATATYPE to be used for the BLAS benchmarks. Default is double.\n"
		 "\t\t\t\t --> for norm, dot_product, scalar_product, axpy and dmv possible values are int, float, double.\n"
		 "\t\t\t\t --> for spmv and spgemm possible values are float, double.\n");
  printf("\t -h, --help \t\t Displays this help.\n");
  printf("\n\n");

}
