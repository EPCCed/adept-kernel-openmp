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
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <omp.h>

#include "level1.h"
#include "utils.h"

#define REPS 100


void float_stencil27(unsigned int size){

	int i, j, k, iter;
	int n = size-2;
	float fac = 1.0/26;

	/* Work buffers, with halos */
	float *a0 = (float*)malloc(sizeof(float)*size*size*size);
	float *a1 = (float*)malloc(sizeof(float)*size*size*size);

	if(a0==NULL||a1==NULL){
		/* Something went wrong in the memory allocation here, fail gracefully */
		printf("27-point Single Precision Stencil Error: Unable to allocate memory\n");
	}

#pragma omp parallel 
	{
		if (omp_get_thread_num() == 0){
			printf("Running on Host with %d OpenMP thread(s):\n\n",omp_get_num_threads());
		}
	}

	struct timespec start, end;

	/* zero all of array (including halos) */
#pragma omp parallel for private(j,k)
	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			for (k = 0; k < size; k++) {
				a0[i*size*size+j*size+k] = 0.0;
			}
		}
	}

#pragma omp parallel for private(j,k)
	/* use random numbers to fill interior */
	for (i = 1; i < n+1; i++) {
		for (j = 1; j < n+1; j++) {
			for (k = 1; k < n+1; k++) {
				a0[i*size*size+j*size+k] = (float) rand()/ (float)(1.0 + RAND_MAX);
			}
		}
	}

	/* run main computation on host */
	clock_gettime(CLOCK, &start);

	for (iter = 0; iter < REPS; iter++) {
#pragma omp parallel
		{
#pragma omp for private(j,k)
			for (i = 1; i < n+1; i++) {
				for (j = 1; j < n+1; j++) {
					for (k = 1; k < n+1; k++) {
						a1[i*size*size+j*size+k] = (
								a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
								a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
								a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
								a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

								a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
								a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +
								a0[(i-1)*size*size+(j-1)*size+(k-1)] + a0[(i-1)*size*size+(j+1)*size+(k-1)] +
								a0[(i+1)*size*size+(j-1)*size+(k-1)] + a0[(i+1)*size*size+(j+1)*size+(k-1)] +

								a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
								a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +
								a0[(i-1)*size*size+(j-1)*size+(k+1)] + a0[(i-1)*size*size+(j+1)*size+(k+1)] +
								a0[(i+1)*size*size+(j-1)*size+(k+1)] + a0[(i+1)*size*size+(j+1)*size+(k+1)] +

								a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
						) * fac;
					}
				}
			}

#pragma omp for private(j,k)
			for (i = 1; i < n+1; i++) {
				for (j = 1; j < n+1; j++) {
					for (k = 1; k < n+1; k++) {
						a0[i*size*size+j*size+k] = a1[i*size*size+j*size+k];
					}
				}
			}
		} // end omp parallel region
	} /* end iteration loop */

	clock_gettime(CLOCK, &end);
	elapsed_time_hr(start, end, "Single Precision Stencil - 27 point");
	/* Free malloc'd memory to prevent leaks */
	free(a0);
	free(a1);

}


void double_stencil27(unsigned int size){

	int i, j, k, iter;
	int n = size-2;
	double fac = 1.0/26;

	/* Work buffers, with halos */
	double *a0 = (double*)malloc(sizeof(double)*size*size*size);
	double *a1 = (double*)malloc(sizeof(double)*size*size*size);

	if(a0==NULL||a1==NULL){
		/* Something went wrong in the memory allocation here, fail gracefully */
		printf("27-point Double Precision Stencil Error: Unable to allocate memory\n");
	}

#pragma omp parallel 
	{
		if (omp_get_thread_num() == 0){
			printf("Running on Host with %d OpenMP thread(s):\n\n",omp_get_num_threads());
		}
	}

	struct timespec start, end;

	/* zero all of array (including halos) */
#pragma omp parallel for private(j,k)
	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			for (k = 0; k < size; k++) {
				a0[i*size*size+j*size+k] = 0.0;
			}
		}
	}

#pragma omp parallel for private(j,k)
	/* use random numbers to fill interior */
	for (i = 1; i < n+1; i++) {
		for (j = 1; j < n+1; j++) {
			for (k = 1; k < n+1; k++) {
				a0[i*size*size+j*size+k] = (double) rand()/ (double)(1.0 + RAND_MAX);
			}
		}
	}

	/* run main computation on host */
	clock_gettime(CLOCK, &start);

	for (iter = 0; iter < REPS; iter++) {
#pragma omp parallel
		{
#pragma omp for private(j,k)
			for (i = 1; i < n+1; i++) {
				for (j = 1; j < n+1; j++) {
					for (k = 1; k < n+1; k++) {
						a1[i*size*size+j*size+k] = (
								a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
								a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
								a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
								a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

								a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
								a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +
								a0[(i-1)*size*size+(j-1)*size+(k-1)] + a0[(i-1)*size*size+(j+1)*size+(k-1)] +
								a0[(i+1)*size*size+(j-1)*size+(k-1)] + a0[(i+1)*size*size+(j+1)*size+(k-1)] +

								a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
								a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +
								a0[(i-1)*size*size+(j-1)*size+(k+1)] + a0[(i-1)*size*size+(j+1)*size+(k+1)] +
								a0[(i+1)*size*size+(j-1)*size+(k+1)] + a0[(i+1)*size*size+(j+1)*size+(k+1)] +

								a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
						) * fac;
					}
				}
			}

#pragma omp for private(j,k)
			for (i = 1; i < n+1; i++) {
				for (j = 1; j < n+1; j++) {
					for (k = 1; k < n+1; k++) {
						a0[i*size*size+j*size+k] = a1[i*size*size+j*size+k];
					}
				}
			}
		} // end omp parallel region
	} /* end iteration loop */

	clock_gettime(CLOCK, &end);
	elapsed_time_hr(start, end, "Double Precision Stencil - 27 point");

	/* Free malloc'd memory to prevent leaks */
	free(a0);
	free(a1);

}


void float_stencil19(unsigned int size){

	int i, j, k, iter;
	int n = size-2;
	float fac = 1.0/18;

	/* Work buffers, with halos */
	float *a0 = (float*)malloc(sizeof(float)*size*size*size);
	float *a1 = (float*)malloc(sizeof(float)*size*size*size);

	if(a0==NULL||a1==NULL){
		/* Something went wrong in the memory allocation here, fail gracefully */
		printf("19-point Single Precision Stencil Error: Unable to allocate memory\n");
	}

#pragma omp parallel 
	{
		if (omp_get_thread_num() == 0){
			printf("Running on Host with %d OpenMP thread(s):\n\n",omp_get_num_threads());
		}
	}

	struct timespec start,end;

	/* zero all of array (including halos) */
#pragma omp parallel for private(j,k)
	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			for (k = 0; k < size; k++) {
				a0[i*size*size+j*size+k] = 0.0;
			}
		}
	}

	/* use random numbers to fill interior */
#pragma omp parallel for private(j,k)
	for (i = 1; i < n+1; i++) {
		for (j = 1; j < n+1; j++) {
			for (k = 1; k < n+1; k++) {
				a0[i*size*size+j*size+k] = (float) rand()/ (float)(1.0 + RAND_MAX);
			}
		}
	}

	/* run main computation on host */

	clock_gettime(CLOCK, &start);
	for (iter = 0; iter < REPS; iter++) {
#pragma omp parallel
		{
#pragma omp for private(j,k)
			for (i = 1; i < n+1; i++) {
				for (j = 1; j < n+1; j++) {
					for (k = 1; k < n+1; k++) {
						a1[i*size*size+j*size+k] = (
								a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
								a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
								a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
								a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

								a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
								a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +

								a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
								a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +

								a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
						) * fac;
					}
				}
			}

#pragma omp for private(j,k)
			for (i = 1; i < n+1; i++) {
				for (j = 1; j < n+1; j++) {
					for (k = 1; k < n+1; k++) {
						a0[i*size*size+j*size+k] = a1[i*size*size+j*size+k];
					}
				}
			}
		} // end omp parallel region
	} /* end iteration loop */

	clock_gettime(CLOCK, &end);
	elapsed_time_hr(start, end, "Single Precision Stencil - 19 point");

	/* Free malloc'd memory to prevent leaks */
	free(a0);
	free(a1);

}

void double_stencil19(unsigned int size){

	int i, j, k, iter;
	int n = size-2;
	double fac = 1.0/18;

	/* Work buffers, with halos */
	double *a0 = (double*)malloc(sizeof(double)*size*size*size);
	double *a1 = (double*)malloc(sizeof(double)*size*size*size);

	if(a0==NULL||a1==NULL){
		/* Something went wrong in the memory allocation here, fail gracefully */
		printf("19-point Double Precision Stencil Error: Unable to allocate memory\n");
	}

#pragma omp parallel 
	{
		if (omp_get_thread_num() == 0){
			printf("Running on Host with %d OpenMP thread(s):\n\n",omp_get_num_threads());
		}
	}

	struct timespec start,end;

	/* zero all of array (including halos) */
#pragma omp parallel for private(j,k)
	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			for (k = 0; k < size; k++) {
				a0[i*size*size+j*size+k] = 0.0;
			}
		}
	}

	/* use random numbers to fill interior */
#pragma omp parallel for private(j,k)
	for (i = 1; i < n+1; i++) {
		for (j = 1; j < n+1; j++) {
			for (k = 1; k < n+1; k++) {
				a0[i*size*size+j*size+k] = (double) rand()/ (double)(1.0 + RAND_MAX);
			}
		}
	}

	/* run main computation on host */

	clock_gettime(CLOCK, &start);
	for (iter = 0; iter < REPS; iter++) {
#pragma omp parallel
		{
#pragma omp for private(j,k)
			for (i = 1; i < n+1; i++) {
				for (j = 1; j < n+1; j++) {
					for (k = 1; k < n+1; k++) {
						a1[i*size*size+j*size+k] = (
								a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
								a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
								a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
								a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

								a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
								a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +

								a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
								a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +

								a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
						) * fac;
					}
				}
			}

#pragma omp for private(j,k)
			for (i = 1; i < n+1; i++) {
				for (j = 1; j < n+1; j++) {
					for (k = 1; k < n+1; k++) {
						a0[i*size*size+j*size+k] = a1[i*size*size+j*size+k];
					}
				}
			}
		} // end omp parallel region
	} /* end iteration loop */

	clock_gettime(CLOCK, &end);
	elapsed_time_hr(start, end, "Double Precision Stencil - 19 point");

	/* Free malloc'd memory to prevent leaks */
	free(a0);
	free(a1);

}


void float_stencil9(unsigned int size){

	int i, j, iter;
	int n = size-2;
	float fac = 1.0/8;

	/* Work buffers, with halos */
	float *a0 = (float*)malloc(sizeof(float)*size*size);
	float *a1 = (float*)malloc(sizeof(float)*size*size);

	if(a0==NULL||a1==NULL){
		/* Something went wrong in the memory allocation here, fail gracefully */
		printf("9-point Single Precision Stencil Error: Unable to allocate memory\n");
	}

#pragma omp parallel 
	{
		if (omp_get_thread_num() == 0){
			printf("Running on Host with %d OpenMP thread(s):\n\n",omp_get_num_threads());
		}
	}

	struct timespec start,end;

	/* zero all of array (including halos) */
#pragma omp parallel for private(j)
	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			a0[i*size+j] = 0.0;
		}
	}

	/* use random numbers to fill interior */
#pragma omp parallel for private(j)
	for (i = 1; i < n+1; i++) {
		for (j = 1; j < n+1; j++) {
			a0[i*size+j] = (float) rand()/ (float)(1.0 + RAND_MAX);
		}
	}

	/* run main computation on host */
	clock_gettime(CLOCK, &start);
	for (iter = 0; iter < REPS; iter++) {
#pragma omp parallel
		{
#pragma omp for private(j)
			for (i = 1; i < n+1; i++) {
				for (j = 1; j < n+1; j++) {
					a1[i*size+j] = (
							a0[i*size+(j-1)] + a0[i*size+(j+1)] +
							a0[(i-1)*size+j] + a0[(i+1)*size+j] +
							a0[(i-1)*size+(j-1)] + a0[(i-1)*size+(j+1)] +
							a0[(i+1)*size+(j-1)] + a0[(i+1)*size+(j+1)]

					) * fac;
				}
			}

#pragma omp for private(j)
			for (i = 1; i < n+1; i++) {
				for (j = 1; j < n+1; j++) {
					a0[i*size+j] = a1[i*size+j];
				}
			}
		} // end omp parallel for
	} /* end iteration loop */

	clock_gettime(CLOCK, &end);
	elapsed_time_hr(start, end, "Single Precision Stencil - 9 point");

	/* Free malloc'd memory to prevent leaks */
	free(a0);
	free(a1);

}


void double_stencil9(unsigned int size){

	int i, j, iter;
	int n = size-2;
	double fac = 1.0/8;

	/* Work buffers, with halos */
	double *a0 = (double*)malloc(sizeof(double)*size*size);
	double *a1 = (double*)malloc(sizeof(double)*size*size);

	if(a0==NULL||a1==NULL){
		/* Something went wrong in the memory allocation here, fail gracefully */
		printf("9-point Double Precision Stencil Error: Unable to allocate memory\n");
	}

#pragma omp parallel 
	{
		if (omp_get_thread_num() == 0){
			printf("Running on Host with %d OpenMP thread(s):\n\n",omp_get_num_threads());
		}
	}

	struct timespec start,end;

	/* zero all of array (including halos) */
#pragma omp parallel for private(j)
	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			a0[i*size+j] = 0.0;
		}
	}

	/* use random numbers to fill interior */
#pragma omp parallel for private(j)
	for (i = 1; i < n+1; i++) {
		for (j = 1; j < n+1; j++) {
			a0[i*size+j] = (double) rand()/ (double)(1.0 + RAND_MAX);
		}
	}

	/* run main computation on host */
	clock_gettime(CLOCK, &start);
	for (iter = 0; iter < REPS; iter++) {
#pragma omp parallel
		{
#pragma omp for private(j)
			for (i = 1; i < n+1; i++) {
				for (j = 1; j < n+1; j++) {
					a1[i*size+j] = (
							a0[i*size+(j-1)] + a0[i*size+(j+1)] +
							a0[(i-1)*size+j] + a0[(i+1)*size+j] +
							a0[(i-1)*size+(j-1)] + a0[(i-1)*size+(j+1)] +
							a0[(i+1)*size+(j-1)] + a0[(i+1)*size+(j+1)]

					) * fac;
				}
			}

#pragma omp for private(j)
			for (i = 1; i < n+1; i++) {
				for (j = 1; j < n+1; j++) {
					a0[i*size+j] = a1[i*size+j];
				}
			}
		} // end omp parallel for
	} /* end iteration loop */

	clock_gettime(CLOCK, &end);
	elapsed_time_hr(start, end, "Double Precision Stencil - 9 point");

	/* Free malloc'd memory to prevent leaks */
	free(a0);
	free(a1);

}


void float_stencil5(unsigned int size){

	int i, j, iter;
	int n = size-2;
	float fac = 1.0/8;

	/* Work buffers, with halos */
	float *a0 = (float*)malloc(sizeof(float)*size*size);
	float *a1 = (float*)malloc(sizeof(float)*size*size);

	if(a0==NULL||a1==NULL){
		/* Something went wrong in the memory allocation here, fail gracefully */
		printf("9-point Single Precision Stencil Error: Unable to allocate memory\n");
	}
#pragma omp parallel 
	{
		if (omp_get_thread_num() == 0){
			printf("Running on Host with %d OpenMP thread(s):\n\n",omp_get_num_threads());
		}
	}

	struct timespec start,end;

	/* zero all of array (including halos) */
#pragma omp parallel for private(j)
	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			a0[i*size+j] = 0.0;
		}
	}

	/* use random numbers to fill interior */
#pragma omp parallel for private(j)
	for (i = 1; i < n+1; i++) {
		for (j = 1; j < n+1; j++) {
			a0[i*size+j] = (float) rand()/ (float)(1.0 + RAND_MAX);
		}
	}

	/* run main computation on host */
	clock_gettime(CLOCK, &start);
	for (iter = 0; iter < REPS; iter++) {
#pragma omp parallel
		{
#pragma omp for private(j)
			for (i = 1; i < n+1; i++) {
				for (j = 1; j < n+1; j++) {
					a1[i*size+j] = (
							a0[i*size+(j-1)] + a0[i*size+(j+1)] +
							a0[(i-1)*size+j] + a0[(i+1)*size+j]
					) * fac;
				}
			}

#pragma omp for private(j)
			for (i = 1; i < n+1; i++) {
				for (j = 1; j < n+1; j++) {
					a0[i*size+j] = a1[i*size+j];
				}
			}
		} //end omp parallel region
	} /* end iteration loop */

	clock_gettime(CLOCK, &end);
	elapsed_time_hr(start, end, "Single Precision Stencil - 5 point");

	/* Free malloc'd memory to prevent leaks */
	free(a0);
	free(a1);

}


void double_stencil5(unsigned int size){

	int i, j, iter;
	int n = size-2;
	double fac = 1.0/8;

	/* Work buffers, with halos */
	double *a0 = (double*)malloc(sizeof(double)*size*size);
	double *a1 = (double*)malloc(sizeof(double)*size*size);

	if(a0==NULL||a1==NULL){
		/* Something went wrong in the memory allocation here, fail gracefully */
		printf("9-point Double Precision Stencil Error: Unable to allocate memory\n");
	}

#pragma omp parallel 
	{
		if (omp_get_thread_num() == 0){
			printf("Running on Host with %d OpenMP thread(s):\n\n",omp_get_num_threads());
		}
	}

	struct timespec start,end;

	/* zero all of array (including halos) */
#pragma omp parallel for private(j)
	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			a0[i*size+j] = 0.0;
		}
	}

	/* use random numbers to fill interior */
#pragma omp parallel for private(j)
	for (i = 1; i < n+1; i++) {
		for (j = 1; j < n+1; j++) {
			a0[i*size+j] = (double) rand()/ (double)(1.0 + RAND_MAX);
		}
	}

	/* run main computation on host */
	clock_gettime(CLOCK, &start);
	for (iter = 0; iter < REPS; iter++) {
#pragma omp parallel
		{
#pragma omp for private(j)
			for (i = 1; i < n+1; i++) {
				for (j = 1; j < n+1; j++) {
					a1[i*size+j] = (
							a0[i*size+(j-1)] + a0[i*size+(j+1)] +
							a0[(i-1)*size+j] + a0[(i+1)*size+j]
						  ) * fac;
				}
			}

#pragma omp for private(j)
			for (i = 1; i < n+1; i++) {
				for (j = 1; j < n+1; j++) {
					a0[i*size+j] = a1[i*size+j];
				}
			}
		} //end omp parallel region
	} /* end iteration loop */

	clock_gettime(CLOCK, &end);
	elapsed_time_hr(start, end, "Double Precision Stencil - 5 point");

	/* Free malloc'd memory to prevent leaks */
	free(a0);
	free(a1);

}
