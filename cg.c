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
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#include "utils.h"

#define PCG_TOLERANCE 1e-3
#define PCG_MAX_ITER 1000
#define PCG_FLOAT_TOLERANCE 1e-2

/* Conjugate gradient benchmark */


/* struct for CSR matrix type */
typedef struct {
	int     nrow;
	int     ncol;
	int     nzmax;
	int    *colIndex;
	int    *rowStart;
	double *values;
} CSRmatrix;

typedef struct
{
  int     nrow;
  int     ncol;
  int     nzmax;
  int    *colIndex;
  int    *rowStart;
  float  *values;
} CSRmatrixF;

/*
 *
 * Sparse matrix and vector utility functions
 *
 */
static void CSR_matrix_vector_mult(CSRmatrix *A, double *x, double *b) {
	int i, j;
#pragma omp parallel for schedule(static) private(j)
	for (i = 0; i < A->nrow; i++) {
		double sum = 0.0;
		for (j = A->rowStart[i]; j < A->rowStart[i+1]; j++) {
			sum += A->values[j] * x[A->colIndex[j]];
		}
		b[i] = sum;
	}
}

static void CSR_matrix_vector_multF(CSRmatrixF *A, float *x, float *b) {
	int i, j;
#pragma omp parallel for schedule(static) private(j)
	for (i = 0; i < A->nrow; i++) {
		float sum = 0.0;
		for (j = A->rowStart[i]; j < A->rowStart[i+1]; j++) {
			sum += A->values[j] * x[A->colIndex[j]];
		}
		b[i] = sum;
	}
}

static double dotProduct(double *v1, double *v2, int size) {
	int i;
	double result = 0.0;
#pragma omp parallel for schedule(static) reduction(+:result)
	for (i = 0; i < size; i++) {
		result += v1[i] * v2[i];
	}
	return result;
}

static double dotProductF(float *v1, float *v2, int size) {
	int i;
	float result = 0.0;
#pragma omp parallel for schedule(static) reduction(+:result)
	for (i = 0; i < size; i++) {
		result += v1[i] * v2[i];
	}
	return result;
}

static void vecAxpy(double *x, double *y, int size, double alpha) {
	int i;
#pragma omp parallel for schedule(static)
	for (i = 0; i < size; i++) {
		y[i] = y[i] + alpha * x[i];
	}
}

static void vecAxpyF(float *x, float *y, int size, float alpha) {
	int i;
#pragma omp parallel for schedule(static)
	for (i = 0; i < size; i++) {
		y[i] = y[i] + alpha * x[i];
	}
}


static void vecAypx(double *x, double *y, int size, double alpha) {
	int i;
#pragma omp parallel for schedule(static)
	for (i = 0; i < size; i++) {
		y[i] = alpha * y[i] + x[i];
	}
}

static void vecAypxF(float *x, float *y, int size, float alpha) {
	int i;
#pragma omp parallel for schedule(static)
	for (i = 0; i < size; i++) {
		y[i] = alpha * y[i] + x[i];
	}
}


int conjugate_gradient(int s) {
	CSRmatrix *A;
	int i;
	double *x, *b, *r, *p, *omega;
	int k;
	double r0, r1, beta, dot, alpha;
	double tol = PCG_TOLERANCE * PCG_TOLERANCE;

	struct timespec start, end;

	/*======================================================================
	 *
	 * generate a random matrix of size s x s
	 *
	 *======================================================================*/
	A = malloc(sizeof(CSRmatrix));
	A->nrow = s;
	A->ncol = s;
	A->nzmax = s;
	A->colIndex = malloc(A->nzmax * sizeof(int));
	A->rowStart = malloc((A->nrow+1) * sizeof(int));
	A->values = malloc(A->nzmax * sizeof(double));

	/* generate structure for matrix */
#pragma omp parallel for schedule(static)
	for (i = 0; i < A->nrow; i++) {
		A->rowStart[i] = i;
		A->colIndex[i] = i;
	}
	A->rowStart[i] = i;

	/* now generate values for matrix */
	srand((unsigned int)time(NULL));

#pragma omp parallel for schedule(static)
	for (i = 0; i < A->nzmax; i++) {
		A->values[i] = rand() / 32768.0;
	}

	/*
	 *
	 * Initialise vectors
	 *
	 */

	/* allocate vectors (unknowns, RHS and temporaries) */
	x = malloc(s * sizeof(double));
	b = malloc(s * sizeof(double));
	r = malloc(s * sizeof(double));
	p = malloc(s * sizeof(double));
	omega = malloc(s * sizeof(double));

	/* generate a random vector of size s for the unknowns */
#pragma omp parallel for schedule(static)
	for (i = 0; i < s; i++) {
		x[i] = rand() / 32768.0;
	}

	/* multiply matrix by vector to get RHS */
	CSR_matrix_vector_mult(A, x, b);

	/* clear initial guess and initialise temporaries */
#pragma omp parallel for schedule(static)
	for (i = 0; i < s; i++) {
		x[i] = 0.0;

		/* r = b - Ax; since x is 0, r = b */
		r[i] = b[i];

		/* p = r ( = b)*/
		p[i] = b[i];

		omega[i] = 0.0;
	}

	/* compute initial residual */
	r1 = dotProduct(r, r, s);
	r0 = r1;

	/*
	 *
	 * Actual solver loop
	 *
	 */
	k = 0;

	clock_gettime(CLOCK, &start);

	while ((r1 > tol) && (k <= PCG_MAX_ITER)) {
		/* omega = Ap */
		CSR_matrix_vector_mult(A, p, omega);

		/* dot = p . omega */
		dot = dotProduct(p, omega, s);

		alpha = r1 / dot;

		/* x = x + alpha.p */
		vecAxpy(p, x, s, alpha);

		/* r = r - alpha.omega */
		vecAxpy(omega, r, s, -alpha);

		r0 = r1;

		/* r1 = r . r */
		r1 = dotProduct(r, r, s);

		beta = r1 / r0;

		/* p = r + beta.p */
		vecAypx(r, p, s, beta);
		k++;
	}

	clock_gettime(CLOCK, &end);
	elapsed_time_hr(start, end, "Conjugate gradient solve.");

	/*
	 *
	 * Free memory
	 *
	 */

	/* free the vectors */
	free(omega);
	free(p);
	free(r);
	free(b);
	free(x);

	/* free the matrix */
	free(A->colIndex);
	free(A->rowStart);
	free(A->values);
	free(A);
	return 0;
}


/* mixed precision version */
int conjugate_gradient_mixed(unsigned int s)
{
    CSRmatrix *A;
    CSRmatrixF *AF;
    int i;
    double *x, *b, *r, *p, *omega;
    float *xf, *bf, *rf, *pf, *omegaf;
    int k;
    double r0, r1, beta, dot, alpha;
    float r0f, r1f, betaf, dotf, alphaf;
    double tol = PCG_FLOAT_TOLERANCE * PCG_FLOAT_TOLERANCE;

    struct timespec start, end;

    /*======================================================================
     *
     * generate a random matrix of size s x s
     *
     *======================================================================*/
    A = malloc(sizeof(CSRmatrix));
    A->nrow = s;
    A->ncol = s;
    A->nzmax = s;
    A->colIndex = malloc(A->nzmax * sizeof(int));
    A->rowStart = malloc((A->nrow+1) * sizeof(int));
    A->values = malloc(A->nzmax * sizeof(double));

    AF = malloc(sizeof(CSRmatrixF));
    AF->nrow = s;
    AF->ncol = s;
    AF->nzmax = s;
    AF->colIndex = malloc(AF->nzmax * sizeof(int));
    AF->rowStart = malloc((AF->nrow+1) * sizeof(int));
    AF->values = malloc(AF->nzmax * sizeof(float));

    /* generate structure for matrix */
#pragma omp parallel for schedule(static)
    for (i = 0; i < A->nrow; i++) {
	A->rowStart[i] = i;
	A->colIndex[i] = i;

	AF->rowStart[i] = i;
	AF->colIndex[i] = i;
    }
    A->rowStart[i] = i;
    AF->rowStart[i] = i;

    /* now generate values for matrix */
    srand((unsigned int)time(NULL));

#pragma omp parallel for schedule(static)
    for (i = 0; i < A->nzmax; i++) {
	A->values[i] = rand() / 32768.0;
	AF->values[i] = (float)A->values[i];
    }

    /*======================================================================
     *
     * Initialise vectors
     *
     *======================================================================*/
    /* allocate vectors (unknowns, RHS and temporaries) */
    x = malloc(s * sizeof(double));
    b = malloc(s * sizeof(double));
    r = malloc(s * sizeof(double));
    p = malloc(s * sizeof(double));
    omega = malloc(s * sizeof(double));

    xf = malloc(s * sizeof(float));
    bf = malloc(s * sizeof(float));
    rf = malloc(s * sizeof(float));
    pf = malloc(s * sizeof(float));
    omegaf = malloc(s * sizeof(float));

    /* generate a random vector of size s for the unknowns */
#pragma omp parallel for schedule(static)
    for (i = 0; i < s; i++) {
	x[i] = rand() / 32768.0;
	xf[i] = (float)x[i];
    }

    /* multiply matrix by vector to get RHS */
    CSR_matrix_vector_mult(A, x, b);
    CSR_matrix_vector_multF(AF, xf, bf);

    /* clear initial guess and initialise temporaries */
#pragma omp parallel for schedule(static)
    for (i = 0; i < s; i++) {
	x[i] = 0.0;
	xf[i] = 0.0;

	/* r = b - Ax; since x is 0, r = b */
	r[i] = b[i];
	rf[i] = bf[i];

	/* p = r ( = b)*/
	p[i] = b[i];
	pf[i] = bf[i];

	omega[i] = 0.0;
	omegaf[i] = 0.0;
    }


    clock_gettime(CLOCK, &start);

    /* compute initial residual */
    r1f = dotProductF(rf, rf, s);
    r0f = r1f;

    /*======================================================================
     *
     * Actual solver loop (single precision)
     *
     *======================================================================*/
    k = 0;
    while ((r1f > tol) && (k <= PCG_MAX_ITER)) {
	/* omega = Ap */
	CSR_matrix_vector_multF(AF, pf, omegaf);

	/* dot = p . omega */
	dotf = dotProductF(pf, omegaf, s);

	alphaf = r1f / dotf;

	/* x = x + alpha.p */
	vecAxpyF(pf, xf, s, alphaf);

	/* r = r - alpha.omega */
	vecAxpyF(omegaf, rf, s, -alphaf);

	r0f = r1f;

	/* r1 = r . r */
	r1f = dotProductF(rf, rf, s);

	betaf = r1f / r0f;

	/* p = r + beta.p */
	vecAypxF(rf, pf, s, betaf);
	k++;
    }

    /* convert for double precision iterations */
    r1 = (double)r1f;
    r0 = (double)r0f;
#pragma omp parallel for schedule(static)
    for (i = 0; i < s; i++) {
	r[i] = (double)rf[i];
	p[i] = (double)pf[i];
	x[i] = (double)xf[i];
    }

    tol = PCG_TOLERANCE * PCG_TOLERANCE;

    /*======================================================================
     *
     * Actual solver loop
     *
     *======================================================================*/
    while ((r1 > tol) && (k <= PCG_MAX_ITER)) {
	/* omega = Ap */
	CSR_matrix_vector_mult(A, p, omega);

	/* dot = p . omega */
	dot = dotProduct(p, omega, s);

	alpha = r1 / dot;

	/* x = x + alpha.p */
	vecAxpy(p, x, s, alpha);

	/* r = r - alpha.omega */
	vecAxpy(omega, r, s, -alpha);

	r0 = r1;

	/* r1 = r . r */
	r1 = dotProduct(r, r, s);

	beta = r1 / r0;

	/* p = r + beta.p */
	vecAypx(r, p, s, beta);
	k++;
    }

    clock_gettime(CLOCK, &end);
    elapsed_time_hr(start, end, "Conjugate gradient solve.");

    /*======================================================================
     *
     * Free memory
     *
     *======================================================================*/
    /* free the vectors */
    free(omega);
    free(p);
    free(r);
    free(b);
    free(x);

    free(omegaf);
    free(pf);
    free(rf);
    free(bf);
    free(xf);


    /* free the matrix */
    free(A->colIndex);
    free(A->rowStart);
    free(A->values);
    free(A);

    free(AF->colIndex);
    free(AF->rowStart);
    free(AF->values);
    free(AF);

    return 0;
}
