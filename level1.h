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

void bench_level1(char *, unsigned int, unsigned long, char *, char *, char *);

int int_dot_product(unsigned int);
int float_dot_product(unsigned int);
int double_dot_product(unsigned int);

int int_scalar_mult(unsigned int);
int float_scalar_mult(unsigned int);
int double_scalar_mult(unsigned int);

int int_norm(unsigned int);
int float_norm(unsigned int);
int double_norm(unsigned int);

int int_axpy(unsigned int);
int float_axpy(unsigned int);
int double_axpy(unsigned int);

int int_dmatvec_product(unsigned int);
int float_dmatvec_product(unsigned int);
int double_dmatvec_product(unsigned int);

void double_stencil27(unsigned int);
void double_stencil19(unsigned int);
void double_stencil9(unsigned int);
void double_stencil5(unsigned int);

void float_stencil27(unsigned int);
void float_stencil19(unsigned int);
void float_stencil9(unsigned int);
void float_stencil5(unsigned int);

void fileparse(unsigned int);

int conjugate_gradient(unsigned int);
int conjugate_gradient_mixed(unsigned int);

/* Marsaglia's RNGs (fast on Odroid) */
/*
 * See
 * http://www.cse.yorku.ca/~oz/marsaglia-rng.html
 */

#define znew (z=36969*(z&65535)+(z>>16))
#define wnew (w=18000*(w&65535)+(w>>16))
#define MWC ((znew<<16)+wnew )
#define SHR3 (jsr^=(jsr<<17), jsr^=(jsr>>13), jsr^=(jsr<<5))
#define CONG (jcong=69069*jcong+1234567)
#define FIB ((b=a+b),(a=b-a))
#define KISS ((MWC^CONG)+SHR3)
#define LFIB4 (c++,t[c]=t[c]+t[UC(c+58)]+t[UC(c+119)]+t[UC(c+178)])
#define SWB (c++,bro=(x<y),t[c]=(x=t[UC(c+34)])-(y=t[UC(c+19)]+bro))
#define UNI (KISS*2.328306e-10)
#define VNI ((long) KISS)*4.656613e-10
#define UC (unsigned char) /*a cast operation*/
typedef unsigned long UL;
/* Global static variables: */
static UL z=362436069, w=521288629, jsr=123456789, jcong=380116160;
static UL a=224466889, b=7584631, t[256];
/* Use random seeds to reset z,w,jsr,jcong,a,b, and the table
t[256]*/
static UL x=0,y=0,bro; static unsigned char c=0;

