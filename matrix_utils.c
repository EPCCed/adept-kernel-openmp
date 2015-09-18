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

/*
 * Utility functions for sparse matrices.
 *
 * Currently focusses on reading in a sparse matrix file in
 * Matrix Market Format (http://math.nist.gov/MatrixMarket)
 * converting this to CSR.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/*
 *
 * reads matrix market file header and get number of rows,
 * number of columns, and number of non-zero elements.
 *
 */

void get_matrix_size(char *fn, int *rows, int *cols, int *nonzeros){
  FILE *f;
  char header[64];
  char *rv = NULL;

  if ((f = fopen(fn, "r")) == NULL) {
    printf ("can't open file <%s> \n", fn);
    exit(1);
  }

  rv = fgets(header, sizeof(header), f);
  if (*rv==EOF){
    printf("Error reading file.\n");
    exit(1);
  }
  rv = fgets(header, sizeof(header), f);
  if (*rv==EOF){
    printf("Error reading file.\n");
    exit(1);
  }else{
    sscanf(header, "%d %d %d", rows, cols, nonzeros);
  }

  printf("Rows: %d, Columns: %d, Non-zeros: %d\n", *rows, *cols, *nonzeros);
  fclose(f);

}

/*
 *
 * convert a matrix in Matrix Market Format (COO) to CSR
 *
 */
void  mm_to_csr (char *fn, int m, int n, int nz, int *row_idx, int *col_idx, double *values)
{

  FILE *fin, *fout;
  int i,j;
  int base;
  char body[64];
  int  row_idx_current, inc;
  char *rv = NULL;

  int *new_row_idx, *new_col_idx;
  double *new_values;

  if ((fin = fopen(fn, "r")) == NULL) {
    printf ("can't open input file <%s> \n", fn);
    exit(1);
  }

  printf("here\n");

  if((fout = fopen("matrix_in.csr","w")) == NULL) {
    printf ("can't open output file <%s> \n", fn);
    exit(1);
  }

  /* discard first two lines */
  rv = fgets(body, sizeof(body), fin);
  if (*rv==EOF){
    printf("Error reading file.\n");
    exit(1);
  }
  rv = fgets(body, sizeof(body), fin);
  if (*rv==EOF){
    printf("Error reading file.\n");
    exit(1);
  }


  base = 1;
  i = 0;

  /* walk through the file line by line */
  while (fgets(body, sizeof(body), fin)){
    sscanf(body, "%d %d %lf", &row_idx[i], &col_idx[i], &values[i]);
    row_idx[i] -= base;  /* adjust from 1-based to 0-based */
    col_idx[i] -= base;
    i++;
  }

  fclose(fin);

  /* allocate space for new arrays which will hold the    */
  /* newly ordered values, and the column and row indices */
  new_row_idx = malloc(nz * sizeof(int));
  new_col_idx = malloc(nz * sizeof(int));
  new_values = malloc(nz * sizeof(double));

  /* set first values for all three arrays   */
  /* as there is nothing to be done for them */
  row_idx_current=row_idx[0];
  new_row_idx[0]=row_idx[0];
  new_col_idx[0]=col_idx[0];
  new_values[0]=values[0];

  inc=1;

  /* this is where the arrays are being reordered */
  for(j=1; j<nz; j++){
    for(i=1; i<nz; i++){
      if(row_idx[i] == row_idx_current){
        new_values[inc]=values[i];
        new_col_idx[inc]=col_idx[i];
        inc++;
      }
    }
    new_row_idx[j]=inc;
    row_idx_current++;
  }

  fprintf(fout, "%d %d %d\n", nz, nz, m+1);

  /* fprintf(fout, "Values:\n"); */
  /* copy the new colum indices and values into the old arrays */
  for (i=0; i<nz; i++)  {
    values[i]=new_values[i];
    fprintf(fout, "%f\n", new_values[i]);
  }

  /* fprintf(fout, "\nColumn indices:\n"); */
  /* copy the new colum indices and values into the old arrays */
  for (i=0; i<nz; i++)  {
    col_idx[i]=new_col_idx[i];
    fprintf(fout, "%d\n", new_col_idx[i]);
  }

  /* fprintf(fout, "\nRow pointers:\n"); */
  /* copy the new row indices into the old array */
  for (i=0; i<=m; i++)  {
    row_idx[i]=new_row_idx[i];
    fprintf(fout, "%d\n", new_row_idx[i]);
  }

  /* free memory for the temporary new arrays */
  free(new_row_idx);
  free(new_col_idx);
  free(new_values);

  fclose(fout);

}
