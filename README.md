Copyright (c) 2015 The University of Edinburgh.
 
This software was developed as part of the                       
EC FP7 funded project Adept (Project ID: 610490)                 
    http://www.adept-project.eu                                            

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


#Adept Kernel Benchmarks - OpenMP

This README describes the OpenMP parallelised kernel benchmarks. They are implemented in C.

## BLAS-type benchmarks

In our BLAS-type benchmarks we implement a few of the most common linear algebra computations.

#### AXPY
This benchmark takes two vectors `x` and `y`, and the scalar `a`, and computes:
``` 
  y = a * x + y
```
The user can choose the length (number of elements) of the vectors, as well as their data type (int, float or double).

#### Dot product 
The dot product benchmark multiplies two vectors x and y of length n and returns a scalar:
```
  result = x_0 y0 + x_1 y_1 + ... x_n y_n
```
The user can choose the length (number of elements) of the vectors, as well as their data type (int, float or double).

#### Scalar multiplication
Thise benchmark scales the vector x by a fixed scalar a:
```
  x = a * x
```
The user can choose the length (number of elements) of the vectors, as well as their data type (int, float or double).

#### Euclidean norm
This benchmarks computes for Euclidean norm of vector x:
```
  || x || = sqrt ( |x_1|^2 + |x_2|^2 + ... |x_n|^2 )
```
The user can choose the length (number of elements) of the vectors, as well as their data type (int, float or double).
  
#### Dense matrix-vector multiplication
This benchmarks multiplies a square dense matrix A with a vector x to compute vector y:
```
y = A * x
```
Both A and x are randomly generated. The user can choose the size of the data structures (where size*size equals the number of elements in the matrix), as well as their data type (int, float or double).
## Stencil computation

The stencil benchmarks compute values for each element in a 2D or 3D grid based on the values of their nearest neighbours.
 
#### 2D grid: 5-point and 9-point Stencil
On a 2D grid, the 5-point stencil computes the value of A[i][j] by taking the values from left, right, up and down from the current position, and scale them with a constant. The 9-point stencil is similar, but also includes the diagonals.
The user can choose the data type to be used in the grid (int, float or double).
 
#### 3D grid: 19-point and 27-point Stencil 
The 19-point and 27-point stencils are analogous to the 5 and 9 point stencil, but they operate in a 3D space. 
The user can choose the data type to be used in the grid (int, float or double).

## File parsing
The file parsing benchmark creates a file filled with sequences of random characters, as well as a fixed search phrase (here: "AdeptProject"). The benchmark then searches through the file and counts the occurences of the search phrase. 
The user can determine the size of the file by passing the number of lines to be created (using size).
 
## Conjugate Gradient solver
 This benchmark implements a simple CG solver, with a random matrix A of (user defined) size s. The CG computation includes BLAS computations (AXPY, AYPX and dot product) which are part of the slover loop. Only the solver loop is measured, the setup time is discarded.