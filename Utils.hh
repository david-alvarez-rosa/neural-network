#ifndef UTILS_HH
#define UTILS_HH

#include <iostream>
#include <cstdlib>
#include "Defs.hh"


// Swaps two elements.
void swap(double &a, double &b);

// Print a vector.
void print(VF v);

// Print a vector.
void print(std::vector<int> v);

// Prints tensor.
void print(VVVF T);

// Print a matrix.
void print(VVF A);

// Fill randomnly a vector (entries between 0 and 1).
void fillRandomly(VF &v);

// Fill randomnly a matrix (entries between 0 and 1).
void fillRandomly(VVF &A);

// Compute relative error.
double relativeError(double x, double y);


#endif
