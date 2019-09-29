#ifndef MATH_HH
#define MATH_HH

#include "defs.hh"


// Given two vectors v and w, computes v + w.
VF sum(VF v, VF w);

// Given two vectors v and w, computes v - w.
VF vectorDifference(VF v, VF w);

// Given a vector returns position of the (first) maxim.
int vectorMaxPos(VF v);

// Multiplies matrix and vector.
VF multiply(VVF A, VF v);

// Computes a^n, with n a non-negative integer (quick-exponentiation).
float pow(float a, int n);

// Computes exp(x) for small x (between 0 and 1).
float expInterval(float x);

// Computes exp(x).
float exp(float x);

// Softmax function (inplace).
void applySoftmax(VF &v);


#endif
