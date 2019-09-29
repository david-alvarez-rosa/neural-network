#ifndef MATH_HH
#define MATH_HH

#include "defs.hh"


// Given two vectors v and w, computes v + w.
VF sum(VF v, VF w);

// Given two vectors v and w, computes v - w.
VF vectorDifference(VF v, VF w);

// Given a vector returns position of the (first) maxim.
int vectorMaxPos(VI v);

// Multiplies matrix and vector.
VF multiply(VVF A, VF v);

// Computes a^n, with n a non-negative integer.
double pow(double a, int n);

// Computes exp(x) for small x (between 0 and 1).
double expInterval(double x);

// Computes exp(x).
double exp(double x);

// Computes the square of the Euclidian norm of a vector.
double euclidianNormSquared(VF v);

// Computes the Euclidian Distance between two vectors.
float euclidianDistanceSquared(VF v, VF w);

// Softmax function (inplace).
void applySoftmax(VF &v);

// Sigmoid function.
double sigmoid(double x);

// Sigmoid function for a vector (inplace).
VF applySigmoid(VF &v);


#endif
