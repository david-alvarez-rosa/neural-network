#ifndef CUSTOM_HH
#define CUSTOM_HH

#include "Math.hh"


// Activation function used for activate neurons.
double activation(double x);

// Derivative of the activation function.
double activationDerivative(double x);

// Error function to minimize in gradient descent.
double errorFunction(double y, double yp);

// Derivative of the error function whith respect yp.
double errorDerivative(double y, double yp);

// Functions thath normalizes output into a probability distribution.
VF convertIntoProbDist(VF v);

// Derivative of the previous function respect v[k].
double convertIntoProbDistDerivative(int p, int q, const VF& out);


#endif
