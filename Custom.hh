#ifndef CUSTOM_HH
#define CUSTOM_HH

#include "Math.hh"


// Activation function used for activate neurons.
float activation(float x);

// Derivative of the activation function.
float activationDerivative(float x);

// Error function to minimize in gradient descent.
float errorFunction(float y, float yp);

// Derivative of the error function whith respect yp.
float errorDerivative(float y, float yp);

// Functions thath normalizes output into a probability distribution.
VF convertIntoProbDist(VF v);

// Derivative of the previous function respect v[k].
VF convertIntoProbDistDerivative(VF w, int k);


#endif
