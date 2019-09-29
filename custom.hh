#ifndef CUSTOM_HH
#define CUSTOM_HH

#include "math.hh"


// Activation function used for activate neurons.
float activationFunction(float x);

// Loss function to minimize in gradient descent.
float lossFunction(VVF Y, VVF YP);


#endif
