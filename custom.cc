#include "custom.hh"


// Define here your activation function.
float activationFunction(float x) {
  // This is the ReLU
  if (x >= 0)
    return x;
  return 0;
  // This is the sigmoid.
  return 1/(1 + exp(-x));
}


// Define here your error function.
float errorFunction(VF Y, VF YP) {
  // This is euclidian distance.
  float error = 0;
  for (int i = 0; i < int(Y.size()); ++i)
    error += (YP[i] - Y[i]) * (YP[i] - Y[i]);
  return error;
}
