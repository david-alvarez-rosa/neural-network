#include "custom.hh"


// Define here your activation function.
float activationFunction(float x) {
  // This is the sigmoid.
  return 1/(1 + exp(-x));
}


// Define here your loss function.
float lossFunction(VF Y, VF YP) {
  float loss = 0;
  for (int i = 0; i < int(Y.size()); ++i)
    loss += (YP[i] - Y[i]) * (YP[i] - Y[i]);
  return loss;
}
