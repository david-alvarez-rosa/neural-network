#include "Custom.hh"


// Define here your activation function.
float activation(float x) {
  // This is a modified ReLU
  if (x >= 0)
    return x;
  return x/100;
  // This is the ReLU
  if (x >= 0)
    return x;
  return 0;
  // This is the sigmoid.
  return 1/(1 + exp(-x));
}


// Define here the derivative of the activation function.
float activationDerivative(float x) {
  if (x >= 0)
    return 1;
  return 0;
}


// Define here your error function (for comparing two real numbers).
float error(float y, float yp) {
  // // This is the cross-entropy error.
  // return -y * log(yp);
  // This is for euclidian distance.
  return (yp - y)*(yp - y);
}


// Define here the derivative respect yp of the error funcion.
float errorDerivative(float y, float yp) {
  // // This is for the cross-entropy error.
  // return -y/yp;
  // This is for the euclidian distance.
  return 2*(yp - y);
}


// Define here your functions that normalizes output into a probability
// distribution.
void convertIntoProbDist(VF& v) {
  // This is the softmax.
  float denominator = 0;
  for (int i = 0; i < int(v.size()); ++i) {
    v[i] = exp(v[i]);
    denominator += v[i];
  }
  for (int i = 0; i < int(v.size()); ++i)
    v[i] /= denominator;
}


// Define here the derivative of the previous functions with respect v[i].
// TODO: this is not correct, only applies to softmax! That's becacause the
// first argument has the softmax function already applied.
VF convertIntoProbDistDerivative(VF w, int k) {
  // This is the softmax derivative.
  // normalizeIntoProbabilityDist(w);
  VF derivative(w.size());
  for (int i = 0; i < int(derivative.size()) and i != k; ++i)
    derivative[k] = - w[k] * w[i];
  derivative[k] = w[k] * (1 - w[k]);
  return derivative;
}