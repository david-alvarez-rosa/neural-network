#include "Custom.hh"


// Define here your activation function.
float activation(float x) {
  // This is without activation function.
  return x;
  // This is the sigmoid.
  return 1/(1 + exponential(-x));
  // This is a modified ReLU
  if (x >= 0)
    return x;
  return x/100;
  // This is the ReLU
  if (x >= 0)
    return x;
  return 0;
}


// Define here the derivative of the activation function.
float activationDerivative(float x) {
  // This is without activation function.
  return 1;
  // This is the sigmoid derivative.
  float sigmoid = activation(x);
  return sigmoid * (1 - sigmoid);
  // This is the modified ReLU derivative.
  if (x >= 0)
    return 1;
  return 1/100;
  // This is the ReLU derivative.
  if (x >= 0)
    return 1;
  return 0;
}


// Define here your error function (for comparing two real numbers).
float errorFunction(float y, float yp) {
  // This is for euclidian distance.
  return (yp - y) * (yp - y);
  // This is the cross-entropy error.
  return -y * logarithm(yp);
  // This is a cross-entropy erorr function modified.
  if (y == 0)
    return -logarithm(1 - yp);
  else
    return -logarithm(yp);
}


// Define here the derivative respect yp of the error funcion.
float errorDerivative(float y, float yp) {
  // This is for the euclidian distance.
  return 2*(yp - y);
  // This is a cross-entropy erorr function modified.
  if (y == 0)
    return -1/(1 - yp);
  else
    return -1/yp;
  // This is for the cross-entropy error.
  return -y/yp;
}


// Define here your function that normalizes output into a probability
// distribution.
VF convertIntoProbDist(VF v) {
  // This is without this function.
  return v;

  // This is the softmax.
  float denominator = 0;
  for (int i = 0; i < int(v.size()); ++i) {
    v[i] = exponential(v[i]);
    denominator += v[i];
  }
  for (int i = 0; i < int(v.size()); ++i)
    v[i] /= denominator;

  return v;
}


// Define here the derivative of the previous function p elemenent with respect
// out[q].
float convertIntoProbDistDerivative(int p, int q, const VF& out) {
  // This is without this function.
  if (p == q)
    return 1;
  return 0;

  // This is the softmax derivative.
  if (p != q)
    return - out[p] * out[q];
  return out[p] * (1 - out[p]);
}
