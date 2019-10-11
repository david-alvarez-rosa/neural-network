#include "Custom.hh"


// Define here your activation function.
float activation(float x) {
  // This is without activation function.
  return x;
  // This is the sigmoid.
  return 1/(1 + exp(-x));
  // This is the ReLU
  if (x >= 0)
    return x;
  return 0;
  // This is a modified ReLU
  if (x >= 0)
    return x;
  return x/100;
}


// Define here the derivative of the activation function.
float activationDerivative(float x) {
  // This is without activation function.
  return 1;
  // This is the sigmoid derivative.
  float sigmoid = activation(x);
  return sigmoid * (1 - sigmoid);
  // This is the ReLU derivative.
  if (x >= 0)
    return 1;
  return 0;
  // This is the modified ReLU derivative.
  if (x >= 0)
    return 1;
  return 1/100;
}


// Define here your error function (for comparing two real numbers).
float errorFunction(float y, float yp) {
  // // This is the cross-entropy error.
  // return -y * log(yp);
  // This is for euclidian distance.
  return (yp - y) * (yp - y);
}


// Define here the derivative respect yp of the error funcion.
float errorDerivative(float y, float yp) {
  // // This is for the cross-entropy error.
  // return -y/yp;
  // This is for the euclidian distance.
  return 2*(yp - y);
}


// Define here your function that normalizes output into a probability
// distribution.
VF convertIntoProbDist(VF v) {
  // This is the softmax.
  float denominator = 0;
  for (int i = 0; i < int(v.size()); ++i) {
    v[i] = exp(v[i]);
    denominator += v[i];
  }
  for (int i = 0; i < int(v.size()); ++i)
    v[i] /= denominator;

  return v;

  // This is without this function.
  return v;
}


// Define here the derivative of the previous function p elemenent with respect
// out[q].
float convertIntoProbDistDerivative(int p, int q, const VF& out) {
  // This is the softmax derivative.
  if (p != q)
    return - out[p] * out[q];
  return out[p] * (1 - out[p]);
}


// // Define here the derivative of the previous functions with respect v[i].
// // TODO: this is not correct, only applies to softmax! That's becacause the
// // first argument has the softmax function already applied.
// VF convertIntoProbDistDerivative(const VF& out, int k) {
//   // This is the softmax derivative.
//   VF derivative(out.size());
//   for (int i = 0; i < int(derivative.size()); ++i)
//     if (i != k) derivative[i] = - out[k] * out[i];
//   derivative[k] = out[k] * (1 - out[k]);
//   return derivative;
// }
