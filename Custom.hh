#ifndef CUSTOM_HH
#define CUSTOM_HH


#include <cmath>


// Define here your activation function.
template <typename T>
T activationFunction(T x) {
  // This is the ReLU
  if (x >= 0)
    return x;
  return 0;
  // This is without activation function.
  return x;
  // This is the sigmoid.
  return 1/(1 + std::exp(-x));
  // This is a modified ReLU
  if (x >= 0)
    return x;
  return x/100;
}


// Define here the derivative of the activation function.
template <typename T>
T activationDerivative(T x) {
  // This is the ReLU derivative.
  if (x >= 0)
    return 1;
  return 0;
  // This is without activation function.
  return 1;
  // This is the sigmoid derivative.
  T sigmoid = activationFunction(x);
  return sigmoid * (1 - sigmoid);
  // This is the modified ReLU derivative.
  if (x >= 0)
    return 1;
  return 1/100;
}


// Define here your loss function (for comparing two real numbers).
template <typename T>
T lossFunction(T y, T yp) {
  // This is for euclidian distance.
  return (yp - y) * (yp - y);
  // This is a cross-entropy loss function modified.
  if (y == 0)
    return -std::log(1 - yp);
  else
    return -std::log(yp);
  // This is the cross-entropy loss.
  return -y*std::log(yp);
}


// Define here the derivative respect yp of the loss funcion.
template <typename T>
T lossDerivative(T y, T yp) {
  // This is for the euclidian distance.
  return 2*(yp - y);
  // This is a cross-entropy erorr function modified.
  if (y == 0)
    return -1/(1 - yp);
  else
    return -1/yp;
  // This is for the cross-entropy loss.
  return -y/yp;
}


#endif
