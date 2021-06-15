#ifndef UTILS_HH
#define UTILS_HH


#include <cmath>
#include "Layer.hh"


/**
 * Activation functions (and derivatives).
 */

// Linear. No activation function.
template <typename T>
T linear(T x) {
  return x;
}

template <typename T>
T linearDerivative(T x) {
    return 1;
}


// ReLU.
template <typename T>
T ReLU(T x) {
  if (x >= 0)
    return x;
  return 0;
}

template <typename T>
T ReLUDerivative(T x) {
  if (x >= 0)
    return 1;
  return 0;
}


// Sigmoid.
template <typename T>
T sigmoid(T x) {
  return 1/(1 + std::exp(-x));
}

// template <typename T>
// void sigmoid(Layer<T> *layer) {
//   activateEqually(layer, sigmoid);
// }

template <typename T>
T sigmoidDerivative(T x) {
  T sigmoid = activationFunction(x);
  return sigmoid * (1 - sigmoid);
}


// Leaky ReLU.
template <typename T>
T leakyReLU(T x) {
  if (x >= 0)
    return x;
  return x/100;
}

// template <typename T>
// void leakyReLU(Layer<T> *layer) {
//   activateEqually(layer, leakyReLU);
// }

template <typename T>
T leakyReLUDerivative(T x) {
  if (x >= 0)
    return 1;
  return 1/100;
}


/**
 * Loss functions (and derivatives).
 */

template <typename T>
T euclidianLoss(T y, T yp) {
  return (yp - y) * (yp - y);
}


template <typename T>
T euclidianLossDerivative(T y, T yp) {
  return 2*(yp - y);
}


template <typename T>
T crossEntropyLoss(T y, T yp) {
  return 0;
}


template <typename T>
T crossEntropyLossDerivative(T y, T yp) {
  return 0;
}


/**
 * Other auxiliary functions.
 */

// Relative error between two numbers.
template <typename T>
T relativeError(T x, T y) {
  return std::abs((x - y)/(std::max(T(1e-3), std::max(x, y))));
}


#endif
