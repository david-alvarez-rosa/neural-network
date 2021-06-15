#ifndef NEURON_HH
#define NEURON_HH


#include "Custom.hh"


template <typename T>
class Neuron {
public:
  T deactivated, activated;

  Neuron(T (*activationFunction)(T), T (*activationDerivative)(T));

  void activate();

  T derivative(); // Compute activation function derivative in neuron.

private:
  T (*activationFunction)(T);
  T (*activationDerivative)(T);
};


template <typename T>
Neuron<T>::Neuron(T (*activationFunction)(T), T (*activationDerivative)(T)) {
  this->activationFunction = activationFunction;
  this->activationDerivative = activationDerivative;
}


template <typename T>
void Neuron<T>::activate() {
  activated = activationFunction(deactivated);
}


template <typename T>
T Neuron<T>::derivative() {
  return activationDerivative(deactivated);
}


#endif
