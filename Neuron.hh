#ifndef NEURON_HH
#define NEURON_HH


#include "Custom.hh"


template <typename T> class Neuron {
public:
  T deactivated;
  T activated;

  void activate();
};


template <typename T>
void Neuron<T>::activate() {
  activated = activationFunction<T>(deactivated);
}


#endif
