#include "Layer.hh"


void Neuron::activate() {
  activated = activationFunction(deactivated);
}
