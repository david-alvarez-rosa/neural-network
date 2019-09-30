#include "Auxiliares.hh"


float computeError(VVF& neurons, VF& Y) {
  VF YP = neurons[neurons.size() - 1];
  float error_ = 0;
  for (int i = 0; i < int(YP.size()); ++i)
    error_ += error(Y[i], YP[i]);

  return error_;
}
