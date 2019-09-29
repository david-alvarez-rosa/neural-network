#ifndef NEURALNETWORK_HH
#define NEURALNETWORK_HH

#include "defs.hh"
#include "utils.hh"
#include "math.hh"
#include "custom.hh"


class NeuralNetwork {
public:
  VVF neurons;
  VVVF weights;
  VVF biases;

  // Constructor (given number of neurons per layer as a vector).
  NeuralNetwork(VI neuronsPerLayer);

  // Update the neural network.
  void update(VF X, VF Y);

private:
  struct gradient {
    VVVF weights;
    VVF biases;
  };

  // Forward propagation to compute values of all neurons.
  void forwardPropagation();

  // Compute gradient.
  void backPropagation();

  // Back propagate to compute a derivative respect a weight.
  void backPropagationWeight(int l, int i, int j);

  // Back propagate to compute a derivative respect a bias.
  void backPropagationBias(int l, int i);

  // Activation neurons in layer l.
  void activateNeurons(int l);
};


#endif
