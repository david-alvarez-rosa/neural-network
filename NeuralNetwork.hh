#ifndef NEURALNETWORK_HH
#define NEURALNETWORK_HH

#include "defs.hh"

class NeuralNetwork {
public:
  VVF neurons;
  VVVF weights; VVVF weightsGradient;
  VVF biases;   VVF  biasesGradient;

  NeuralNetwork(VI neuronsPerLayer) {
    int numLayers = neuronsPerLayer.size();
    neurons = VVF(numLayers - 1);
    weights, weightsGradient = VVVF(numLayers - 1);
    biases,  biasesGradient  =  VVF(numLayers - 1);
    for (int l = 0; l < numLayers - 1; ++l) {
      neurons[l] = VF(neuronsPerLayer[l]);
      weights[l] = VVF(neuronsPerLayer[l + 1], VF(neuronsPerLayer[l]));
      fillRandomly(weights[l]);
      biases[l] = VF(neuronsPerLayer[l + 1]);
      fillRandomly(biases[l]);
    }
  }

  void update(VF X, VF Y) {
    neurons[0] = X;
    forwardPropagation();
    backPropagation();
  }

private:
  // Forward propagation to compute values of all neurons.
  void forwardPropagation() {
    for (int l = 0; l < (int)neurons.size() - 1; ++l) {
      neurons[l + 1] = sum(multiply(weights[l], neurons[l]), biases[l]);
      activateNeurons(l + 1);
    }
  }

  // Compute gradient.
  void backPropagation() {
    for (int l = 0; l < weights.size(); ++l)
      for (int i = 0; i < weights.size[l]; ++i)
        for (int j = 0; j < weights.size[j]; ++j)
          backPropagationWeight(l, i, j);

    for (int l = 0; l < biases.size(); ++l).
      for (int i = 0; i < biases.size(); ++i)
        backPropagationBias(l, i);
  }

  // Back propagate to compute a derivative respect a weight.
  void backPropagationWeight(int l, int i, int j) {
    weightsGradient[l][i][j] = fp(multiply(neurons[l][j]))
      }

  // Back propagate to compute a derivative respect a bias.
  void backPropagationBias(int l, int i) {
    for (int k = 0; k < 2; ++k)

      }

  // Activation neurons in layer l.
  void activateNeurons(int l) {
    applySigmoid(neurons[l]);
  }
};


#endif
