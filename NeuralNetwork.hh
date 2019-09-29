#ifndef NEURALNETWORK_HH
#define NEURALNETWORK_HH

#include "defs.hh"
#include "utils.hh"
#include "math.hh"
#include "custom.hh"
#include "Image.hh"


class NeuralNetwork {
public:
  VVF neurons;
  VVVF weights;
  VVF biases;

  // Constructor (given number of neurons per layer as a vector).
  NeuralNetwork(VI neuronsPerLayer);

  // Train neural network.
  void train(vector<Image>& images);

  // Test neural network.
  void test(vector<Image>& images);

private:
  struct Gradient {
    VVVF weights;
    VVF biases;
  };

  Gradient gradient;

  // Forward propagation to compute values of all neurons.
  void forwardPropagation(VF X);

  // Activation neurons in layer l.
  void activateNeurons(int l);

  // Compute gradient.
  void backPropagation();

  // Back propagate to compute a derivative respect a weight.
  void backPropagationWeight(int l, int i, int j);

  // Back propagate to compute a derivative respect a bias.
  void backPropagationBias(int l, int i);

  // Update weights and biases.
  void updateWeightsAndBiases();
};


#endif
