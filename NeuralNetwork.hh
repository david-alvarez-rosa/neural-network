#ifndef NEURALNETWORK_HH
#define NEURALNETWORK_HH

#include "Defs.hh"
#include "Utils.hh"
#include "Math.hh"
#include "Custom.hh"
#include "Data.hh"
#include "Image.hh"
#include "Auxiliares.hh"


class NeuralNetwork {
public:
  VVF neurons;
  VVVF weights; VVF biases;

  // Constructor (given number of neurons per layer as a vector).
  NeuralNetwork(VI neuronsPerLayer);

  // Train neural network.
  void train(vector<Data>& data);

  // Test neural network.
  void test(vector<Data>& data);

private:
  struct Gradient {
    VVVF weights;
    VVF biases;
  };

  Gradient gradient;

  // Partial derivative of x[l][k] respect weight[t][i][j].
  VVVVVF partialsNeuronsWeights;

  // Partial derivative of yp[k] respect x[L][p]. Is derivativesAux[p][k].
  VVF partialsOutputNeurons;

  // Before aplying activation function.
  VVF neuronsNotActivated;

  // Train step.
  void trainStep(vector<Data>& dataset);

  // Forward propagation to compute values of all neurons.
  void feedForward(VF X);

  // Activation neurons in layer l.
  void activateNeurons(int l);

  // Compute gradient.
  void computeGradient(vector<Data>& dataset);

  // Initialize with -1's tensor with partial derivatives and gradient with 0's.
  void initializeGradientVariables();

  // Compute individual gradient.
  float individualGradient(int t, int i, int j, Data& data);

  // Compute partial derivatives of x[l][k] respect weight[t][i][j].
  float partialNeuronWeight(int l, int k, int t, int i, int j);

  // TODO: Compute somo derivatives.
  void partialOutputNeurons();

  // Update weights and biases.
  void updateWeightsAndBiases();
};


#endif
