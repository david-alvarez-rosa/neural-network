#ifndef NEURALNETWORK_HH
#define NEURALNETWORK_HH

#include <iostream>
#include <fstream>
#include <algorithm>
#include "Defs.hh"
#include "Utils.hh"
#include "Math.hh"
#include "Custom.hh"
#include "Data.hh"


class NeuralNetwork {
public:
  VVF neurons;
  VVVF weights; VVF biases;
  VF out; // Output of the Neural Network.
  std::vector<Data> trainDataset, testDataset;
  double learningRate;

  NeuralNetwork(std::vector<int> neuronsPerLayer);

  void train(const std::vector<Data>& trainDataset, int epochs,
             int batchSize = 100, double learningRate = 0.1);

  void test(const std::vector<Data>& testDataset);

private:
  struct Gradient {
    VVVF weights;
    VVF biases;
  };

  Gradient gradient;

  // Partial derivative of x[l][k] respect weight[t][i][j].
  VVVVVF partialsNeuronsWeights;

  // Partial derivative of x[l][k] respect bias[t][i].
  VVVVF partialsNeuronsBiases;

  // Partial derivative of yp[k] respect x[L][p]. Is partialsOutputneurons[p][k].
  VVF partialsOutputNeurons;

  // Before aplying activation function.
  VVF neuronsNotActivated;

  // Train epoch.
  void trainEpoch(int batchSize);

  // Train iteration.
  void trainIteration(int batchSize, const std::vector<int>& order, int iterNumber);

  // Forward propagation to compute values of all neurons.
  void feedForward(VF X);

  // Activation neurons in layer l.
  void activateNeurons(int l);

  // Acumulate the gradient of error respect single dataset in dataset.
  void dataGradient(int d, int batchSize);

  // Initialize gradient with 0's.
  void initializeGradient();

  // Initialize with -1's tensor with partial derivatives.
  void initializePartialsNeuronsWeights();

  // Initialize with -1's tensor with partial derivatives.
  void initializePartialsNeuronsBiases();

  // Compute individual gradient respect a weight.
  double partialDataErrorWeight(int d, int t, int i, int j);

  // Compute individual gradient respect a bias.
  double partialDataErrorBias(int d, int t, int i);

  // Compute partial derivatives of x[l][k] respect weight[t][i][j].
  double partialNeuronWeight(int l, int k, int t, int i, int j);

  // Compute partial derivatives of x[l][k] respect biast[t][i].
  double partialNeuronBias(int l, int k, int t, int i);

  // Compute partial derivatives of output respect last layer neurons.
  void partialOutputNeurons();

  // Update weights and biases.
  void updateWeightsAndBiases();

  // Compute error.
  double errorData(const Data& data);

  // Save data to files.
  void saveData();
};


#endif
