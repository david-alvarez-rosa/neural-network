#ifndef NEURALNETWORK_HH
#define NEURALNETWORK_HH

#include <iostream>
#include <fstream>
#include <algorithm>
#include "Utils.hh"
#include "Math.hh"
#include "Custom.hh"
#include "Data.hh"
#include "Layer.hh"


class NeuralNetwork {
public:
  std::vector<Layer> layers;

  VF out; // Output of the Neural Network.

  NeuralNetwork(std::vector<int> neuronsPerLayer);

  void train(const std::vector<Data>& trainDataset, int epochs,
             int batchSize = 100, double learningRate = 0.002);

  void test(const std::vector<Data>& testDataset);

private:
  int numLayers;
  std::vector<Data> trainDataset, testDataset;
  double learningRate;

  // Forward propagation to compute values of all neurons.
  void feedForward(VF x);

  // Train epoch.
  void trainEpoch(int batchSize);

  // Train iteration.
  void trainIteration(int batchSize, const std::vector<int>& order, int iterNumber);

  // Compute gradients given data.
  void dataGradientNumerical(int d);

  // Backpropagation algorithm.
  void backPropagate(int d);

  // Update weights and biases.
  void updateParameters();

  // Compute error.
  double errorData(const Data& data);

  // Set all gradients to zeros.
  void zeroGradients();

  // // Save data to files.
  // void saveData();
};


#endif
