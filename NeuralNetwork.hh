#ifndef NEURALNETWORK_HH
#define NEURALNETWORK_HH

#include <cstdlib>
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

  NeuralNetwork(std::vector<int> neuronsPerLayer);

  void train(const std::vector<Data>& dataset, int epochs, int batchSize = 100,
             float alpha = 0.1);

  void test(const std::vector<Data>& dataset);

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
  void trainEpoch(const std::vector<Data>& dataset, int batchSize, float alpha);

  // Train iteration.
  void trainIteration(const std::vector<Data>& dataset, int batchSize,
                                    float alpha, const std::vector<int>& order,
                                    int iterNumber);

  // Forward propagation to compute values of all neurons.
  void feedForward(VF X);

  // Activation neurons in layer l.
  void activateNeurons(int l);

  // Acumulate the gradient of error respect single dataset in dataset.
  void dataGradient(const Data& data, int batchSize);

  // Initialize gradient with 0's.
  void initializeGradient();

  // Initialize with -1's tensor with partial derivatives.
  void initializePartialsNeuronsWeights();

  // Initialize with -1's tensor with partial derivatives.
  void initializePartialsNeuronsBiases();

  // Compute individual gradient respect a weight.
  float partialDataErrorWeight(int t, int i, int j, const Data& data);

  // Compute individual gradient respect a bias.
  float partialDataErrorBias(int t, int i, const Data& datas);

  // Compute partial derivatives of x[l][k] respect weight[t][i][j].
  float partialNeuronWeight(int l, int k, int t, int i, int j);

  // Compute partial derivatives of x[l][k] respect biast[t][i].
  float partialNeuronBias(int l, int k, int t, int i);

  // Compute partial derivatives of output respect last layer neurons.
  void partialOutputNeurons();

  // Update weights and biases.
  void updateWeightsAndBiases(float alpha);

  // Compute error.
  float errorData(const Data& data);

  // Auxiliar function for saving dataset.
  void saveData(float error, float accuracy);

  // Computing partial numerically to check.
  float partialDataErrorWeightNumerical(int t, int i, int j, const Data& data);

  // TODO: delete this.
  float relativeError(const float& x, const float& y);
};


#endif
