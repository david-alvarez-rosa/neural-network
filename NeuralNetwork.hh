#ifndef NEURALNETWORK_HH
#define NEURALNETWORK_HH


#include <iostream>
#include <fstream>
#include <algorithm>
#include "Custom.hh"
#include "Data.hh"
#include "Layer.hh"


template <typename T = float> class NeuralNetwork {
public:
  std::vector< Layer<T> > layers;
  std::vector<T> out; // Output of the Neural Network.
  T learningRate = 0.002;
  int batchSize = 100;

  // Constructor given neuronsPerLayer vector.
  NeuralNetwork(std::vector<int> neuronsPerLayer);

  void train(std::vector< Data<T> >* trainDataset, int epochs);

  void test(std::vector< Data<T> >* testDataset);


private:
  int numLayers;
  std::vector< Data<T> >* trainDataset;
  std::vector< Data<T> >* testDataset;

  // Train epoch.
  void trainEpoch();

  // Train iteration.
  void trainIteration(const std::vector<int>& order, int iterNumber);

  // Forward propagation to compute values of all neurons.
  void feedForward(const std::vector<T>& x);

  // Backpropagation algorithm.
  void backPropagate(int d);

  // Set all gradients to zeros.
  void zeroGradients();

  // Update weights and biases.
  void updateParameters();

  // Compute loss.
  T lossData(const Data<T>& data);

  // Compute gradients given data.
  void dataGradientNumerical(int d);








  T relativeError(T x, T y);
};


template <typename T>
NeuralNetwork<T>::NeuralNetwork(std::vector<int> neuronsPerLayer) {
  numLayers = neuronsPerLayer.size();

  for (int l = 0; l < numLayers - 1; ++l)
    layers.push_back(Layer<T>(neuronsPerLayer[l], neuronsPerLayer[l + 1]));

  // Create output layer.
  layers.push_back(Layer<T>(neuronsPerLayer[numLayers - 1]));

  // Connect layers.
  layers[0].nextLayer = &layers[1];
  for (int l = 1; l < numLayers - 1; ++l) {
    layers[l].prevLayer = &layers[l - 1];
    layers[l].nextLayer = &layers[l + 1];
  }
  layers[numLayers - 1].prevLayer = &layers[numLayers - 2];
}


template <typename T>
void NeuralNetwork<T>::train(std::vector< Data<T> >* trainDataset, int epochs) {
  this->trainDataset = trainDataset;

  test(trainDataset);
  for (int epoch = 0; epoch < epochs; ++epoch) {
    trainEpoch();
    test(trainDataset);
  }
}


template <typename T>
void NeuralNetwork<T>::test(std::vector< Data<T> >* testDataset) {
  this->testDataset = testDataset;

  T loss = 0;
  int correct = 0;

  for (int d = 0; d < int(testDataset->size()); ++d) {
    feedForward((*testDataset)[d].in);

    loss += lossData((*testDataset)[d]);

    int number = std::distance(out.begin(),
                               std::max_element(out.begin(), out.end()));
    if (number == (*testDataset)[d].label)
      ++correct;
  }

  std::cout << "Loss: " << loss / float(testDataset->size()) << "\t\t"
            << "Accuraccy: " << 100 * correct / float(testDataset->size())
            << std::endl;
}


template <typename T>
void NeuralNetwork<T>::trainEpoch() {
  // Create random vector.
  std::vector<int> order(trainDataset->size());
  for (int i = 0; i < int(order.size()); ++i)
    order[i] = i;
  std::random_shuffle(order.begin(), order.end());

  for (int i = 0; i < int(trainDataset->size()) / T(batchSize); ++i)
    trainIteration(order, i);
}


template <typename T>
void NeuralNetwork<T>::trainIteration(const std::vector<int>& order,
                                      int iterNumber) {
  zeroGradients();

  // Check if there is enough data for batchSize.
  if (batchSize*(iterNumber + 1) >= int(order.size()))
    batchSize = order.size() - batchSize*iterNumber;

  for (int i = 0; i < batchSize; ++i) {
    int d = order[batchSize*iterNumber + i];
    feedForward((*trainDataset)[d].in);
    backPropagate(d);
    dataGradientNumerical(d);
  }

  updateParameters();
}


template <typename T>
void NeuralNetwork<T>::feedForward(const std::vector<T>& x) {
  // Pass input to first layer.
  Layer<T>& layer = layers[0];
  for (int i = 0; i < int(x.size()); ++i)
    layer.neurons[i].deactivated = layer.neurons[i].activated = x[i];

  // Forward.
  for (int l = 0; l < numLayers - 1; ++l)
    layers[l].forward();


  // Without softmax.
  Layer<T>& lastLayer = layers[numLayers - 1];
  out = std::vector<T>(lastLayer.size);
  for (int i = 0; i < lastLayer.size; ++i)
    out[i] = lastLayer.neurons[i].activated;

  return;


  // Softmax last layer.
  // Layer& lastLayer = layers[numLayers - 1];
  out = std::vector<T>(lastLayer.size);
  double aux = 0;
  for (int i = 0; i < lastLayer.size; ++i)
    aux += std::exp(lastLayer.neurons[i].deactivated);
  for (int i = 0; i < lastLayer.size; ++i)
    out[i] = std::exp(lastLayer.neurons[i].deactivated)/aux;
}


template <typename T>
void NeuralNetwork<T>::backPropagate(int d) {
  Layer<T>& lastLayer = layers[numLayers - 1];
  lastLayer.deltas = std::vector<T>(lastLayer.size, 0);
  for (int i = 0; i < lastLayer.size; ++i)
    lastLayer.deltas[i] = lossDerivative((*trainDataset)[d].out[i], out[i]);

  for (int l = numLayers - 2; l >= 0; --l) {
    layers[l].backward();
    layers[l].computeGradients();
  }
}


template <typename T>
void NeuralNetwork<T>::zeroGradients() {
  for (int l = 0; l < numLayers - 1; ++l)
    layers[l].zeroGradients();
}


template <typename T>
void NeuralNetwork<T>::updateParameters() {
  for (int l = 0; l < numLayers - 1; ++l)
    layers[l].updateParameters(learningRate);
}


template <typename T>
T NeuralNetwork<T>::lossData(const Data<T>& data) {
  T loss = 0;
  for (int i = 0; i < int(data.out.size()); ++i)
    loss += lossFunction(data.out[i], out[i]);
  return loss;
}


template <typename T>
void NeuralNetwork<T>::dataGradientNumerical(int d) {
  int numCorrectW = 0;
  int numIncorrectW = 0;
  int numCorrectB = 0;
  int numIncorrectB = 0;

  const T EPS = 1e-9;

  for (int l = 0; l < numLayers - 1; ++l) {
    Layer<T>& layer = layers[l];
    Layer<T>& nextLayer = layers[l + 1];
    for (int i = 0; i < nextLayer.size; ++i) {
      for (int j = 0; j < layer.size; ++j) {

        layer.weights[i][j].value -= EPS;
        feedForward(trainDataset->at(d).in);
        T lossMinus = lossData(trainDataset->at(d));

        layer.weights[i][j].value += 2*EPS;
        feedForward(trainDataset->at(d).in);
        T lossPlus = lossData(trainDataset->at(d));

        layer.weights[i][j].value -= EPS;

        T gradientAux = (lossPlus - lossMinus)/(2*EPS);
        T gradient = layer.weights[i][j].gradient;

        // layer.weights[i][j].gradient += gradientAux;

        if (relativeError(gradientAux, gradient) > 1e-2) {
          ++numIncorrectW;
          std::cout << l << " " << i << " " << j << std::endl;
          std::cout << gradientAux << std::endl;
          std::cout << gradient << std::endl << std::endl;
        }
        else
          ++numCorrectW;

        // if (gradient == 0)
        //   std::cout << "All is zero!" << std::endl;
        // else
        //   std::cout << "Not zero!" << std::endl;


        // double relLoss = relativeError(layer.weights[i][j].gradient, gradientAux);
        // double absLoss = std::abs(layer.weights[i][j].gradient - gradientAux);
      }

      layer.biases[i].value -= EPS;
      feedForward(trainDataset->at(d).in);
      T lossMinus = lossData(trainDataset->at(d));

      layer.biases[i].value += 2*EPS;
      feedForward(trainDataset->at(d).in);
      T lossPlus = lossData(trainDataset->at(d));

      layer.biases[i].value -= EPS;

      T gradientAux = (lossPlus - lossMinus)/(2*EPS);
      T gradient = layer.biases[i].gradient;
      if (relativeError(gradientAux, gradient) > 1e-3)
          ++numIncorrectB;
        else
          ++numCorrectB;

      layer.biases[i].gradient += gradientAux;
    }
  }

  std::cout << "Number correct W: " << numCorrectW << std::endl;
  std::cout << "Number incorrec Wt: " << numIncorrectW << std::endl;
  std::cout << "Number correct B: " << numCorrectB << std::endl;
  std::cout << "Number incorrect B: " << numIncorrectB << std::endl;
}


template <typename T>
T NeuralNetwork<T>::relativeError(T x, T y) {
  return std::abs((x - y)/(std::max(T(1e-3), std::max(x, y))));
}


#endif
