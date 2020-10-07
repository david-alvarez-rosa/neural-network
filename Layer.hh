#ifndef LAYER_HH
#define LAYER_HH


#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include "Neuron.hh"
#include "Data.hh"
#include "Custom.hh"


template <typename T> class Layer {
public:
  struct Param {
    T value;
    T gradient;
  };

  int size, sizeNextLayer;

  std::vector< Neuron<T> > neurons;
  std::vector< std::vector<Param> > weights;
  std::vector<Param> biases;
  std::vector<T> deltas;

  Layer* prevLayer; Layer* nextLayer;

  Layer(int size, int sizeNextLayer = 0);

  // Forward step. Compute values of next layer.
  void forward();

  // Backward step. Compute deltas of previous layer.
  void backward();

  // Set all gradients to zero.
  void zeroGradients();

  // Compute actual gradients of loos wrt parameters.
  void computeGradients();

  // Optimization step. Update value of parameters.
  void updateParameters(const T& learningRate);


private:
  T mean = 0, std = 0.01;

  void fillRandomly(std::vector<Param>& v);

  void fillRandomly(std::vector< std::vector<Param> >& v);
};


template <typename T>
Layer<T>::Layer(int size, int sizeNextLayer) {
  this->size = size;

  neurons = std::vector< Neuron<T> >(size);
  weights = std::vector< std::vector<Param> >(sizeNextLayer,
                                              std::vector<Param>(size));
  biases = std::vector<Param>(sizeNextLayer);

  // Initialize parameters with normal distribution.
  fillRandomly(weights);
  fillRandomly(biases);
}


template <typename T>
void Layer<T>::forward() {
  for (int i = 0; i < nextLayer->size; ++i) {
    nextLayer->neurons[i].deactivated = 0;
    for (int j = 0; j < size; ++j)
      nextLayer->neurons[i].deactivated += weights[i][j].value*neurons[j].activated;
    nextLayer->neurons[i].deactivated += biases[i].value;

    nextLayer->neurons[i].activate();
  }
}


template <typename T>
void Layer<T>::backward() {
  deltas = std::vector<T>(size, 0);

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < nextLayer->size; ++j)
      deltas[i] += weights[j][i].value*nextLayer->deltas[j]*
        activationDerivative<T>(nextLayer->neurons[j].deactivated);
  }
}


template <typename T>
void Layer<T>::zeroGradients() {
  // Weights.
  for (int i = 0; i < nextLayer->size; ++i)
    for (int j = 0; j < size; ++j)
      weights[i][j].gradient = 0;

  // Biases.
  for (int i = 0; i < nextLayer->size; ++i)
    biases[i].gradient = 0;
}


template <typename T>
void Layer<T>::computeGradients() {
  // Weights.
  for (int i = 0; i < nextLayer->size; ++i)
    for (int j = 0; j < size; ++j)
      weights[i][j].gradient += neurons[j].activated*nextLayer->deltas[i]*
        activationDerivative<T>(nextLayer->neurons[i].deactivated);

  // Biases.
  for (int i = 0; i < nextLayer->size; ++i) {
    biases[i].gradient += nextLayer->deltas[i]*
      activationDerivative<T>(nextLayer->neurons[i].deactivated);
  }
}


template <typename T>
void Layer<T>::updateParameters(const T& learningRate) {
  // Weights.
  for (int i = 0; i < nextLayer->size; ++i)
    for (int j = 0; j < size; ++j)
      weights[i][j].value -= learningRate*weights[i][j].gradient;

  // Biases.
  for (int i = 0; i < nextLayer->size; ++i)
    biases[i].value -= learningRate*biases[i].gradient;
}


template <typename T>
void Layer<T>::fillRandomly(std::vector<Param>& v) {
  std::default_random_engine generator;
  std::normal_distribution<T> normalDistribution(mean, std);
  for (int i = 0; i < int(v.size()); ++i)
    v[i].value = normalDistribution(generator);
}


template <typename T>
void Layer<T>::fillRandomly(std::vector< std::vector<Param> >& v) {
  std::default_random_engine generator;
  std::normal_distribution<T> normalDistribution(mean, std);

  for (int i = 0; i < int(v.size()); ++i)
    for (int j = 0; j < int(v[i].size()); ++j)
      v[i][j].value = normalDistribution(generator);
}



#endif
