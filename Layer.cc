#include "Layer.hh"

Layer::Layer(int size, int sizeNextLayer) {
  this->size = size;

  neurons = std::vector<Neuron>(size);
  weights = std::vector< std::vector<Param> >(sizeNextLayer,
                                              std::vector<Param>(size));
  biases = std::vector<Param>(sizeNextLayer);

  // Initialize parameters with normal distribution.
  fillRandomly(weights);
  fillRandomly(biases);
}


void Layer::forward() {
  for (int i = 0; i < nextLayer->size; ++i) {
    nextLayer->neurons[i].deactivated = 0;
    for (int j = 0; j < size; ++j)
      nextLayer->neurons[i].deactivated += weights[i][j].value*neurons[j].activated;
    nextLayer->neurons[i].deactivated += biases[i].value;

    nextLayer->neurons[i].activate();
  }
}


void Layer::backward() {
  deltas = std::vector<double>(size, 0);

  for (int i = 0; i < size; ++i)
    for (int j = 0; j < nextLayer->size; ++j)
      deltas[i] += weights[j][i].value*nextLayer->deltas[j];

  // print(deltas);
}


void Layer::fillRandomly(std::vector<Param>& v) {
  std::default_random_engine generator;
  std::normal_distribution<double> normalDistribution(mean, std);
  for (int i = 0; i < int(v.size()); ++i)
    v[i].value = normalDistribution(generator);
}


void Layer::fillRandomly(std::vector< std::vector<Param> >& v) {
  std::default_random_engine generator;
  std::normal_distribution<double> normalDistribution(mean, std);

  for (int i = 0; i < int(v.size()); ++i)
    for (int j = 0; j < int(v[i].size()); ++j)
      v[i][j].value = normalDistribution(generator);
}



void Layer::computeGradients() {
  // Weights.
  for (int i = 0; i < nextLayer->size; ++i)
    for (int j = 0; j < size; ++j) {
      // std::cout << nextLayer->deltas[i] << std::endl;
      // std::cout << neurons[j].deactivated << std::endl << std::endl;

      weights[i][j].gradient += neurons[j].deactivated*nextLayer->deltas[i];
    }


  // for (int i = 0; i < nextLayer->size; ++i)
  //   for (int j = 0; j < size; ++j)
  //     std::cout << weights[i][j].gradient << "\t";
  // std::cout << std::endl;



  // Biases.
  for (int i = 0; i < nextLayer->size; ++i)
    biases[i].gradient += nextLayer->deltas[i];
}


void Layer::zeroGradients() {
  // Weights.
  for (int i = 0; i < nextLayer->size; ++i)
    for (int j = 0; j < size; ++j)
      weights[i][j].gradient = 0;

  // Biases.
  for (int i = 0; i < nextLayer->size; ++i)
    biases[i].gradient = 0;
}


void Layer::updateParameters(const double& learningRate) {
  // Weights.
  for (int i = 0; i < nextLayer->size; ++i)
    for (int j = 0; j < size; ++j)
      weights[i][j].value -= learningRate*weights[i][j].gradient;

  // Biases.
  for (int i = 0; i < nextLayer->size; ++i)
      biases[i].value -= learningRate*biases[i].gradient;
}
