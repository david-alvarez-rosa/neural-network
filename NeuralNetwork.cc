#include "NeuralNetwork.hh"


NeuralNetwork::NeuralNetwork(std::vector<int> neuronsPerLayer) {
  int numLayers = neuronsPerLayer.size();
  neurons = VVF(numLayers); neuronsNotActivated = VVF(numLayers);
  weights = VVVF(numLayers - 1); biases =  VVF(numLayers - 1);
  for (int l = 0; l < numLayers; ++l) {
    neurons[l] = VF(neuronsPerLayer[l]);
    neuronsNotActivated[l] = VF(neuronsPerLayer[l]);
  }

  for (int l = 0; l < numLayers - 1; ++l) {
    weights[l] = VVF(neuronsPerLayer[l + 1], VF(neuronsPerLayer[l]));
    fillRandomly(weights[l]);
    biases[l] = VF(neuronsPerLayer[l + 1]);
    fillRandomly(biases[l]);
  }
}


void NeuralNetwork::train(const std::vector<Data>& dataset, int steps) {
  for (int step = 0; step < steps; ++step)
    trainStep(dataset);
}


// TODO: delete this line.
#include <fstream>


void NeuralNetwork::trainStep(const std::vector<Data>& dataset) {
  // Initialize gradient with 0's.
  initializeGradient();

  // For coputing the error over all the dataset.
  float error = 0;

  for (int d = 0; d < int(dataset.size()); ++d) {
    initializePartialsNeuronsWeights();
    initializePartialsNeuronsBiases();

    feedForward(dataset[d].in);
    dataGradient(dataset[d]);

    error += errorData(dataset[d]);
  }

  // TODO: this saves the weigths in a file, get rid of this.
  // TODO: get rid of this (start).
  std::ofstream file("weights.dat");
  // Normal weights.
  for (int l = 0; l < int(weights.size()); ++l)
    for (int i = 0; i < int(weights[l].size()); ++i)
      for (int j = 0; j < int(weights[l][i].size()); ++j)
        file << weights[l][i][j] << std::endl;
  // Biases.
  for (int l = 0; l < int(biases.size()); ++l)
    for (int i = 0; i < int(biases[l].size()); ++i)
      file << biases[l][i] << std::endl;
  // TODO: get rid of this (end).

  updateWeightsAndBiases();
  std::cout << error << std::endl;
}


void NeuralNetwork::test(const std::vector<Data>& dataset) {
  std::cout << std::endl << "Testing starts." << std::endl;
  int correct = 0;
  for (int d = 0; d < int(dataset.size()); ++d) {
    feedForward(dataset[d].in);
    int number = vectorMaxPos(out);
    std::cout << number << std::endl;
  }

  float percentage = 100 * correct / float(dataset.size());
  std::cout << "Percentage: " << percentage << "%." << std::endl;
}


void NeuralNetwork::feedForward(VF X) {
  neurons[0] = X;
  for (int l = 0; l < int(neurons.size()) - 1; ++l) {
    neurons[l + 1] = sum(multiply(weights[l], neurons[l]), biases[l]);
    neuronsNotActivated[l + 1] = neurons[l + 1]; // Save this for computing gradients.
    activateNeurons(l + 1);
  }

  out = convertIntoProbDist(neurons[neurons.size() - 1]);
}


void NeuralNetwork::activateNeurons(int l) {
  for (int i = 0; i < int(neurons[l].size()); ++i)
    neurons[l][i] = activation(neurons[l][i]);
}


void NeuralNetwork::dataGradient(const Data& data) {
  // Compute partial derivatives of output respect neurons.
  partialOutputNeurons();

  // Gradient respect weights.
  for (int t = 0; t < int(weights.size()); ++t)
    for (int i = 0; i < int(weights[t].size()); ++i)
      for (int j = 0; j < int(weights[t][i].size()); ++j)
        gradient.weights[t][i][j] += partialDataErrorWeight(t, i, j, data);

  // Gradient respect biases.
  for (int t = 0; t < int(biases.size()); ++t)
    for (int i = 0; i < int(biases[t].size()); ++i)
      gradient.biases[t][i] += partialDataErrorBias(t, i, data);
}


float NeuralNetwork::partialDataErrorWeight(int t, int i, int j,
                                            const Data& data) {
  float derivative = 0;
  for (int q = 0; q < int(out.size()); ++q) {
    int numLayers = neurons.size();
    float aux = 0;
    for (int p = 0; p < int(neurons[numLayers - 1].size()); ++p)
      aux += partialsOutputNeurons[q][p]
        * partialNeuronWeight(numLayers - 1, p, t, i, j);
    derivative += (aux * errorDerivative(data.out[q], out[q]));
  }

  return derivative;
}


float NeuralNetwork::partialDataErrorBias(int t, int i, const  Data& data) {
  float derivative = 0;
  for (int q = 0; q < int(out.size()); ++q) {
    int numLayers = neurons.size();
    float aux = 0;
    for (int p = 0; p < int(neurons[numLayers - 1].size()); ++p)
      aux += partialsOutputNeurons[q][p] * partialNeuronBias(numLayers - 1, p, t, i);
    derivative += (aux * errorDerivative(data.out[q], out[q]));
  }

  return derivative;
}


void NeuralNetwork::initializeGradient() {
  gradient.weights = VVVF(neurons.size() - 1);
  gradient.biases =  VVF(neurons.size() - 1);
  for (int l = 0; l < int(neurons.size()) - 1; ++l) {
    gradient.weights[l] = VVF(neurons[l + 1].size(), VF(neurons[l].size(), 0));
    gradient.biases[l] = VF(neurons[l + 1].size(), 0);
  }
}


void NeuralNetwork::initializePartialsNeuronsWeights() {
  partialsNeuronsWeights = VVVVVF(neurons.size());
  for (int l = 0; l < int(partialsNeuronsWeights.size()); ++l) {
    partialsNeuronsWeights[l] = VVVVF(neurons[l].size());
    for (int k = 0; k < int(partialsNeuronsWeights[l].size()); ++k) {
      partialsNeuronsWeights[l][k] = VVVF(neurons.size() - 1);
      for (int t = 0; t < int(partialsNeuronsWeights[l][k].size()); ++t)
        partialsNeuronsWeights[l][k][t] = VVF(neurons[t + 1].size(),
                                              VF(neurons[t].size(), -1));
    }
  }
}


void NeuralNetwork::initializePartialsNeuronsBiases() {
  partialsNeuronsBiases = VVVVF(neurons.size());
  for (int l = 0; l < int(partialsNeuronsBiases.size()); ++l) {
    partialsNeuronsBiases[l] = VVVF(neurons[l].size());
    for (int k = 0; k < int(partialsNeuronsBiases[l].size()); ++k) {
      partialsNeuronsBiases[l][k] = VVF(neurons.size() - 1);
      for (int t = 0; t < int(partialsNeuronsBiases[l][k].size()); ++t)
        partialsNeuronsBiases[l][k][t] = VF(neurons[t].size(), -1);
    }
  }
}


void NeuralNetwork::partialOutputNeurons() {
  int numLayers = neurons.size();
  partialsOutputNeurons = VVF(neurons[numLayers - 1].size(),
                              VF(neurons[numLayers - 1].size()));
  for (int p = 0; p < int(partialsOutputNeurons.size()); ++p)
    for (int q = 0; q < int(partialsOutputNeurons[p].size()); ++q)
      partialsOutputNeurons[p][q] = convertIntoProbDistDerivative(p, q, out);
}


float NeuralNetwork::partialNeuronWeight(int l, int k, int t, int i, int j) {
  float& partial = partialsNeuronsWeights[l][k][t][i][j];
  if (partial != -1)
    return partial;
  if (t > l - 1 or (t == l - 1 and k != i))
    return partial = 0;

  if (t == l - 1 and k == i)
    return partial = activationDerivative(neuronsNotActivated[l - 1][k])
      * neurons[l - 1][j];

  float aux = 0;
  for (int p = 0; p < int(neurons[l - 1].size()); ++p)
    aux += (weights[l - 1][k][p] * partialNeuronWeight(l - 1, p, t, i, j));

  return partial = (activationDerivative(neuronsNotActivated[l - 1][k]) * aux);
}


float NeuralNetwork::partialNeuronBias(int l, int k, int t, int i) {
  float& partial = partialsNeuronsBiases[l][k][t][i];
  if (partial != -1)
    return partial;
  if (t > l - 1 or (t == l - 1 and k != i))
    return partial = 0;

  if (t == l - 1 and k == i)
    return partial = activationDerivative(neuronsNotActivated[l - 1][k]);

  float aux = 0;
  for (int p = 0; p < int(neurons[l - 1].size()); ++p)
    aux += (weights[l - 1][k][p] * partialNeuronBias(l - 1, p, t, i));

  return partial = (activationDerivative(neuronsNotActivated[l - 1][k]) * aux);
}


void NeuralNetwork::updateWeightsAndBiases() {
  const float alpha = 0.01; // Learn update value.

  // Update normal weights.
  for (int l = 0; l < int(weights.size()); ++l)
    for (int i = 0; i < int(weights[l].size()); ++i)
      for (int j = 0; j < int(weights[l][i].size()); ++j)
        weights[l][i][j] -= alpha * gradient.weights[l][i][j];

  // Update biases.
  for (int l = 0; l < int(biases.size()); ++l)
    for (int i = 0; i < int(biases[l].size()); ++i)
      biases[l][i] -= alpha * gradient.biases[l][i];
}


float NeuralNetwork::errorData(const Data& data) {
  float error = 0;
  for (int i = 0; i < int(data.out.size()); ++i)
      error += errorFunction(data.out[i], out[i]);
  return error;
}
