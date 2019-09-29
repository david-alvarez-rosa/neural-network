#include "NeuralNetwork.hh"


NeuralNetwork::NeuralNetwork(VI neuronsPerLayer) {
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


void NeuralNetwork::update(VF X, VF Y) {
  neurons[0] = X;
  forwardPropagation();
  backPropagation();
}


void NeuralNetwork::forwardPropagation() {
  for (int l = 0; l < int(neurons.size()) - 1; ++l) {
    neurons[l + 1] = sum(multiply(weights[l], neurons[l]), biases[l]);
    activateNeurons(l + 1);
  }
}


void NeuralNetwork::backPropagation() {
  for (int l = 0; l < int(weights.size()); ++l)
    for (int i = 0; i < int(weights[l].size()); ++i)
      for (int j = 0; j < int(weights[j].size()); ++j)
        backPropagationWeight(l, i, j);

  for (int l = 0; l < int(biases.size()); ++l)
    for (int i = 0; i < int(biases.size()); ++i)
      backPropagationBias(l, i);
}


void NeuralNetwork::backPropagationWeight(int l, int i, int j) {
  // weightsGradient[l][i][j] = fp(multiply(neurons[l][j]))
    }


void NeuralNetwork::backPropagationBias(int l, int i) {
  for (int k = 0; k < 2; ++k)
    cout << "todo" << endl;
}


void NeuralNetwork::activateNeurons(int l) {
  for (int i = 0; i < int(neurons[l].size()); ++i)
    neurons[l][i] = activationFunction(neurons[l][i]);
}
