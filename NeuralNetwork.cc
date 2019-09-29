#include "NeuralNetwork.hh"


NeuralNetwork::NeuralNetwork(VI neuronsPerLayer) {
  int numLayers = neuronsPerLayer.size();
  neurons = VVF(numLayers);
  weights = VVVF(numLayers - 1); gradient.weights = VVVF(numLayers - 1);
  biases =  VVF(numLayers - 1); gradient.biases =  VVF(numLayers - 1);
  for (int l = 0; l < numLayers; ++l)
    neurons[l]  = VF(neuronsPerLayer[l]);

  for (int l = 0; l < numLayers - 1; ++l) {
    weights[l] = VVF(neuronsPerLayer[l + 1], VF(neuronsPerLayer[l]));
    gradient.weights[l] = VVF(neuronsPerLayer[l + 1], VF(neuronsPerLayer[l]));
    fillRandomly(weights[l]);
    biases[l] = VF(neuronsPerLayer[l + 1]);
    gradient.biases[l] = VF(neuronsPerLayer[l + 1]);
    fillRandomly(biases[l]);
  }
}


void NeuralNetwork::train(vector<Image>& images) {
  for (int i = 0; i < int(images.size()); ++i) {
    forwardPropagation(images[i].pixels);
    backPropagation();
    updateWeightsAndBiases();
  }
}


void NeuralNetwork::test(vector<Image>& images) {
  int correct = 0;
  for (int i = 0; i < int(images.size()); ++i) {
    forwardPropagation(images[i].pixels);
    int number = vectorMaxPos(neurons[neurons.size() - 1]);
    cout << number << '\t' << images[i].label << endl;
    if (number == images[i].label)
      ++correct;
  }

  float percentage = 100 * correct / float(images.size());
  cout << "Percentage: " << percentage << "%." << endl;
}


void NeuralNetwork::forwardPropagation(VF X) {
  neurons[0] = X;
  for (int l = 0; l < int(neurons.size()) - 1; ++l) {
    neurons[l + 1] = sum(multiply(weights[l], neurons[l]), biases[l]);
    activateNeurons(l + 1);
  }
  applySoftmax(neurons[neurons.size() - 1]);
}


void NeuralNetwork::activateNeurons(int l) {
  for (int i = 0; i < int(neurons[l].size()); ++i)
    neurons[l][i] = activationFunction(neurons[l][i]);
}


void NeuralNetwork::backPropagation() {
  for (int l = 0; l < int(weights.size()); ++l)
    for (int i = 0; i < int(weights[l].size()); ++i)
      for (int j = 0; j < int(weights[l][i].size()); ++j)
        backPropagationWeight(l, i, j);

  for (int l = 0; l < int(biases.size()); ++l)
    for (int i = 0; i < int(biases[l].size()); ++i)
      backPropagationBias(l, i);
}


void NeuralNetwork::backPropagationWeight(int l, int i, int j) {
  gradient.weights[l][i][j] = 0.012 * (rand() / RAND_MAX - 0.5);
}


void NeuralNetwork::backPropagationBias(int l, int i) {
  gradient.biases[l][i] = 0.013 * (rand() / RAND_MAX - 0.5);
}


void NeuralNetwork::updateWeightsAndBiases() {
  const float alpha = 0.1; // Learn update value.

  // Update normal weights.
  for (int l = 0; l < int(weights.size()); ++l)
    for (int i = 0; i < int(weights[l].size()); ++i)
      for (int j = 0; j < int(weights[l][i].size()); ++j)
        weights[l][i][j] += alpha * gradient.weights[l][i][j];

  // Update biases.
  for (int l = 0; l < int(biases.size()); ++l)
    for (int i = 0; i < int(biases[l].size()); ++i)
      biases[l][i] += alpha * gradient.biases[l][i];
}
