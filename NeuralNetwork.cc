#include "NeuralNetwork.hh"


NeuralNetwork::NeuralNetwork(VI neuronsPerLayer) {
  int numLayers = neuronsPerLayer.size();
  neurons = VVF(numLayers); neuronsNotActivated = VVF(numLayers);
  weights = VVVF(numLayers - 1); gradient.weights = VVVF(numLayers - 1);
  biases =  VVF(numLayers - 1); gradient.biases =  VVF(numLayers - 1);
  for (int l = 0; l < numLayers; ++l) {
    neurons[l]  = VF(neuronsPerLayer[l]);
    neuronsNotActivated[l]  = VF(neuronsPerLayer[l]);
  }

  for (int l = 0; l < numLayers - 1; ++l) {
    weights[l] = VVF(neuronsPerLayer[l + 1], VF(neuronsPerLayer[l]));
    gradient.weights[l] = VVF(neuronsPerLayer[l + 1], VF(neuronsPerLayer[l], -1));
    fillRandomly(weights[l]);
    biases[l] = VF(neuronsPerLayer[l + 1]);
    gradient.biases[l] = VF(neuronsPerLayer[l + 1], 0);
    // fillRandomly(biases[l]);
  }
}


void NeuralNetwork::train(vector<Data>& dataset) {
  for (int step = 0; step < 2; ++step) {
    cout << "Train_step: " << step << endl;
    trainStep(dataset);
  }
}


void NeuralNetwork::trainStep(vector<Data>& dataset) {
  for (int d = 0; d < int(dataset.size()); ++d) {
      feedForward(dataset[d].in);
      computeGradient(dataset);
      updateWeightsAndBiases();
  }
}


void NeuralNetwork::test(vector<Data>& dataset) {
  int correct = 0;
  for (int i = 0; i < int(dataset.size()); ++i) {
    cout << "Test_iteration: " << i << endl;
    feedForward(dataset[i].in);
    int number = vectorMaxPos(neurons[neurons.size() - 1]);
    cout << number << endl;
  }

  float percentage = 100 * correct / float(dataset.size());
  cout << "Percentage: " << percentage << "%." << endl;
}


void NeuralNetwork::feedForward(VF X) {
  neurons[0] = X;
  for (int l = 0; l < int(neurons.size()) - 1; ++l) {
    neurons[l + 1] = sum(multiply(weights[l], neurons[l]), biases[l]);
    neuronsNotActivated[l + 1] = neurons[l + 1]; // Save this computing gradients.
    activateNeurons(l + 1);
  }

  convertIntoProbDist(neurons[neurons.size() - 1]);
}


void NeuralNetwork::activateNeurons(int l) {
  for (int i = 0; i < int(neurons[l].size()); ++i)
    neurons[l][i] = activation(neurons[l][i]);
}


void NeuralNetwork::computeGradient(vector<Data>& dataset) {
  for (int d = 0; d < int(dataset.size()); ++d) {
    // Initialize with -1's tensor with partial derivatives and gradient with 0's.
    initializeGradientVariables();
    for (int t = 0; t < int(weights.size()); ++t)
      for (int i = 0; i < int(weights[t].size()); ++i)
        for (int j = 0; j < int(weights[t][i].size()); ++j)
          gradient.weights[t][i][j] +=
            (individualGradient(t, i, j, dataset[d]) / float(dataset.size()));
  }
}


float NeuralNetwork::individualGradient(int t, int i, int j, Data& data) {
  float derivative = 0;
  for (int d = 0; d < int(data.out.size()); ++d) {
    int numLayers = neurons.size();
    float aux;
    for (int p = 0; p < int(neurons[numLayers - 1].size()); ++p)
      aux += partialsOutputNeurons[d][p] * partialNeuronWeight(numLayers- 1, p, t, i, j);
    derivative += (aux * errorDerivative(data.out[d], neurons[numLayers - 1][d]));
  }

  return derivative;
}


void NeuralNetwork::initializeGradientVariables() {
  // Partial derivatives tensor.
  partialsNeuronsWeights = VVVVVF(neurons.size());
  for (int l = 0; l < int(neurons.size()); ++l) {
    partialsNeuronsWeights[l] = VVVVF(neurons[l].size());
    for (int k = 0; k < int(partialsNeuronsWeights[l].size()); ++k) {
      partialsNeuronsWeights[l][k] = VVVF(neurons[l].size() - 1);
      for (int t = 0; t < int(partialsNeuronsWeights[l][k].size()); ++t)
        partialsNeuronsWeights[l][k][t] = VVF(neurons[t + 1].size(),
                                              VF(neurons[t].size(), -1));
    }
  }

  // Gradient.
  for (int l = 0; l < int(neurons.size()) - 1; ++l) {
    gradient.weights[l] = VVF(neurons[l + 1].size(), VF(neurons[l].size(), 0));
    gradient.biases[l] = VF(neurons[l + 1].size(), 0);
  }
}


void NeuralNetwork::partialOutputNeurons() {
  int numLayers = neurons.size();
  partialsOutputNeurons = VVF(neurons[numLayers - 1].size());
  for (int p = 0; p < int(partialsOutputNeurons.size()); ++p)
    partialsOutputNeurons[p] = convertIntoProbDistDerivative(neurons[numLayers - 1], p);
}


float NeuralNetwork::partialNeuronWeight(int l, int k, int t, int i, int j) {
  float& partial = partialsNeuronsWeights[l][k][t][i][j];
  if (partial != -1)
    return partial;
  if (t > l -1 or (t = l - 1 and k != i))
    return partial = 0;

  if (t == l - 1 and k == i)
    return partial = activationDerivative(neuronsNotActivated[l - 1][k]) * neurons[l - 1][j];

  partial = 0;
  for (int p = 0; p < int(neurons[l - 1].size()); ++p)
    partial += (weights[l - 1][k][p] * partialNeuronWeight(l - 1, p, t, i, j));

  return partial *= activationDerivative(neuronsNotActivated[l - 1][k]);
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
