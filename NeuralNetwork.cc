#include "NeuralNetwork.hh"


// TODO: delete this line, when saving weights is done in other fileWeights.
#include <fstream>
#include <algorithm> // TODO: check this. This is used for random shuffle.



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

  // weights[0] = {{0.16, 0.02, 0.63, 0.36},
  //               {0.16, 0.25, 0.22, 0.29}};
  // weights[1] = {{0.05, 0.43},
  //               {0.33, 0.79},
  //               {0.72, 0.91}};
}



void NeuralNetwork::train(const std::vector<Data>& dataset, int epochs,
                          int batchSize, float alpha) {
  // Show initial status.
  test(dataset);

  for (int epoch = 0; epoch < epochs; ++epoch) {
    std::cout << std::endl << "Epoch " << epoch + 1 << std::endl;
    trainEpoch(dataset, batchSize, alpha);
  }
}


void NeuralNetwork::trainEpoch(const std::vector<Data>& dataset, int batchSize,
                               float alpha) {
  // Create random vector.
  std::vector<int> order(dataset.size());
  for (int i = 0; i < int(order.size()); ++i)
    order[i] = i;
  std::random_shuffle(order.begin(), order.end());

  for (int i = 0; i < int(dataset.size()) / float(batchSize); ++i) {
    trainIteration(dataset, batchSize, alpha, order, i);
    std::cout << "Iter: " << i + 1 << "\t\t";
    test(dataset);
  }
}


void NeuralNetwork::trainIteration(const std::vector<Data>& dataset, int batchSize,
                                   float alpha, const std::vector<int>& order,
                                   int iterNumber) {
  initializeGradient();

  if (batchSize*(iterNumber + 1) >= int(order.size()))
    batchSize = order.size() - batchSize*iterNumber;

  for (int i = 0; i < batchSize; ++i) {
    initializePartialsNeuronsWeights();
    initializePartialsNeuronsBiases();

    // std::cout << "partial: " << partialNeuronWeight(2, 0, 1, 0, 0) << std::endl;
    // return; // TODO get rid of this and the above.

    feedForward(dataset[order[batchSize*iterNumber + i]].in);
    dataGradient(dataset[order[batchSize*iterNumber + i]], batchSize);
  }

  updateWeightsAndBiases(alpha);
}


void NeuralNetwork::test(const std::vector<Data>& dataset) {
  float error = 0;
  int correct = 0;

  for (int d = 0; d < int(dataset.size()); ++d) {
    feedForward(dataset[d].in);
    error += errorData(dataset[d]) / float(dataset.size());

    int number = vectorMaxPos(out);
    if (number == vectorMaxPos(dataset[d].out))
      ++correct;
  }

  saveData(error, correct / float(dataset.size()));
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


void NeuralNetwork::dataGradient(const Data& data, int batchSize) {
  // Compute partial derivatives of output respect neurons.
  partialOutputNeurons();

  // Gradient respect weights.
  for (int t = 0; t < int(weights.size()); ++t)
    for (int i = 0; i < int(weights[t].size()); ++i)
      for (int j = 0; j < int(weights[t][i].size()); ++j)
        gradient.weights[t][i][j] +=
          partialDataErrorWeight(t, i, j, data) / float(batchSize);

  // Gradient respect biases.
  for (int t = 0; t < int(biases.size()); ++t)
    for (int i = 0; i < int(biases[t].size()); ++i)
      gradient.biases[t][i] += partialDataErrorBias(t, i, data) / float(batchSize);
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


float NeuralNetwork::partialDataErrorBias(int t, int i, const Data& data) {
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
        partialsNeuronsBiases[l][k][t] = VF(neurons[t + 1].size(), -1);
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

  if (t == l - 1 and k == i) {
    // std::cout << "neuronsNotActivated[l - 1][k]: "
    //           << neuronsNotActivated[l - 1][k] << std::endl;
    // std::cout << "activationDerivative(neuronsNotActivated[l - 1][k]): "
    //           << activationDerivative(neuronsNotActivated[l][k]) << std::endl;
    // std::cout << "neurons[l - 1][j]: " << neurons[l - 1][j] << std::endl;
    // std::cout << "derivative: "
    //           << activationDerivative(neuronsNotActivated[l][k]) * neurons[l - 1][j]
    //           << std::endl;
    return partial = activationDerivative(neuronsNotActivated[l][k])
      * neurons[l - 1][j];
  }

  float aux = 0;
  for (int p = 0; p < int(neurons[l - 1].size()); ++p)
    aux += (weights[l - 1][k][p] * partialNeuronWeight(l - 1, p, t, i, j));

  return partial = (activationDerivative(neuronsNotActivated[l][k]) * aux);
}


float NeuralNetwork::partialNeuronBias(int l, int k, int t, int i) {
  float& partial = partialsNeuronsBiases[l][k][t][i];
  if (partial != -1)
    return partial;
  if (t > l - 1 or (t == l - 1 and k != i))
    return partial = 0;

  if (t == l - 1 and k == i)
    return partial = activationDerivative(neuronsNotActivated[l][k]);

  float aux = 0;
  for (int p = 0; p < int(neurons[l - 1].size()); ++p)
    aux += (weights[l - 1][k][p] * partialNeuronBias(l - 1, p, t, i));

  return partial = (activationDerivative(neuronsNotActivated[l][k]) * aux);
}


void NeuralNetwork::updateWeightsAndBiases(float alpha) {
  // Update normal weights.
  for (int l = 0; l < int(weights.size()); ++l) {
    // std::cout << "l: " << l << std::endl;
    for (int i = 0; i < int(weights[l].size()); ++i)
      for (int j = 0; j < int(weights[l][i].size()); ++j) {
        // std::cout << gradient.weights[l][i][j] << std::endl;
        weights[l][i][j] -= alpha * gradient.weights[l][i][j];
      }
      }

  // TODO: remember to descomment this!!!!
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


void NeuralNetwork::saveData(float error, float accuracy) {
  // This saves the weigths in a file.
  std::ofstream fileWeights("weights.dat");
  // Normal weights.
  for (int l = 0; l < int(weights.size()); ++l)
    for (int i = 0; i < int(weights[l].size()); ++i)
      for (int j = 0; j < int(weights[l][i].size()); ++j)
        fileWeights << weights[l][i][j] << std::endl;
  // Biases.
  for (int l = 0; l < int(biases.size()); ++l)
    for (int i = 0; i < int(biases[l].size()); ++i)
      fileWeights << biases[l][i] << std::endl;

  // This saves the neurons in a file.
  std::ofstream fileNeurons("neurons.dat");
  // Normal weights.
  for (int l = 0; l < int(neurons.size()); ++l) {
    fileNeurons << std::endl;
    for (int i = 0; i < int(neurons[l].size()); ++i)
      fileNeurons << neurons[l][i] << std::endl;
  }
  fileNeurons << std::endl;
  for (int i = 0; i < int(out.size()); ++i)
    fileNeurons << out[i] << std::endl;

  std::cout << "Error: " << error << "\t\t" << "Accuraccy: "
            << 100 * accuracy << std::endl;
}
