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


void NeuralNetwork::train(const std::vector<Data>& trainDataset, int epochs,
                          int batchSize, double learningRate) {
  this->learningRate = learningRate;
  this->trainDataset = trainDataset;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    // Show status.
    test(trainDataset);
    std::cout << std::endl << "Epoch " << epoch + 1 << std::endl;
    trainEpoch(batchSize);
    saveData();
  }
}


void NeuralNetwork::trainEpoch(int batchSize) {
  // Create random vector.
  std::vector<int> order(trainDataset.size());
  for (int i = 0; i < int(order.size()); ++i)
    order[i] = i;
  std::random_shuffle(order.begin(), order.end());

  for (int i = 0; i < int(trainDataset.size()) / double(batchSize); ++i) {
    std::cout << "Iter: " << i + 1 << "\t\t";
    trainIteration(batchSize, order, i);
  }
}


void NeuralNetwork::trainIteration(int batchSize, const std::vector<int>& order,
                                   int iterNumber) {
  initializeGradient();

  if (batchSize*(iterNumber + 1) >= int(order.size()))
    batchSize = order.size() - batchSize*iterNumber;

  double error = 0; int correct = 0;
  for (int i = 0; i < batchSize; ++i) {
    int d = order[batchSize*iterNumber + i];
    initializePartialsNeuronsWeights();
    initializePartialsNeuronsBiases();

    feedForward(trainDataset[d].in);

    // Compute error and accuracy.
    error += errorData(trainDataset[d]);
    int number = vectorMaxPos(out);
    if (number == vectorMaxPos(trainDataset[d].out))
      ++correct;

    dataGradient(d, batchSize);
  }

  updateWeightsAndBiases();
  std::cout << "Error: " << error / double(batchSize) << "\t\t"
            << "Accuraccy: " << correct / double(batchSize) * 100 << std::endl;
}


void NeuralNetwork::test(const std::vector<Data>& testDataset) {
  this->testDataset = testDataset;

  double error = 0;
  int correct = 0;

  for (int d = 0; d < int(testDataset.size()); ++d) {
    feedForward(testDataset[d].in);
    error += errorData(testDataset[d]);

    int number = vectorMaxPos(out);
    if (number == vectorMaxPos(testDataset[d].out))
      ++correct;
  }

  std::cout << "Error: " << error / double(testDataset.size()) << "\t\t"
            << "Accuraccy: " << 100 * correct / double(testDataset.size())
            << std::endl;
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


void NeuralNetwork::dataGradient(int d, int batchSize) {
  // Compute partial derivatives of output respect neurons.
  partialOutputNeurons();

  // Gradient respect weights.
  for (int t = 0; t < int(weights.size()); ++t)
    for (int i = 0; i < int(weights[t].size()); ++i)
      for (int j = 0; j < int(weights[t][i].size()); ++j)
        gradient.weights[t][i][j] +=
          (partialDataErrorWeight(d, t, i, j) / double(batchSize));

  // Gradient respect biases.
  for (int t = 0; t < int(biases.size()); ++t)
    for (int i = 0; i < int(biases[t].size()); ++i)
      gradient.biases[t][i] += (partialDataErrorBias(d, t, i) / double(batchSize));
}


double NeuralNetwork::partialDataErrorWeight(int d, int t, int i, int j) {
  double derivative = 0;
  for (int q = 0; q < int(out.size()); ++q) {
    int numLayers = neurons.size();
    double aux = 0;
    for (int p = 0; p < int(neurons[numLayers - 1].size()); ++p)
      aux += partialsOutputNeurons[q][p]
        * partialNeuronWeight(numLayers - 1, p, t, i, j);
    derivative += (aux * errorDerivative(trainDataset[d].out[q], out[q]));
  }

  return derivative;
}


double NeuralNetwork::partialDataErrorBias(int d, int t, int i) {
  double derivative = 0;
  for (int q = 0; q < int(out.size()); ++q) {
    int numLayers = neurons.size();
    double aux = 0;
    for (int p = 0; p < int(neurons[numLayers - 1].size()); ++p)
      aux += partialsOutputNeurons[q][p] * partialNeuronBias(numLayers - 1, p, t, i);
    derivative += (aux * errorDerivative(trainDataset[d].out[q], out[q]));
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
      partialsNeuronsWeights[l][k] = VVVF(l);
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
      partialsNeuronsBiases[l][k] = VVF(l);
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


double NeuralNetwork::partialNeuronWeight(int l, int k, int t, int i, int j) {
  double& partial = partialsNeuronsWeights[l][k][t][i][j];
  if (partial != -1)
    return partial;
  if (t == l - 1) {
    if (k != i)
      return partial = 0;
    return partial = activationDerivative(neuronsNotActivated[l][k])
      * neurons[l - 1][j];
  }

  double aux = 0;
  for (int p = 0; p < int(neurons[l - 1].size()); ++p)
    aux += (weights[l - 1][k][p] * partialNeuronWeight(l - 1, p, t, i, j));

  return partial = (activationDerivative(neuronsNotActivated[l][k]) * aux);
}


double NeuralNetwork::partialNeuronBias(int l, int k, int t, int i) {
  double& partial = partialsNeuronsBiases[l][k][t][i];
  if (partial != -1)
    return partial;
  if (t == l - 1) {
    if (k != i)
      return partial = 0;
    return partial = activationDerivative(neuronsNotActivated[l][k]);
  }

  double aux = 0;
  for (int p = 0; p < int(neurons[l - 1].size()); ++p)
    aux += (weights[l - 1][k][p] * partialNeuronBias(l - 1, p, t, i));

  return partial = (activationDerivative(neuronsNotActivated[l][k]) * aux);
}


void NeuralNetwork::updateWeightsAndBiases() {
  // Update normal weights.
  for (int l = 0; l < int(weights.size()); ++l)
    for (int i = 0; i < int(weights[l].size()); ++i)
      for (int j = 0; j < int(weights[l][i].size()); ++j)
        weights[l][i][j] -= learningRate * gradient.weights[l][i][j];

  // Update biases.
  for (int l = 0; l < int(biases.size()); ++l)
    for (int i = 0; i < int(biases[l].size()); ++i)
      biases[l][i] -= learningRate * gradient.biases[l][i];
}


double NeuralNetwork::errorData(const Data& data) {
  double error = 0;
  for (int i = 0; i < int(data.out.size()); ++i)
    error += errorFunction(data.out[i], out[i]);
  return error;
}


void NeuralNetwork::saveData() {
  // This saves the weigths in a file.
  std::ofstream fileWeights("weights.dat");
  for (int l = 0; l < int(weights.size()); ++l)
    for (int i = 0; i < int(weights[l].size()); ++i)
      for (int j = 0; j < int(weights[l][i].size()); ++j)
        fileWeights << weights[l][i][j] << std::endl;

  // This saves the biases in a file.
  std::ofstream fileBiases("biases.dat");
  for (int l = 0; l < int(biases.size()); ++l)
    for (int i = 0; i < int(biases[l].size()); ++i)
      fileBiases << biases[l][i] << std::endl;
}
