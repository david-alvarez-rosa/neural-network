#include "NeuralNetwork.hh"


NeuralNetwork::NeuralNetwork(VI neuronsPerLayer) {
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
    biases[l] = VF(neuronsPerLayer[l + 1], 0);
    // fillRandomly(biases[l]);
  }
}


void NeuralNetwork::train(vector<Data>& dataset) {
  for (int step = 0; step < 25; ++step) {
    // cout << "Train_step: " << step << endl;
    trainStep(dataset);
    // cout << endl << endl << endl;
    cout << endl;
  }
}


void NeuralNetwork::trainStep(vector<Data>& dataset) {
  // Initialize gradient with 0's.
  initializeGradient();

  // For coputing the error over all the dataset.
  float error = 0;

  for (int d = 0; d < int(dataset.size()); ++d) {
    initializePartialsNeuronsWeights();
    cout << "d: " << d << endl;
    // cout << "Feeding forward process starts." << endl;
    feedForward(dataset[d].in);

    // cout << "neurons: " << endl;
    // print(neurons);

    cout << "out: ";
    print(out);

    // cout << "Last_layer: ";
    // print(neurons[neurons.size() - 1]);

    // cout << "Weights: ";
    // print(weights);

    // ofstream file("weights.dat");
    // for (int k = 0; k < int(weights.size()); ++k) {
    //   file << "----------------" << k << "----------------" << endl;
    //   for (int i = 0; i < int(weights[k].size()); ++i) {
    //     for (int j = 0; j < int(weights[k][i].size()); ++j)
    //       file << weights[k][i][j] << "\t";
    //     file << endl;
    //   }
    // }

    // cout << "Computing dataGradient process starts." << endl;
    dataGradient(dataset[d]);

    // cout << "gradient: ";
    // print(gradient.weights);

    // ofstream file2("gradient.dat");
    // for (int k = 0; k < int(weights.size()); ++k) {
    //   file2 << "----------------" << k << "----------------" << endl;
    //   for (int i = 0; i < int(weights[k].size()); ++i) {
    //     for (int j = 0; j < int(weights[k][i].size()); ++j)
    //       file2 << weights[k][i][j] << "\t";
    //     file2 << endl;
    //   }
    // }

    // cout << "asdfasdf: " << endl;
    // for (int l = 0; l < neurons.size(); ++l)
    //   for (int k = 0; k < neurons[l].size(); ++k) {
    //     cout << "l_y_k: " << l << " " << k << endl;
    //     print(partialsNeuronsWeights[l][k]);
    //   }
    // cout << endl << endl;

    error += errorData(dataset[d]);
  }
  // cout << "Gradient:" << endl;
  // print(gradient.weights);
  // cout << "Updating weights process starts." << endl;
  updateWeightsAndBiases();

  cout << "Error_dataset: " << error << endl;
}


void NeuralNetwork::test(vector<Data>& dataset) {
  int correct = 0;
  for (int i = 0; i < int(dataset.size()); ++i) {
    cout << "Test_iteration: " << i << endl;
    feedForward(dataset[i].in);
    int number = vectorMaxPos(out);
    cout << number << endl;
  }

  float percentage = 100 * correct / float(dataset.size());
  cout << "Percentage: " << percentage << "%." << endl;
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


void NeuralNetwork::dataGradient(Data& data) {
  // Compute partial derivatives of output respect neurons.
  partialOutputNeurons();

  for (int t = 0; t < int(weights.size()); ++t)
    for (int i = 0; i < int(weights[t].size()); ++i)
      for (int j = 0; j < int(weights[t][i].size()); ++j)
        gradient.weights[t][i][j] += partialDataErrorWeight(t, i, j, data);
}


float NeuralNetwork::partialDataErrorWeight(int t, int i, int j, Data& data) {
  float derivative = 0;
  for (int q = 0; q < int(out.size()); ++q) {
    int numLayers = neurons.size();
    float aux = 0;
    for (int p = 0; p < int(neurons[numLayers - 1].size()); ++p)
      aux += partialsOutputNeurons[q][p] * partialNeuronWeight(numLayers - 1, p, t, i, j);
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


void NeuralNetwork::partialOutputNeurons() {
  int numLayers = neurons.size();
  partialsOutputNeurons = VVF(neurons[numLayers - 1].size(), VF(neurons[numLayers - 1].size())); // TODO: initialized as zero!
  for (int p = 0; p < int(partialsOutputNeurons.size()); ++p)
    // TODO: modify this, is only correct when softmax is not applied!
    for (int q = 0; q < int(partialsOutputNeurons[p].size()); ++q)
      if (p == q)
        partialsOutputNeurons[p][q] = 1;
    // partialsOutputNeurons[p] = convertIntoProbDistDerivative(out, p);
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


void NeuralNetwork::updateWeightsAndBiases() {
  const float alpha = 0.01; // Learn update value.

  // Update normal weights.
  for (int l = 0; l < int(weights.size()); ++l)
    for (int i = 0; i < int(weights[l].size()); ++i)
      for (int j = 0; j < int(weights[l][i].size()); ++j)
        weights[l][i][j] -= alpha * gradient.weights[l][i][j];

  // // Update biases.
  // for (int l = 0; l < int(biases.size()); ++l)
  //   for (int i = 0; i < int(biases[l].size()); ++i)
  //     biases[l][i] -= alpha * gradient.biases[l][i];
}


float NeuralNetwork::errorData(Data& data) {
  float error = 0;
  for (int i = 0; i < int(data.out.size()); ++i)
      error += errorFunction(data.out[i], out[i]);
  return error;
}
