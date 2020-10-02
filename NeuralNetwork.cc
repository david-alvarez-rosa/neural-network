#include "NeuralNetwork.hh"


#include <iostream>



NeuralNetwork::NeuralNetwork(std::vector<int> neuronsPerLayer) {
  numLayers = neuronsPerLayer.size();

  for (int l = 0; l < numLayers - 1; ++l)
    layers.push_back(Layer(neuronsPerLayer[l], neuronsPerLayer[l + 1]));

  // Create output layer.
  layers.push_back(Layer(neuronsPerLayer[numLayers - 1]));

  // Connect layers.
  layers[0].nextLayer = &layers[1];
  for (int l = 1; l < numLayers - 1; ++l) {
    layers[l].prevLayer = &layers[l - 1];
    layers[l].nextLayer = &layers[l + 1];
  }
  layers[numLayers - 1].prevLayer = &layers[numLayers - 2];
}


void NeuralNetwork::feedForward(VF x) {
  // Pass input to first layer.
  Layer& layer = layers[0];
  for (int i = 0; i < int(x.size()); ++i) {
    layer.neurons[i].deactivated = x[i];
    layer.neurons[i].activated = x[i];
  }

  for (int l = 0; l < numLayers - 1; ++l)
    layers[l].forward();



  // Without softmax.
  Layer& lastLayer = layers[numLayers - 1];
  out = VF(lastLayer.size);
  for (int i = 0; i < lastLayer.size; ++i)
    out[i] = lastLayer.neurons[i].deactivated;
  return;







  // Softmax last layer.
  // Layer& lastLayer = layers[numLayers - 1];
  out = VF(lastLayer.size);
  double aux = 0;
  for (int i = 0; i < lastLayer.size; ++i)
    aux += exponential(lastLayer.neurons[i].deactivated);
  for (int i = 0; i < lastLayer.size; ++i)
    out[i] = exponential(lastLayer.neurons[i].deactivated)/aux;
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
    // saveData();
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
  if (batchSize*(iterNumber + 1) >= int(order.size()))
    batchSize = order.size() - batchSize*iterNumber;

  double error = 0; int correct = 0;
  for (int i = 0; i < batchSize; ++i) {
    int d = order[batchSize*iterNumber + i];

    feedForward(trainDataset[d].in);

    // Compute error and accuracy.
    error += errorData(trainDataset[d]);
    int number = vectorMaxPos(out);
    if (number == vectorMaxPos(trainDataset[d].out))
      ++correct;

    backPropagate();

    dataGradientNumerical(d);
  }

  // updateWeightsAndBiases();
  std::cout << "Error: " << error / double(batchSize) << "\t\t"
            << "Accuraccy: " << correct / double(batchSize) * 100 << std::endl;
}


void NeuralNetwork::updateWeightsAndBiases() {
  for (int l = 0; l < numLayers - 1; ++l) {
    Layer& layer = layers[l];
    Layer& nextLayer = layers[l + 1];
    for (int i = 0; i < nextLayer.size; ++i) {
      for (int j = 0; j < layer.size; ++j)
        layer.weights[i][j].value -= learningRate*layer.weights[i][j].gradient;
      layer.biases[i].value -= learningRate*layer.biases[i].gradient;
    }

  }
}


void NeuralNetwork::dataGradientNumerical(int d) {
  for (int l = 0; l < numLayers - 1; ++l) {
    Layer& layer = layers[l];
    Layer& nextLayer = layers[l + 1];
    for (int i = 0; i < nextLayer.size; ++i) {
      for (int j = 0; j < layer.size; ++j) {
        double EPS = 1e-6;

        layer.weights[i][j].value -= EPS;
        feedForward(trainDataset[d].in);
        double errorMinus = errorData(trainDataset[d]);

        layer.weights[i][j].value += 2*EPS;
        feedForward(trainDataset[d].in);
        double errorPlus = errorData(trainDataset[d]);

        layer.weights[i][j].value -= EPS;


        std::cout << layer.weights[i][j].gradient << std::endl;
        std::cout << layer.weights[i][j].gradient - (errorPlus - errorMinus)/(2*EPS)
                  << std::endl << std::endl;
      }
      double EPS = 1e-5;

      layer.biases[i].value -= EPS;
      feedForward(trainDataset[d].in);
      double errorMinus = errorData(trainDataset[d]);

      layer.biases[i].value += 2*EPS;
      feedForward(trainDataset[d].in);
      double errorPlus = errorData(trainDataset[d]);

      layer.biases[i].value -= EPS;

      layer.biases[i].gradient = (errorPlus - errorMinus)/(2*EPS);
    }
  }
}


void NeuralNetwork::backPropagate() {
  for (int l = numLayers - 1; l >= 0; --l) {
    layers[l].backward();
    layers[l].computeGradients();
  }
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


double NeuralNetwork::errorData(const Data& data) {
  double error = 0;
  for (int i = 0; i < int(data.out.size()); ++i)
    error += errorFunction(data.out[i], out[i]);
  return error;
}


// void NeuralNetwork::saveData() {
//   // This saves the weigths in a file.
//   std::ofstream fileWeights("weights.dat");
//   for (int l = 0; l < int(weights.size()); ++l)
//     for (int i = 0; i < int(weights[l].size()); ++i)
//       for (int j = 0; j < int(weights[l][i].size()); ++j)
//         fileWeights << weights[l][i][j] << std::endl;

//   // This saves the biases in a file.
//   std::ofstream fileBiases("biases.dat");
//   for (int l = 0; l < int(biases.size()); ++l)
//     for (int i = 0; i < int(biases[l].size()); ++i)
//       fileBiases << biases[l][i] << std::endl;
// }
