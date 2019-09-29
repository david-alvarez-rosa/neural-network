#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include "functions.hh"

using namespace std;


using VF = vector<float>;
using VVF = vector<VF>;
using VVVF = vector<VVF>;
using VI = vector<int>;


const int numTrainingImages = 300;
const int numTestImages = 10;


struct neuralNetworkStruct {
  int numLayers;
  VI neuronsPerLayer;
  VVVF weightsTensor;
  VVF biasesMatrix;
  VVF neurons;
};

struct gradient {
  VVVF weightsDerivatives;
  VVF biasesDerivatives;
}

// Reads images and labels.
void readData(VVF &images, VF &labels, string dataFileName, int numImages) {
  ifstream dataFile(dataFileName, ifstream::in);
  for (int i = 0; i < numImages; ++i) {
    dataFile >> labels[i];
    for (int j = 0; j < 28*28; ++j) {
      dataFile >> images[i][j];
      images[i][j] /= 255;
    }
  }
}


// Activation function (in place).
void applyActivationFunction(VF &X) {
  applySigmoid(X);
}


// Loss function.
double lossFunction(VVF Y, VVF YP) {
  double loss = 0;
  int m = YP.size();
  for (int i = 0; i < m; ++i)
    loss += euclidianDistanceSquared(YP[i], Y[i]);
  return loss;
}


// Initializes (randomnly) a Neural Network (given sizes of the layers).
neuralNetworkStruct initializeNeuralNetwork(VI neuronsPerLayer) {
  neuralNetworkStruct neuralNetwork;
  int numLayers = neuronsPerLayer.size();
  neuralNetwork.numLayers = numLayers;
  neuralNetwork.neuronsPerLayer = neuronsPerLayer;

  VVVF weightsTensor(numLayers - 1);
  for (int i = 0; i < numLayers - 1; ++i) {
    weightsTensor[i] = VVF(neuronsPerLayer[i + 1], VF(neuronsPerLayer[i]));
    fillMatrixRandomly(weightsTensor[i]);
  }
  neuralNetwork.weightsTensor = weightsTensor;

  VVF biasesMatrix(numLayers - 1);
  for (int i = 0; i < numLayers - 1; ++i) {
    biasesMatrix[i] = VF(neuronsPerLayer[i + 1]);
    fillVectorRandomly(biasesMatrix[i]);
  }
  neuralNetwork.biasesMatrix = biasesMatrix;

  return neuralNetwork;
}


// Forward propagation to compute values of al the neurons, given input.
void forwardPropagation(const neuralNetworkStruct &neuralNetwork, VF X) {
  for (int i = 0; i < neuralNetwork.numLayers - 1; ++i) {
    X = matrixVectorMultiplication(neuralNetwork.weightsTensor[i], X);
    X = vectorSum(neuralNetwork.biasesMatrix[i], X);
    applyActivationFunction(X);
    neuralNetwork.neurons.push_back(X);
  }
}


// Compute the gradient of loss function.
gradient lossFunctionGradient(VF input, VF output) {
  for (int k = 0; k < neuralNetwork.numLayers - 1; ++k)
    for (int i = 0; i < neuralNetwork.neuronsPerLayer[k]; ++i)
      for (int j = 0; j < neuralNetwork.neuronsPerLayer[k + 1]; ++j)
}



int main() {
  // Read data.
  VVF trainingImages(numTrainingImages, VF(28*28)); VF trainingLabels(numTrainingImages);
  string trainingFileName = "labelImagesTraining.dat";
  readData(trainingImages, trainingLabels, trainingFileName, numTrainingImages);

  VVF testImages(numTestImages, VF(28*28)); VF testLabels(numTestImages);
  string testFileName = "labelImagesTest.dat";
  readData(testImages, testLabels, testFileName, numTestImages);

  // Choose model and initialize Neural Network.
  VI neuronsPerLayer = {5, 3, 3, 2};
  neuralNetworkStruct neuralNetwork = initializeNeuralNetwork(neuronsPerLayer);

  printTensor(neuralNetwork.weightsTensor);

  cout << endl;
  VF x = {0.3, 0.8, 0.1, 0.9, 0.9};
  VVF X = forwardPropagation(neuralNetwork, x);

  VF y = X[X.size() - 1];
  applySoftmax(y);
  printVector(y);
}
