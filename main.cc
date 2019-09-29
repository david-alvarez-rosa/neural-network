#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include "defs.hh"
#include "math.hh"
#include "NeuralNetwork.hh"

const int numTrainingImages = 300;
const int numTestImages = 10;


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
  NeuralNetwork neuralNetwork(neuronsPerLayer);

  print(neuralNetwork.weights);

  cout << endl;
  VF x = {0.3, 0.8, 0.1, 0.9, 0.9};
  // VVF X = forwardPropagation(neuralNetwork, x);

  // VF y = X[X.size() - 1];
  // applySoftmax(y);
  // print(y);
  testImag
    }
