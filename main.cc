#include <iostream>
#include <vector>
#include "defs.hh"
#include "math.hh"
#include "NeuralNetwork.hh"
#include "Image.hh"



int main() {
  // Read data.
  const int numTrainImages = 300;
  vector<Image> trainImages;
  for (int i = 0; i < numTrainImages; ++i)
    trainImages.push_back(Image("train"));

  const int numTestImages = 10;
  vector<Image> testImages;
  for (int i = 0; i < numTestImages; ++i)
    testImages.push_back(Image("test"));

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
  // testImag
}
