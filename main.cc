#include <iostream>
#include <fstream>
#include <vector>
#include "defs.hh"
#include "math.hh"
#include "NeuralNetwork.hh"
#include "Image.hh"


int main() {
  cout.setf(ios::fixed);
  cout.precision(4);

  // Read data.
  const int numTrainImages = 2;
  vector<Image> trainImages;
  ifstream file("data/train.dat");
  for (int i = 0; i < numTrainImages; ++i)
    trainImages.push_back(Image(file));

  const int numTestImages = 21;
  vector<Image> testImages;
  file = ifstream("data/test.dat");
  for (int i = 0; i < numTestImages; ++i)
    testImages.push_back(Image(file));

  // Choose model and initialize Neural Network.
  VI neuronsPerLayer = {28*28, 300, 75, 10};
  NeuralNetwork neuralNetwork(neuronsPerLayer);

  // Train Neural Network.
  neuralNetwork.train(trainImages);

  // Test Neural Network.
  neuralNetwork.test(testImages);
}
