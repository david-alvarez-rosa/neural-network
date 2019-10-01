#include <iostream>
#include <fstream>
#include <vector>
#include "Defs.hh"
#include "Math.hh"
#include "NeuralNetwork.hh"
#include "Data.hh"


int main() {
  cout.setf(ios::fixed);
  cout.precision(4);

  // Read data.
  const int sizeTrainDataset = 3;
  vector<Data> trainDataset;
  // ifstream file("test/train.dat");
  ifstream file("data/train.dat");
  for (int i = 0; i < sizeTrainDataset; ++i)
    trainDataset.push_back(Data(file));

  const int sizeTestDataset = 0;
  vector<Data> testDataset;
  file = ifstream("data/test.dat");
  for (int i = 0; i < sizeTestDataset; ++i)
    testDataset.push_back(Data(file));

  // Choose model and initialize Neural Network.
  // VI neuronsPerLayer = {3, 2};

  VI neuronsPerLayer = {28*28, 40, 30, 10}; //
  NeuralNetwork neuralNetwork(neuronsPerLayer);

  // Train Neural Network.
  neuralNetwork.train(trainDataset);

  // Test Neural Network.
  neuralNetwork.test(testDataset);
}
