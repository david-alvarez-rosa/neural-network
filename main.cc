#include <iostream>
#include <fstream>
#include <vector>
#include "Defs.hh"
#include "Math.hh"
#include "NeuralNetwork.hh"
#include "Data.hh"


int main() {
  std::cout.setf(std::ios::fixed);
  std::cout.precision(6);

  // Read data.
  const int sizeTrainDataset = 1;
  std::vector<Data> trainDataset;
  std::ifstream fileTrain("data/train.dat");
  for (int i = 0; i < sizeTrainDataset; ++i)
    trainDataset.push_back(Data(fileTrain));

  const int sizeTestDataset = 0;
  std::vector<Data> testDataset;
  std::ifstream fileTest("data/test.dat");
  for (int i = 0; i < sizeTestDataset; ++i)
    testDataset.push_back(Data(fileTest));

  // Choose model and initialize Neural Network.
  std::vector<int> neuronsPerLayer = {28*28, 10};
  NeuralNetwork neuralNetwork(neuronsPerLayer);

  // Train Neural Network.
  neuralNetwork.train(trainDataset, 5, 100, 0.01);

  // Test Neural Network.
  neuralNetwork.test(testDataset);
}
