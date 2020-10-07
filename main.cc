#include <vector>
#include "NeuralNetwork.hh"
#include "Data.hh"


int main() {
  std::cout.setf(std::ios::fixed);
  std::cout.precision(8);

  // Read data.
  const int sizeTrainDataset = 1;
  std::vector< Data<double> > trainDataset;
  std::ifstream fileTrain("data/train.dat");
  for (int i = 0; i < sizeTrainDataset; ++i)
    trainDataset.push_back(Data<double>(fileTrain));

  const int sizeTestDataset = 2;
  std::vector< Data<double> > testDataset;
  std::ifstream fileTest("data/test.dat");

  for (int i = 0; i < sizeTestDataset; ++i)
    testDataset.push_back(Data<double>(fileTest));

  // Choose model and initialize Neural Network.
  std::vector<int> neuronsPerLayer = {28*28, 25, 15, 10};
  NeuralNetwork<double> neuralNetwork(neuronsPerLayer);

  // Train Neural Network.
  neuralNetwork.train(&trainDataset, 10);

  // // Test Neural Network.
  // neuralNetwork.test(testDataset);
}
