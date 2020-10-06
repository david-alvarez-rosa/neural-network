#ifndef LAYER_HH
#define LAYER_HH

#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include "Defs.hh"
#include "Utils.hh"
#include "Math.hh"
#include "Custom.hh"
#include "Data.hh"
#include "Neuron.hh"



class Layer {
public:
  struct Param {
    double value;
    double gradient;
  };

  int size, sizeNextLayer;

  std::vector<Neuron> neurons;
  std::vector< std::vector<Param> > weights;
  std::vector<Param> biases;
  std::vector<double> deltas;

  Layer* prevLayer; Layer* nextLayer;

  Layer(int size, int sizeNextLayer = 0);

  void forward();

  void backward();

  void computeGradients();

  void zeroGradients();

  void updateParameters(const double& learningRate);


private:
  double mean = 0, std = 0.01;

  void fillRandomly(std::vector<Param>& v);

  void fillRandomly(std::vector< std::vector<Param> >& v);
};


#endif
