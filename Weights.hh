#ifndef WEIGHTS_HH
#define WEIGHTS_HH

#include <iostream>
#include <fstream>
#include <algorithm>
#include "Defs.hh"
#include "Utils.hh"
#include "Math.hh"
#include "Custom.hh"
#include "Data.hh"


class Weights {
public:
  std::vector< std::vector<ValueGradient> > weights;

  Layer(int size);



private:
  void hello();
};


#endif
