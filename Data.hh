#ifndef DATA_HH
#define DATA_HH

#include <iostream>
#include <fstream>
#include "Defs.hh"


class Data {
public:
  VF in; // Vector de entrada.
  VF out; // Vector de salida.

  // Constructor that reads the data.
  Data(std::fstream& file);
};


#endif
