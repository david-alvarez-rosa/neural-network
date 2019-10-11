#include "Data.hh"


// Define here your function that reads data.
Data::Data(std::ifstream& file) {
  int size = 28;
  in = VF(size * size);
  int label;
  file >> label;
  for (int i = 0; i < int(in.size()); ++i) {
    file >> in[i];
    in[i] /= 255; // Normalize.
  }
  out = VF(10, 0);
  out[label] = 1;
}
