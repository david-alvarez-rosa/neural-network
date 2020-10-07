#ifndef DATA_HH
#define DATA_HH


#include <iostream>
#include <fstream>
#include <vector>



template <typename T = float> class Data {
public:
  std::vector<T> in; // Input vector.
  std::vector<T> out; // Output vector.
  int label;

  // Constructor that reads the data.
  Data(std::ifstream& file);
};


// Define here your function that reads data.
template <typename T>
Data<T>::Data(std::ifstream& file) {
  int size = 28;
  in = std::vector<T>(size * size);
  file >> label;
  for (int i = 0; i < int(in.size()); ++i) {
    file >> in[i];
    in[i] /= 255; // Normalize.
  }
  out = std::vector<T>(10, 0);
  out[label] = 1;
}


#endif
