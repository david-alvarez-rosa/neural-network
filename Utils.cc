#include "Utils.hh"


void swap(double &a, double &b) {
  double c = a;
  a = b;
  b = c;
}


void print(VF v) {
  for (int i = 0; i < int(v.size()); ++i)
    std::cout << v[i] << '\t';
  std::cout << std::endl;
}


void print(std::vector<int> v) {
  for (int i = 0; i < int(v.size()); ++i)
    std::cout << v[i] << '\t';
  std::cout << std::endl;
}


void print(VVF A) {
  for (int i = 0; i < int(A.size()); ++i)
    print(A[i]);
}


void print(VVVF T) {
  for (int k = 0; k < int(T.size()); ++k) {
    std::cout << "----------------" << k << "----------------" << std::endl;
    print(T[k]);
  }
}


void fillRandomly(VF &v) {
  for (int i = 0; i < int(v.size()); ++i)
    v[i] = 0.001 * (rand() / double(RAND_MAX) - 0.5);
}


void fillRandomly(VVF &A) {
  for (int i = 0; i < int(A.size()); ++i)
    fillRandomly(A[i]);
}


double relativeError(double x, double y) {
  return std::abs((x - y)/(std::max(std::min(x, y), 1e-3)));
}
