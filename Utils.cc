#include "Utils.hh"


void swap(float &a, float &b) {
  float c = a;
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
    v[i] = 0.02 * (rand() / float(RAND_MAX));
    // v[1] = 1;
  // for (int i = 0; i < int(v.size()); ++i) {
  //   if (i%2 == 0)
  //     v[i] = -1;
  //   else
  //     v[i]= 1;
  // }
}


void fillRandomly(VVF &A) {
  for (int i = 0; i < int(A.size()); ++i)
    fillRandomly(A[i]);
}
