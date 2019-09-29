#include "utils.hh"


void swap(float &a, float &b) {
  float c = a;
  a = b;
  b = c;
}


void print(VF v) {
  for (int i = 0; i < int(v.size()); ++i)
    cout << v[i] << '\t';
  cout << endl;
}


void print(VI v) {
  for (int i = 0; i < int(v.size()); ++i)
    cout << v[i] << '\t';
  cout << endl;
}


void print(VVF A) {
  for (int i = 0; i < int(A.size()); ++i)
    print(A[i]);
}


void print(VVVF T) {
  for (int k = 0; k < int(T.size()); ++k) {
    cout << "----------------" << k << "----------------" << endl;
    print(T[k]);
  }
}


void fillRandomly(VF &v) {
  for (int i = 0; i < int(v.size()); ++i)
    v[i] = 0.018 * rand() / float(RAND_MAX);
}


void fillRandomly(VVF &A) {
  for (int i = 0; i < int(A.size()); ++i)
    for (int j = 0; j < int(A[0].size()); ++j)
      A[i][j] = 0.018 * rand() / float(RAND_MAX);
}
