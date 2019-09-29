#include "utils.hh"


void swap(float &a, float &b) {
  float c = a;
  a = b;
  b = c;
}


void print(VF v) {
  int n = v.size();
  for (int i = 0; i < n; ++i)
    cout << v[i] << '\t';
  cout << endl;
}


void print(VI v) {
  int n = v.size();
  for (int i = 0; i < n; ++i)
    cout << v[i] << '\t';
  cout << endl;
}


void print(VVF A) {
   int m = A.size();
   for (int i = 0; i < m; ++i)
      print(A[i]);
}


void print(VVVF T) {
   int p = T.size();
   for (int k = 0; k < p; ++k) {
      cout << "----------------" << k << "----------------" << endl;
      print(T[k]);
   }
}


void fillRandomly(VF &v) {
  int n = v.size();
  for (int i = 0; i < n; ++i)
    v[i] = rand() / (double)RAND_MAX;
}


void fillRandomly(VVF &A) {
   int m = A.size();
   int n = A[0].size();
   for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
	 A[i][j] = rand() / (double)RAND_MAX;
}
