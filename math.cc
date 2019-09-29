#include "math.hh"


VF sum(VF v, VF w) {
  int n = v.size();
  for (int i = 0; i < n; ++i)
    v[i] += w[i];
  return v;
}


VF difference(VF v, VF w) {
  int n = v.size();
  for (int i = 0; i < n; ++i)
    v[i] -= w[i];
  return v;
}

int vectorMaxPos(VF v) {
  float maxValue = v[0];
  int maxPos = 0;
  for (int i = 1; i < int(v.size()); ++i)
    if (v[i] > maxValue) {
      maxValue = v[i];
      maxPos = i;
    }
  return maxPos;
}


VF multiply(VVF A, VF v) {
  int m = A.size();
  int n = A[0].size();
  VF Av(m, 0);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      Av[i] += A[i][j]*v[j];
  return Av;
}


float pow(float a, int n) {
  if (n == 0)
    return 1;
  if (n%2 == 0) {
    float aux = pow(a, n/2);
    return aux*aux;
  }
  return a*pow(a, n - 1);
}


float expInterval(float x) {
  int maxIter = 16;
  float sol = 0;
  int faci = 1;
  float powx = 1;
  for (int i = 0; i < maxIter; ++i) {
    sol += powx/(faci);
    powx *= x;
    faci *= (i + 1);
  }
  return sol;
}


float exp(float x) {
  const float M_E = 2.7182818284590452353;
  if (x >= 0) {
    int n = x;
    return expInterval(x - n)*pow(M_E, n);
  }
  return 1/exp(-x);
}


void applySoftmax(VF &v) {
  int n = v.size();
  float den = 0;
  for (int i = 0; i < n; ++i) {
    v[i] = exp(v[i]);
    den += v[i];
   }
   for (int i = 0; i < n; ++i)
      v[i] /= den;
}
