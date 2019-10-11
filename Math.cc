#include "Math.hh"


VF sum(VF v, VF w) {
  for (int i = 0; i < int(v.size()); ++i)
    v[i] += w[i];
  return v;
}


VF difference(VF v, VF w) {
  for (int i = 0; i < int(v.size()); ++i)
    v[i] -= w[i];
  return v;
}


float dotProduct(VF v, VF w) {
  float sol = 0;
  for (int i = 0; i < int(v.size()); ++i)
    sol += v[i] * w[i];
  return sol;
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
  VF Av(A.size(), 0);
  for (int i = 0; i < int(A.size()); ++i)
    for (int j = 0; j < int(A[0].size()); ++j)
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
  if (x > 20 or x < -20)
    std::cout << "Se está aplicando la exponencial a un valor x, |x| > 20." << std::endl;

  const float M_E = 2.7182818284590452353;
  if (x >= 0) {
    int n = x;
    return expInterval(x - n)*pow(M_E, n);
  }
  return 1/exp(-x);
}


float log(float x) {
  float solRight = 0;
  float solLeft = 0;
  while (exp(solRight) < x)
    solRight += 10;
  while (exp(solLeft) > x)
    solLeft -= 10;

  while (solRight - solLeft > 1e-6) {
    float solMed = (solRight + solLeft) / 2;
    if (exp(solMed) > x)
      solRight = solMed;
    else
      solLeft = solMed;
  }

  return (solRight + solLeft) / 2;
}
