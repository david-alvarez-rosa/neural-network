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


double dotProduct(VF v, VF w) {
  double sol = 0;
  for (int i = 0; i < int(v.size()); ++i)
    sol += v[i] * w[i];
  return sol;
}


int vectorMaxPos(VF v) {
  double maxValue = v[0];
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


double pow(double a, int n) {
  if (n == 0)
    return 1;
  if (n%2 == 0) {
    double aux = pow(a, n/2);
    return aux*aux;
  }
  return a*pow(a, n - 1);
}


double expInterval(double x) {
  int maxIter = 18;
  double sol = 0;
  int faci = 1;
  double powx = 1;
  for (int i = 0; i < maxIter; ++i) {
    sol += powx/(faci);
    powx *= x;
    faci *= (i + 1);
  }
  return sol;
}


double exponential(double x) {
  if (x > 20 or x < -20)
    std::cout << "Se está aplicando la exponencial a un valor x, |x| > 20." << std::endl;

  const double M_E = 2.7182818284590452353;
  if (x >= 0) {
    int n = x;
    return expInterval(x - n)*pow(M_E, n);
  }
  return 1/exponential(-x);
}


double logarithm(double x) {
  double solRight = 0;
  double solLeft = 0;
  while (exponential(solRight) < x)
    solRight += 10;
  while (exponential(solLeft) > x)
    solLeft -= 10;

  while (solRight - solLeft > 1e-8) {
    double solMed = (solRight + solLeft) / 2;
    if (exponential(solMed) > x)
      solRight = solMed;
    else
      solLeft = solMed;
  }

  return (solRight + solLeft) / 2;
}
