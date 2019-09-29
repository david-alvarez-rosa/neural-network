#include "math.hh"


VF sum(VF v, VF w) {
   int n = v.size();
   for (int i = 0; i < n; ++i)
      v[i] += w[i];
   return v;
}


VF vectorDifference(VF v, VF w) {
   int n = v.size();
   for (int i = 0; i < n; ++i)
      v[i] -= w[i];
   return v;
}


int vectorMaxPos(VI v) {
   int n = v.size();
   int maxValue = v[0];
   int maxPos = 0;
   for (int i = 1; i < n; ++i)
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
   int maxIter = 16;
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


double exp(double x) {
  const double M_E = 2.7182818284590452353;
   if (x >= 0) {
      int n = x;
      return expInterval(x - n)*pow(M_E, n);
   }
   return 1/exp(-x);
}


double euclidianNormSquared(VF v) {
   double norm = 0;
   int n = v.size();
   for (int i = 0; i < n; ++i)
      norm += v[i]*v[i];
   return norm;
}


float euclidianDistanceSquared(VF v, VF w) {
   return euclidianNormSquared(vectorDifference(v, w));
}


void applySoftmax(VF &v) {
   int n = v.size();
   double den = 0;
   for (int i = 0; i < n; ++i) {
      v[i] = exp(v[i]);
      den += v[i];
   }
   for (int i = 0; i < n; ++i)
      v[i] /= den;
}


double sigmoid(double x) {
   return 1/(1 + exp(-x));
}


VF applySigmoid(VF &v) {
   int n = v.size();
   for (int i = 0; i < n; ++i)
      v[i] = sigmoid(v[i]);
   return v;
}
