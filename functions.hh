#include <vector>

using namespace std;

using VF = vector<float>;
using VVF = vector<VF>;
using VVVF = vector<VVF>;
using VI = vector<int>;


const double M_E = 2.7182818284590452353;


// Swaps two elements.
void swap(float &a, float &b) {
   float c = a;
   a = b;
   b = c;
}


// Print a vector.
void print(VF v) {
   int n = v.size();
   for (int i = 0; i < n; ++i)
      cout << v[i] << '\t';
   cout << endl;
}


// Print a vector.
void print(VI v) {
   int n = v.size();
   for (int i = 0; i < n; ++i)
      cout << v[i] << '\t';
   cout << endl;
}


// Fill randomnly a vector (entries between 0 and 1).
void fillRandomly(VF &v) {
   int n = v.size();
   for (int i = 0; i < n; ++i)
      v[i] = rand() / (double)RAND_MAX;
}


// Given two vectors v and w, computes v + w.
VF sum(VF v, VF w) {
   int n = v.size();
   for (int i = 0; i < n; ++i)
      v[i] += w[i];
   return v;
}


// Given two vectors v and w, computes v - w.
VF vectorDifference(VF v, VF w) {
   int n = v.size();
   for (int i = 0; i < n; ++i)
      v[i] -= w[i];
   return v;
}


// Given a vector returns position of the (first) maxim.
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


// Print a matrix.
void print(VVF A) {
   int m = A.size();
   for (int i = 0; i < m; ++i)
      print(A[i]);
}


// Multiplies matrix and vector.
VF multiply(VVF A, VF v) {
   int m = A.size();
   int n = A[0].size();
   VF Av(m, 0);
   for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
	 Av[i] += A[i][j]*v[j];
   return Av;
}


// Fill randomnly a matrix (entries between 0 and 1).
void fillRandomly(VVF &A) {
   int m = A.size();
   int n = A[0].size();
   for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
	 A[i][j] = rand() / (double)RAND_MAX;
}


// Prints tensor.
void print(VVVF T) {
   int p = T.size();
   for (int k = 0; k < p; ++k) {
      cout << "----------------" << k << "----------------" << endl;
      print(T[k]);
   }
}


// Computes a^n, with n a non-negative integer.
double pow(double a, int n) {
   if (n == 0)
      return 1;
   if (n%2 == 0) {
      double aux = pow(a, n/2);
      return aux*aux;
   }
   return a*pow(a, n - 1);
}


// Computes exp(x) for small x (between 0 and 1).
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


// Computes exp(x).
double exp(double x) {
   if (x >= 0) {
      int n = x;
      return expInterval(x - n)*pow(M_E, n);
   }
   return 1/exp(-x);
}


// Computes the square of the Euclidian norm of a vector.
double euclidianNormSquared(VF v) {
   double norm = 0;
   int n = v.size();
   for (int i = 0; i < n; ++i)
      norm += v[i]*v[i];
   return norm;
}


// Computes the Euclidian Distance between two vectors.
float euclidianDistanceSquared(VF v, VF w) {
   return euclidianNormSquared(vectorDifference(v, w));
}


// Softmax function (inplace).
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


// Sigmoid function.
double sigmoid(double x) {
   return 1/(1 + exp(-x));
}


// Sigmoid function for a vector (inplace).
VF applySigmoid(VF &v) {
   int n = v.size();
   for (int i = 0; i < n; ++i)
      v[i] = sigmoid(v[i]);
   return v;
}
