#ifndef CUSTOM_HH
#define CUSTOM_HH


#include "Utils.hh"


// // Define here your custom activation function.
// template <typename T>
// T activationFunction(T x) {

//   return 0;
// }


// template <typename T>
// T activationDerivative(T x) {
//   if (x >= 0)
//     return 1;
//   return 0;
// }




// Define here your loss function (for comparing two real numbers).
template <typename T>
T lossFunction(T y, T yp) {
  // This is for euclidian distance.
  return (yp - y) * (yp - y);
  // This is a cross-entropy loss function modified.
  if (y == 0)
    return -std::log(1 - yp);
  else
    return -std::log(yp);
  // This is the cross-entropy loss.
  return -y*std::log(yp);
}


// Define here the derivative respect yp of the loss funcion.
template <typename T>
T lossDerivative(T y, T yp) {
  // This is for the euclidian distance.
  return 2*(yp - y);
  // This is a cross-entropy erorr function modified.
  if (y == 0)
    return -1/(1 - yp);
  else
    return -1/yp;
  // This is for the cross-entropy loss.
  return -y/yp;
}


#endif
