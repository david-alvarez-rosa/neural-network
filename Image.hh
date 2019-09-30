#ifndef IMAGE_HH
#define IMAGE_HH

#include <iostream>
#include <fstream>
#include <string>
#include "Defs.hh"


class Image {
public:
  VF pixels;
  int label;
  VF Y; // Convert label to output vector of 0's and 1's.

  // Reads next image of selected type.
  Image(ifstream& file);

private:
  int size = 28; // Size of the image (height and width).
};


#endif
