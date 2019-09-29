#ifndef IMAGE_HH
#define IMAGE_HH

#include <iostream>
#include <fstream>
#include <string>
#include "defs.hh"


class Image {
public:
  VF pixels;
  int label;

  // Reads next image of selected type.
  Image(ifstream& file);

private:
  int size = 28; // Size of the image (height and width).
};


#endif
