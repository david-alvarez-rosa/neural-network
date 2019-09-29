#ifndef IMAGE_HH
#define IMAGE_HH

#include "defs.hh"
#include <fstream>
#include <string>


class Image {
public:
  VF pixels;
  float label;

  // Reads next image of selected type.
  Image(string type);

private:
  int size = 28; // Size of the image (height and width).
  string trainFileName = "Data/labelImagesTraining.dat";
  string testFileName = "Data/labelImagesTest.dat";
};


#endif
