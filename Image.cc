#include "Image.hh"


Image::Image(string type) {
  string fileName = trainFileName;
  if (type == "test")
    fileName = testFileName;

  ifstream file(fileName, ifstream::in);
  pixels = VF(size * size);
  file >> label;
  for (int i = 0; i < int(pixels.size()); ++i) {
    file >> pixels[i];
    pixels[i] /= 255;
  }
}
