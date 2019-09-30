#include "Image.hh"


Image::Image(ifstream& file) {
  pixels = VF(size * size);
  file >> label;
  for (int i = 0; i < int(pixels.size()); ++i) {
    file >> pixels[i];
    pixels[i] /= 255;
  }
  Y = VF(10, 0);
  Y[label] = 1;
}
