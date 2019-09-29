
int swapEndian(int val) {
   val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
   return (val << 16) | (val >> 16);
}


VC readLabels() {
   // Abrir archivo.
   string labelFileName = "train-labels-idx1-ubyte";
   ifstream labelFile(labelFileName, ifstream::in | ios::binary);

   // Leer el número de elementos.
   int magic;
   labelFile.read(reinterpret_cast<char*>(&magic), 4);
   magic = swapEndian(magic);

   cout << magic << endl;

   int numLabels = 60000;
   uint32_t numLabelsAux;
   labelFile.read(reinterpret_cast<char*>(&numLabelsAux), 4);
   // int numLabels = numLabelsAux;
   // numLabels = swapEndian(numLabels);

   cout << numLabels << endl;

   cout << numLabelsAux << endl;


   // Leer archivo.
   VC labels(numLabels);
   for (int i = 0; i < numLabels; ++i)
      labelFile.read(&labels[i], 1);

   return labels;
}


VVC readImages() {
   string imageFileName = "train-images-idx3-ubyte";
   ifstream imageFile(imageFileName, ifstream::in | ios::binary);

   // Leer el número mágico, número de elementos, filas y columnas.
   int magic;
   imageFile.read(reinterpret_cast<char*>(&magic), 4);
   magic = swapEndian(magic);

   int numImages;
   imageFile.read(reinterpret_cast<char*>(&numImages), 4);
   numImages = swapEndian(numImages);

   int rows;
   imageFile.read(reinterpret_cast<char*>(&rows), 4);
   rows = swapEndian(rows);
   int cols;
   imageFile.read(reinterpret_cast<char*>(&cols), 4);
   cols = swapEndian(cols);

   // Leer las imágenes.
   VVC images;

   numImages = 5;

   for (int i = 0; i < numImages; ++i) {
      // VC pixels(rows * cols);
      char* pixels = new char[rows * cols];

      // read image pixel
      imageFile.read(pixels, rows * cols);

      // images[i] = pixels;
   }

   return images;
}
