#include<iostream>

using namespace std;


uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

void read_mnist_cv(const char* image_filename, const char* label_filename){
    // Open files
    std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
    std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);

    // Read the magic and the meta data
    uint32_t magic;
    uint32_t num_items;
    uint32_t num_labels;
    uint32_t rows;
    uint32_t cols;

    image_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if(magic != 2051){
        cout<<"Incorrect image file magic: "<<magic<<endl;
        return;
    }

    label_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if(magic != 2049){
        cout<<"Incorrect image file magic: "<<magic<<endl;
        return;
    }

    image_file.read(reinterpret_cast<char*>(&num_items), 4);
    num_items = swap_endian(num_items);
    label_file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = swap_endian(num_labels);
    if(num_items != num_labels){
        cout<<"image file nums should equal to label num"<<endl;
        return;
    }

    image_file.read(reinterpret_cast<char*>(&rows), 4);
    rows = swap_endian(rows);
    image_file.read(reinterpret_cast<char*>(&cols), 4);
    cols = swap_endian(cols);

    cout<<"image and label num is: "<<num_items<<endl;
    cout<<"image rows: "<<rows<<", cols: "<<cols<<endl;

    char label;
    char* pixels = new char[rows * cols];

    for (int item_id = 0; item_id < num_items; ++item_id) {
        // read image pixel
        image_file.read(pixels, rows * cols);
        // read label
        label_file.read(&label, 1);

        string sLabel = std::to_string(int(label));
        cout<<"lable is: "<<sLabel<<endl;
        // convert it to cv Mat, and show it
        cv::Mat image_tmp(rows,cols,CV_8UC1,pixels);
        // resize bigger for showing
        cv::resize(image_tmp, image_tmp, cv::Size(100, 100));
        cv::imshow(sLabel, image_tmp);
        cv::waitKey(0);
    }

    delete[] pixels;
}


string base_dir = "/home/xy/caffe-master/data/mnist/";
string img_path = base_dir + "train-images-idx3-ubyte";
string label_path = base_dir + "train-labels-idx1-ubyte";

read_mnist_cv(img_path.c_str(), label_path.c_str());
