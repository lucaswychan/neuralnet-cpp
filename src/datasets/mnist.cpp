#include "mnist.hpp"

Tensor<> MNIST::readImages(const string& path) {
    std::ifstream file(path, std::ios::binary);

    if(!file.is_open()) {
        throw runtime_error("Failed to open file: " + path);
    }

    int32_t magic;
    int32_t numImages;
    int32_t numRows;
    int32_t numCols;

    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
    file.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
    file.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));

    magic = reverseInt(magic);
    numImages = reverseInt(numImages);
    numRows = reverseInt(numRows);
    numCols = reverseInt(numCols);

    vector<vector<double>> images;
    images.resize(numImages, vector<double>(numRows * numCols));

    // Read and normalize image data
    for(int i = 0; i < numImages; i++) {
        for(int j = 0; j < numRows * numCols; j++) {
            unsigned char temp = 0;
            file.read(reinterpret_cast<char*>(&temp), sizeof(temp));
            images[i][j] = normalize(static_cast<double>(temp));
        }
    }

    return Tensor<>(images);
}

Tensor<int> MNIST::readLabels(const string& path) {
    std::ifstream file(path, std::ios::binary);
    
    if(!file.is_open()) {
        throw runtime_error("Failed to open file: " + path);
    }

    int32_t magic;
    int32_t numLabels;

    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));

    magic = reverseInt(magic);
    numLabels = reverseInt(numLabels);

    vector<int> labels;
    labels.resize(numLabels);

    for(int i = 0; i < numLabels; i++) {
        unsigned char temp;
        file.read(reinterpret_cast<char*>(&temp), 1);
        labels[i] = static_cast<int>(temp); // Store as integer without normalization
    }

    return Tensor<int>(labels);
}

vector<Tensor<>> MNIST::sampleBatchImages(const Tensor<>& images, const size_t batch_size) {
    const size_t num_samples = images.shapes()[0];
    const size_t num_batch = num_samples / batch_size;

    vector<Tensor<>> batch_images(num_batch, Tensor<>({batch_size, 784}, 0.0f));
    
    for(size_t i = 0; i < num_batch; i++) {
        for(size_t j = 0; j < batch_size; j++) {
            for(size_t k = 0; k < 784; k++) {
                batch_images[i][j, k] = images[i * batch_size + j, k];
            }
        }
    }

    return batch_images;
}

vector<Tensor<int>> MNIST::sampleBatchLabels(const Tensor<>& labels, const size_t)