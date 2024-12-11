#include "mnist.hpp"

bool MNIST::loadData(const string& imageFile, const string& labelFile) {
    if (!readImages(imageFile) || !readLabels(labelFile)) {
        return false;
    }

    // Calculate number of batches (ceiling division)
    this->numBatches = (this->images.size() + this->batchSize - 1) / this->batchSize;
    
    cout << "Dataset loaded successfully:\n";
    cout << "Total images: " << images.size() << "\n";
    cout << "Batch size: " << batchSize << "\n";
    cout << "Number of batches: " << numBatches << "\n";
    
    return true;
}


Batch MNIST::getNextBatch() {
    Batch batch;
    size_t startIdx = currentBatchIndex * batchSize;
    size_t endIdx = std::min(startIdx + batchSize, this->images.size());
    size_t actualBatchSize = endIdx - startIdx;

    batch.batchData.resize(actualBatchSize);
    batch.batchLabels.resize(actualBatchSize);

    // Copy data for this batch
    for (size_t i = 0; i < actualBatchSize; i++) {
        batch.batchData[i] = this->images[startIdx + i];
        batch.batchLabels[i] = this->oneHotEncoding(this->labels[startIdx + i]);
    }

    // Update batch index
    this->currentBatchIndex = (this->currentBatchIndex + 1) % this->numBatches;

    return batch;
}

template<typename T>
T MNIST::reverseInt(T value) {
    T result = 0;
    for(size_t i = 0; i < sizeof(T); i++) {
        result = (result << 8) | ((value >> (i * 8)) & 0xFF);
    }
    return result;
}

double MNIST::normalize(double value) {
    double scaled = value / 255.0f;
    return (scaled - this->MNIST_MEAN) / this->MNIST_STD;
}

bool MNIST::readImages(const string& path) {
    ifstream file(path, ios::binary);
    if(!file.is_open()) {
        cerr << "Failed to open file: " << path << endl;
        return false;
    }

    int32_t magic, numImages, numRows, numCols;

    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
    file.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
    file.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));

    magic = reverseInt(magic);
    numImages = reverseInt(numImages);
    numRows = reverseInt(numRows);
    numCols = reverseInt(numCols);

    this->images.resize(numImages, vector<double>(numRows * numCols));

    for(int i = 0; i < numImages; i++) {
        for(int j = 0; j < numRows * numCols; j++) {
            unsigned char temp = 0;
            file.read(reinterpret_cast<char*>(&temp), sizeof(temp));
            this->images[i][j] = normalize(static_cast<double>(temp));
        }
    }

    return true;
}

bool MNIST::readLabels(const string& path) {
    ifstream file(path, ios::binary);
    if(!file.is_open()) {
        cerr << "Failed to open file: " << path << endl;
        return false;
    }

    int32_t magic, numLabels;

    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));

    magic = this->reverseInt(magic);
    numLabels = this->reverseInt(numLabels);

    labels.resize(numLabels);

    for(int i = 0; i < numLabels; i++) {
        unsigned char temp;
        file.read(reinterpret_cast<char*>(&temp), 1);
        this->labels[i] = static_cast<int>(temp);
    }

    return true;
}

vector<int> MNIST::oneHotEncoding(const int& labels) {
    vector<int> oneHotVector(this->MNIST_NUM_LABELS, 0);
    oneHotVector[labels] = 1;
    return oneHotVector;
}

tuple<Tensor<>, Tensor<>> Batch::toTensor() {
    Tensor<> data = this->batchData;
    
    Tensor<int> labels_int = this->batchLabels;
    Tensor<> labels = labels_int.dtype<>();
    
    return make_tuple(data, labels);
}