#pragma once
#include <fstream>
#include <vector>
#include <string>
#include <stdint.h>
#include <iostream>
#include <cmath>
#include <tuple>
#include "tensor.hpp"
using namespace std;

struct Batch {
    vector<vector<double>> batchData;
    vector<vector<int>> batchLabels;

    tuple<Tensor<>, Tensor<>> toTensor();
};

class MNIST {
private:
    const double MNIST_MEAN = 0.1307f;
    const double MNIST_STD = 0.3081f;
    const int MNIST_NUM_LABELS = 10;

    vector<vector<double>> images;
    vector<int> labels;

    size_t currentBatchIndex = 0;
    size_t batchSize;
    size_t numBatches;

    template<typename T>
    T reverseInt(T value);
    double normalize(double value);

    bool readImages(const string& path);
    bool readLabels(const string& path);

    vector<int> oneHotEncoding(const int& labels);

public:

    MNIST(size_t batchSize = 64) : batchSize(batchSize), currentBatchIndex(0) {}

    bool loadData(const string& imageFile, const string& labelFile);

    // Get next batch
    Batch getNextBatch();

    // Reset batch counter
    inline void reset() {
        this->currentBatchIndex = 0;
    }

    // Get number of batches
    inline size_t getNumBatches() const {
        return this->numBatches;
    }

    // Get current batch index
    inline size_t getCurrentBatchIndex() const {
        return this->currentBatchIndex;
    }
};