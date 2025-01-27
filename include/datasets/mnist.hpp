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
    vector<vector<double>> batch_data;
    vector<int> batch_labels;

    tuple<Tensor<>, Tensor<>> to_tensor();
};

class MNIST {
private:
    const double MNIST_MEAN = 0.1307f;
    const double MNIST_STD = 0.3081f;
    const int MNIST_NUM_LABELS = 10;

    vector<vector<double>> images;
    vector<int> labels;

    size_t current_batch_idxs = 0;
    size_t batch_size;
    size_t num_batches;

    bool verbose = true;

    template<typename T>
    T reverse_int(T value);
    double normalize(double value);

    bool read_images(const string& path);
    bool read_labels(const string& path);

public:

    MNIST(size_t batch_size = 64, bool verbose = true) : batch_size(batch_size), verbose(verbose), current_batch_idxs(0) {}

    bool load_data(const string& image_file, const string& label_file);

    // Get next batch
    Batch get_next_batch();

    // Reset batch counter
    inline void reset() {
        this->current_batch_idxs = 0;
    }

    // Get number of batches
    inline size_t get_num_batches() const {
        return this->num_batches;
    }

    // Get current batch index
    inline size_t get_current_batch_idxs() const {
        return this->current_batch_idxs;
    }
};