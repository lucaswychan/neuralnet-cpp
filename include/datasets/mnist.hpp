#pragma once
#include <fstream>
#include <vector>
#include <string>
#include <stdint.h>
#include <iostream>
#include "tensor.hpp"
using namespace std;

class MNIST {
    private:
        const double MNIST_MEAN = 0.1307f;
        const double MNIST_STD = 0.3081f;

        // Function to reverse byte order (MNIST is big-endian)
        template<typename T>
        T reverseInt(T value) {
            T result = 0;
            for(size_t i = 0; i < sizeof(T); i++) {
                result = (result << 8) | ((value >> (i * 8)) & 0xFF);
            }
            return result;
        }

        // Normalize a single value using mean and std
        double normalize(double value) {
            // First scale to [0,1], then apply normalization
            double scaled = value / 255.0f;
            return (scaled - MNIST_MEAN) / MNIST_STD;
        }

    public:
        Tensor<> readImages(const string& path);
        Tensor<int> readLabels(const string& path);

        vector<Tensor<>> sampleBatchImages(const Tensor<>& images, const size_t batch_size);
        vector<Tensor<int>> sampleBatchLabels(const Tensor<int>& labels, const size_t batch_size);
};