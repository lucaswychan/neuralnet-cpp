#include "modules/layers/linear.hpp"

Linear::Linear(int in_features, int out_features) {
    this->in_features = in_features;
    this->out_features = out_features;
    this->weights = new float*[in_features];
    for (int i = 0; i < in_features; i++) {
        this->weights[i] = new float[out_features];
    }
    
    this->biases = new float[out_features];

    this->input_cache_ = nullptr;
}