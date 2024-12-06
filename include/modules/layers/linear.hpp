#pragma once
#include <stdio.h>
#include <iostream>
#include "module.hpp"
using namespace std;

namespace nn {

class Linear : Module{
public:
    Linear(int in_features, int out_features, bool bias);
    
    vector<vector<float>> forward(const vector<vector<float>>& input) override;
    vector<vector<float>> backward(const vector<vector<float>>& grad_output) override;
    void update_params(const float lr) override;

    void setWeights(const vector<vector<float>>& desiredWeights) {
        this->weights_ = desiredWeights;
    };
    void setBiases(const vector<vector<float>>& desiredBiases) {
        this->biases_ = desiredBiases;
    }

    const vector<vector<float>>& getWeights() { return this->weights_; }
    const vector<vector<float>>& getBiases() { return this->biases_; }

private:
    int in_features_;
    int out_features_;
    bool bias_;
    vector<vector<float>> weights_;
    vector<vector<float>> biases_;
};  

}