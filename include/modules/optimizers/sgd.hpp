#pragma once
#include "optimizer.hpp"

namespace nn {

class SGD : public Optimizer {
public:
    SGD() = default;

    // Constructor that takes a model and automatically extracts parameters
    SGD(const Module &model, float learning_rate = 0.01f, float momentum = 0.0f, float weight_decay = 0.0f);
    
    virtual void step() override;
    
private:
    float momentum_;
    float weight_decay_;
    unordered_map<string, Tensor<>> velocity_;
};

}  // namespace nn 