#pragma once
#include "optimizer.hpp"

namespace nn {

class SGD : public Optimizer {
public:
    SGD() = default;

    // Constructor that takes a model and automatically extracts parameters
    SGD(const Module &model, float learning_rate = 0.01f, float momentum = 0.0f, float weight_decay = 0.0f);
    
    // Default constructor for manual parameter registration
    SGD(float learning_rate = 0.01f, float momentum = 0.0f, float weight_decay = 0.0f);
    
    // Override to initialize momentum buffer
    // virtual void register_parameter(const string& name, Tensor<>& param, Tensor<>& grad) override;
    
    virtual void step() override;
    
private:
    float momentum_;
    float weight_decay_;
    unordered_map<string, Tensor<>> velocity_;
};

}  // namespace nn 