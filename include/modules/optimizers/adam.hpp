#pragma once
#include "optimizer.hpp"
#include <cmath>

namespace nn {

class Adam : public Optimizer {
public:
    Adam() = default;
    
    // Constructor that takes a model and automatically extracts parameters
    Adam(const Module &model, float learning_rate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);
    
    // Override to initialize moment buffers
    // virtual void register_parameter(const string& name, Tensor<>& param, Tensor<>& grad) override;
    
    virtual void step() override;
    
private:
    float beta1_;
    float beta2_;
    float epsilon_;
    int t_;  // Timestep counter
    unordered_map<string, Tensor<>> m_;  // First moment (Momentum)
    unordered_map<string, Tensor<>> v_;  // Second moment (RMSprop)
};

}  // namespace nn 