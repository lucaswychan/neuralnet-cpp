#include "sgd.hpp"
using namespace nn;

// Constructor with model
SGD::SGD(const Module &model, float learning_rate, float momentum, float weight_decay)
    : Optimizer(model, learning_rate), momentum_(momentum), weight_decay_(weight_decay)
{
    // Initialize velocity for momentum if needed
    if (momentum_ > 0.0f) {
        for (auto &[name, param] : this->params_)
        {
            this->velocity_[name] = Tensor<>(param->shapes(), 0.0f);
        }
    }
}

void SGD::step() {
    for (auto& [name, param] : params_) {
        Tensor<>& grad = *grads_[name];
        
        // Apply weight decay
        if (weight_decay_ > 0.0f) {
            grad = grad + (*param) * weight_decay_;
        }
        
        // Apply momentum if needed
        if (momentum_ > 0.0f) {
            Tensor<>& v = velocity_[name];
            v = v * momentum_ + grad * (1.0f - momentum_);
            *param = *param - v * learning_rate_;
        } else {
            *param = *param - grad * learning_rate_;
        }
    }
}