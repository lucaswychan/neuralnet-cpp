#pragma once
#include <unordered_map>
#include <string>
#include "tensor.hpp"
#include "module.hpp"

namespace nn
{

    class Optimizer
    {
    public:
        // Constructor that takes a model and extracts parameters automatically
        Optimizer(const Module &model, float learning_rate = 0.01f);
        
        // Default constructor (for manual parameter registration)
        Optimizer();
        
        virtual ~Optimizer() = default;

        // Pure virtual method for updating parameters
        virtual void step() = 0;

        // Reset gradients to zero
        void zero_grad();

    protected:
        float learning_rate_ = 0.01f;
        unordered_map<string, Tensor<> *> params_;
        unordered_map<string, Tensor<> *> grads_;
    };

} // namespace nn