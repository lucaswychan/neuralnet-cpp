#include "optimizer.hpp"
using namespace nn;

// Constructor that extracts parameters from model
Optimizer::Optimizer(const Module &model, float learning_rate) : learning_rate_(learning_rate)
{
    // Get all parameters from the model
    model.register_parameters(this->params_, this->grads_);
}

void Optimizer::zero_grad()
{
    // Reset gradients to zero
    for (auto &[name, grad] : this->grads_)
    {
        if (grad == nullptr) {
            cout << "Warning: Null gradient pointer for parameter " << name << endl;
            continue;
        }
        
        // Create a zero tensor with the same shape and assign it
        *grad = Tensor<>(grad->shapes(), 0.0f);
    }
}