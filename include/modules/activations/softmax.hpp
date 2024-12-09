#pragma once
#include "module.hpp"

namespace nn {
class Softmax : public Module {
    private:
        // Helper function to deal with multiple dimensions
        Tensor<> softmax_helper(const Tensor<>& input);
        vector<double> softmax_helper(const vector<double>& input);
    public:
        Tensor<> forward(const Tensor<>& input);
        Tensor<> backward(const Tensor<>& grad_output);
    };
}