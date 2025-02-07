#pragma once
#include "module.hpp"

namespace nn {
class Softmax : public Module {
    private:
        Tensor<> softmax_input_cache_;

        // Helper function to deal with multiple dimensions
        Tensor<> softmax_helper(const Tensor<>& input);
        vector<double> softmax_helper(const vector<double>& input);
    public:
        Softmax();
        Tensor<> forward(const Tensor<>& input);
        Tensor<> backward(const Tensor<>& grad_output);
        const Tensor<>& get_softmax_input_cache() const { return this->softmax_input_cache_; }
    };
}