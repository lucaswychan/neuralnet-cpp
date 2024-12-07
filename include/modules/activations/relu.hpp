#pragma once
#include "module.hpp"

namespace nn {
class ReLU : public Module {
    public:
        ReLU();
        virtual vector<vector<float>> forward(const vector<vector<float>>& input) override;
        virtual vector<vector<float>> backward(const vector<vector<float>>& grad_output) override;
    };
}