#pragma once
#include <iostream>
#include <stdio.h>
#include <vector>
#include "utils/matrix_utils.hpp"
using namespace std;


namespace nn {

class Module {
    public:
        virtual ~Module() = default;
        virtual vector<vector<float>> forward(const vector<vector<float>>& input) = 0;
        virtual vector<vector<float>> backward(const vector<vector<float>>& grad_output) = 0;
        virtual void update_params(const float lr) { return; };

        vector<vector<float>> operator()(const vector<vector<float>>& input) {
            return this->forward(input);
        }
    
    protected:
        vector<vector<float>> input_cache_;
};

}