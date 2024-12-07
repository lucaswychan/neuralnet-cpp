#pragma once
#include "module.hpp"

namespace nn {
    class MSE {
        public:
            MSE();
            float forward(const vector<vector<float>>& Y, const vector<vector<float>>& Y_hat);
            vector<vector<float>> backward();
        
        private:
            vector<vector<float>> grad_output_;
            vector<vector<float>> Y_cache_;
            vector<vector<float>> Y_hat_cache_;
    };
}