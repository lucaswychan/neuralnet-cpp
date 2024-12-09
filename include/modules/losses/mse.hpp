#pragma once
#include "module.hpp"

namespace nn {
    class MSE {
        public:
            MSE();
            float forward(const Tensor<>& Y, const Tensor<>& Y_hat);
            Tensor<> backward();
        
        private:
            Tensor<> grad_output_;
            Tensor<> Y_cache_;
            Tensor<> Y_hat_cache_;
    };
}