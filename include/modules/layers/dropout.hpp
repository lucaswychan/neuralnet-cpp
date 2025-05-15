#pragma once
#include "module.hpp"
#include <random>

namespace nn {
    class Dropout : public Module {
        public:
            Dropout(float p = 0.5);
            virtual Tensor<> forward(const Tensor<>& input) override;
            virtual Tensor<> backward(const Tensor<>& grad_output) override;
        private:
            float p_;
            float scale_;
            Tensor<> mask_cache_;
            // probability distribution of the dropout
            bernoulli_distribution pmf_;
            mt19937 gen_;
    };
}