#pragma once
#include <iostream>
#include <stdio.h>
#include <vector>
#include "tensor.hpp"
using namespace std;


namespace nn {

class Module {
    public:
        /**
         * Virtual destructor for the Module class.
         */
        virtual ~Module() = default;

        /**
         * Pure virtual function for forward pass computation.
         * @param input The input data as a 2D Tensor.
         * @return The output of the forward pass as a 2D Tensor.
         */
        virtual Tensor<> forward(const Tensor<>& input) = 0;

        /**
         * Pure virtual function for backward pass computation.
         * @param grad_output The gradient of the loss with respect to the output.
         * @return The gradient of the loss with respect to the input.
         */
        virtual Tensor<> backward(const Tensor<>& grad_output) = 0;

        /**
         * Virtual function to update the parameters of the module.
         * @param lr The learning rate for the update.
         */
        virtual void update_params(const float lr) { return; };

        /**
         * Operator overload to enable calling the module like a function.
         * @param input The input data as a 2D Tensor.
         * @return The output of the forward pass as a 2D Tensor.
         */
        Tensor<> operator()(const Tensor<>& input) {
            return this->forward(input);
        }
    
    protected:
        /**
         * Cached input data for use in backward pass computations.
         */
        Tensor<> input_cache_;
};

}