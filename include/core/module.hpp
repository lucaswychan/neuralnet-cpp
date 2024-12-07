#pragma once
#include <iostream>
#include <stdio.h>
#include <vector>
#include "utils/matrix_utils.hpp"
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
         * @param input The input data as a 2D vector.
         * @return The output of the forward pass as a 2D vector.
         */
        virtual vector<vector<float>> forward(const vector<vector<float>>& input) = 0;

        /**
         * Pure virtual function for backward pass computation.
         * @param grad_output The gradient of the loss with respect to the output.
         * @return The gradient of the loss with respect to the input.
         */
        virtual vector<vector<float>> backward(const vector<vector<float>>& grad_output) = 0;

        /**
         * Virtual function to update the parameters of the module.
         * @param lr The learning rate for the update.
         */
        virtual void update_params(const float lr) { return; };

        /**
         * Operator overload to enable calling the module like a function.
         * @param input The input data as a 2D vector.
         * @return The output of the forward pass as a 2D vector.
         */
        vector<vector<float>> operator()(const vector<vector<float>>& input) {
            return this->forward(input);
        }
    
    protected:
        /**
         * Cached input data for use in backward pass computations.
         */
        vector<vector<float>> input_cache_;
};

}