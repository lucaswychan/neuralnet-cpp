#pragma once
#include <iostream>
#include <stdio.h>
#include <vector>
#include <unordered_map>
#include "tensor.hpp"
using namespace std;

namespace nn
{

    class Module
    {
    public:
        Module() = default;
        /**
         * Virtual destructor for the Module class.
         */
        virtual ~Module() = default;

        /**
         * Pure virtual function for forward pass computation.
         * @param input The input data as a 2D Tensor.
         * @return The output of the forward pass as a 2D Tensor.
         */
        virtual Tensor<> forward(const Tensor<> &input) = 0;

        /**
         * Pure virtual function for backward pass computation.
         * @param grad_output The gradient of the loss with respect to the output.
         * @return The gradient of the loss with respect to the input.
         */
        virtual Tensor<> backward(const Tensor<> &grad_output) = 0;

        /**
         * Operator overload to enable calling the module like a function.
         * @param input The input data as a 2D Tensor.
         * @return The output of the forward pass as a 2D Tensor.
         */
        Tensor<> operator()(const Tensor<> &input)
        {
            return this->forward(input);
        }

        /**
         * Sets the module in training mode (train=true) or evaluation mode (train=false).
         * This affects certain modules like Dropout and BatchNorm whose behavior differs
         * between training and evaluation.
         *
         * @param mode If true, sets the module to training mode, otherwise to evaluation mode.
         * @return A reference to this module for method chaining.
         */
        virtual Module &train(const bool mode = true)
        {
            this->training = mode;
            apply_to_children([mode](Module &child)
                              { child.train(mode); });
            return *this;
        }

        /**
         * Sets the module in evaluation mode. This affects certain modules like Dropout
         * and BatchNorm whose behavior differs between training and evaluation.
         *
         * This method also propagates to any child modules that might be members of this module.
         *
         * @return A reference to this module for method chaining.
         */
        virtual Module &eval()
        {
            this->training = false;
            apply_to_children([](Module &child)
                              { child.eval(); });
            return *this;
        }

        /**
         * Check if the module is in training mode.
         *
         * @return true if the module is in training mode, false otherwise.
         */
        inline bool is_training() const { return this->training; }

        /**
         * Register all parameters of this module that need optimization into the given maps with keys as the parameter names and values as the parameter tensors
         * @param params Map to store parameters
         * @param grads Map to store gradients
         * @param prefix Prefix for parameter names
         */
        virtual void register_parameters(
            unordered_map<string, Tensor<> *> &params,
            unordered_map<string, Tensor<> *> &grads,
            const string &prefix = "") const
        {
            // Default implementation does nothing
            // Modules with parameters should override this
        }

    protected:
        /**
         * Cached input data for use in backward pass computations.
         */
        Tensor<> input_cache_;
        bool training = true;

        /**
         * Virtual method to apply a function to all child modules.
         * Modules that contain other modules should override this method.
         *
         * @param fn A function that takes a Module reference and returns void.
         */
        virtual void apply_to_children(const function<void(Module &)> &fn)
        {
            // Default implementation does nothing
            // Modules with children should override this
        }
    };

}