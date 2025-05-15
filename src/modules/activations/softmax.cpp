#include <math.h>
#include "softmax.hpp"
using namespace nn;

Softmax::Softmax()
{
    cout << "Starting Softmax" << endl;
    cout << "Softmax initialized" << endl;
}

Tensor<> Softmax::softmax_helper(const Tensor<> &input)
{
    Tensor<> result = input.map([](float x)
                                { return exp(x); });
    float sum = result.sum();

    return result / sum;
}

vector<float> Softmax::softmax_helper(const vector<float> &input)
{
    float sum = 0.0f;
    vector<float> result;

    for (size_t i = 0; i < input.size(); i++)
    {
        sum += exp(input[i]);
    }
    for (size_t i = 0; i < input.size(); i++)
    {
        result.push_back(exp(input[i]) / sum);
    }

    return result;
}

// Only support 1D and 2D Tensors
Tensor<> Softmax::forward(const Tensor<> &input)
{
    // In softmax case, we don't have to store the input as it is not used in the backward pass
    // Instead, we store the softmax(input)

    if (input.ndim() == 1)
    {
        return this->softmax_helper(input);
    }

    // const size_t leading_ndim = input.ndim() - 2;

    // vector<size_t> leading_shape(input.shapes().begin(), input.shapes().end() - 2);

    // const size_t n = input.shapes()[leading_ndim];
    // const size_t m = input.shapes()[leading_ndim + 1];

    vector<vector<float>> softmax_input;

    for (size_t i = 0; i < input.shapes()[0]; i++)
    {
        vector<float> input_row;
        input_row.reserve(input.shapes()[1]);

        for (size_t j = 0; j < input.shapes()[1]; j++)
        {
            input_row.push_back(input[i, j]);
        }
        softmax_input.push_back(this->softmax_helper(input_row));
    }

    this->softmax_input_cache_ = Tensor<>(softmax_input);

    return this->softmax_input_cache_;
}

Tensor<> Softmax::backward(const Tensor<> &grad_output)
{
    Tensor<> softmax_grad;

    return softmax_grad;
}