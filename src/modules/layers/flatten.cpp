#include "flatten.hpp"
using namespace nn;

Flatten::Flatten(int64_t start_dim, int64_t end_dim) : start_dim_(start_dim), end_dim_(end_dim)
{
    cout << "Flatten layer initialized with start_dim = " << start_dim << " and end_dim = " << end_dim << endl;
}

Tensor<> Flatten::forward(const Tensor<> &input)
{
    this->original_input_shape_ = input.shapes();

    return input.flatten(this->start_dim_, this->end_dim_);
}

Tensor<> Flatten::backward(const Tensor<> &grad_output)
{
    return grad_output.reshape(this->original_input_shape_);
}
