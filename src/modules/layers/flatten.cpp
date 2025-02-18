#include "flatten.hpp"
using namespace nn;

Flatten::Flatten(int64_t start_dim, int64_t end_dim) : start_dim_(start_dim), end_dim_(end_dim)
{
    cout << "Flatten layer initialized with start_dim = " << start_dim << " and end_dim = " << end_dim << endl;
}

Tensor<> Flatten::forward(const Tensor<> &input)
{
    this->original_shape_ = input.shapes();

    vector<size_t> new_shape;
    // return input.flatten(this->start_dim_, this->end_dim_);

    Tensor<> input_data = input;
    input_data.reshape(new_shape);

    return input_data;
}

Tensor<> Flatten::backward(const Tensor<> &grad_output)
{
    // return grad_output.reshape(this->original_shape_);
    return Tensor<>();
}

void Flatten::update_params(const float lr)
{
    return;
}
