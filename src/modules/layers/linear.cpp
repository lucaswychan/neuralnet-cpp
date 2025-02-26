#include <random>
#include <cmath>
#include "linear.hpp"
using namespace nn;

Linear::Linear(size_t in_features, size_t out_features, bool bias) : in_features_(in_features), out_features_(out_features), use_bias_(bias)
{
    this->weight_ = Tensor<>({in_features, out_features}, 0.0f);

    if (this->use_bias_)
    {
        this->bias_ = Tensor<>({out_features, 1}, 0.0f);
    }

    // randomize the weights and bias based on PyTorch implementation
    this->reset_parameters();

    cout << "Linear layer initialized with in_features = " << in_features << " and out_features = " << out_features << endl;
    cout << &this->input_cache_ << endl;
}

Tensor<> Linear::forward(const Tensor<> &input)
{
    this->input_cache_ = input;
    size_t batchSize = input.shapes()[0];

    const Tensor<> &XW = input.matmul(this->weight_);

    if (!this->use_bias_)
    {
        return XW;
    }

    Tensor<> biases_repeated = Tensor<>({batchSize, this->out_features_}, 0.0f);

    for (size_t i = 0; i < batchSize; i++)
    {
        for (size_t j = 0; j < this->out_features_; j++)
        {
            biases_repeated[i, j] = this->bias_[j, 0];
        }
    }

    return XW + biases_repeated;
}

Tensor<> Linear::backward(const Tensor<> &grad_output)
{
    // dL/dY = grad_output

    // dL/dW = X^T * dL/dY
    this->grad_weight_ = this->input_cache_.transpose().matmul(grad_output);

    // cout << endl << "dL/dW: " << endl;
    // this->grad_weight_.print();
    // cout << endl;

    // dL/dX = dL/dY * W^T
    Tensor<> grad_input = grad_output.matmul(this->weight_.transpose());

    /*
    dL/db = dL/dY^T * 1_B (1_B is a vector of ones of size batchSize)
    dL/db = dL/dY.sum(axis=0)
    */
    if (this->use_bias_)
        this->grad_bias_ = grad_output.transpose().matmul(Tensor<>({grad_output.shapes()[0], 1}, 1.0f));

    // cout << endl << "dL/db: " << endl;
    // this->grad_bias_.print();
    // cout << endl;

    return grad_input;
}

void Linear::update_params(const float lr)
{

    this->weight_ -= this->grad_weight_ * lr;
    this->bias_ -= this->grad_bias_ * lr;

    return;
}

void Linear::reset_parameters()
{
    /*
    PyTorch implementation:

    stdv = 1. / math.sqrt(self.weight.size(1))
    self.weight.data.uniform_(-stdv, stdv)
    if self.bias is not None:
        self.bias.data.uniform_(-stdv, stdv)

    */
    // Calculate the limit for the uniform distribution
    const double stdv = 1.0 / sqrt(this->weight_.shapes()[0]); // since the weight is transposed

    // Set up the random number generator
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(-stdv, stdv);

    // Xavier initialization
    for (size_t i = 0; i < this->in_features_; i++)
    {
        for (size_t j = 0; j < this->out_features_; j++)
        {
            this->weight_[i, j] = dis(gen);
        }
    }

    if (this->use_bias_)
    {
        for (size_t i = 0; i < this->out_features_; i++)
        {
            this->bias_[i, 0] = dis(gen);
        }
    }
}