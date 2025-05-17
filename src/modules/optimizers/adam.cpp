#include "adam.hpp"
#include <cmath>
using namespace nn;

// Constructor with model
Adam::Adam(const Module &model, float learning_rate, float beta1, float beta2, float epsilon, float weight_decay)
    : Optimizer(model, learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0), weight_decay_(weight_decay)
{
    // Initialize moment buffers
    for (auto &[name, param] : this->params_)
    {
        this->m_[name] = Tensor<>(param->shapes(), 0.0f);
        this->v_[name] = Tensor<>(param->shapes(), 0.0f);
    }
}

void Adam::step()
{
    /*
    Adam Algorithm:
    m_t = beta1 * m_{t-1} + (1 - beta1) * grad
    v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2

    m_t_corrected = m_t / (1 - beta1^t) // Bias correction
    v_t_corrected = v_t / (1 - beta2^t) // Bias correction

    param = param - learning_rate * m_t_corrected / (sqrt(v_t_corrected) + epsilon)
    */

    this->t_++; // Increment timestep

    float beta1_correction = 1.0f - pow(this->beta1_, this->t_);
    float beta2_correction = 1.0f - pow(this->beta2_, this->t_);

    for (auto &[name, param] : this->params_)
    {
        Tensor<> &grad = *this->grads_[name];
        Tensor<> &m = this->m_[name];
        Tensor<> &v = this->v_[name];

        // Apply weight decay
        if (this->weight_decay_ > 0.0f)
        {
            grad = grad + *param * this->weight_decay_;
        }

        // Update biased first and second moment estimates
        m = m * this->beta1_ + grad * (1.0f - this->beta1_);
        v = v * this->beta2_ + (grad * grad) * (1.0f - this->beta2_);

        // Compute bias-corrected moment estimates
        Tensor<> m_corrected = m / beta1_correction;
        Tensor<> v_corrected = v / beta2_correction;

        // Update parameters
        Tensor<> v_sqrt_eps = v_corrected.sqrt() + this->epsilon_;

        Tensor<> update = m_corrected / v_sqrt_eps;
        *param = *param - update * this->learning_rate_;
    }
}