#pragma once
#include "module.hpp"

namespace nn {

class Linear : public Module{
    public:
        Linear(size_t in_features, size_t out_features, bool bias);
        
        virtual Tensor<> forward(const Tensor<>& input) override;
        virtual Tensor<> backward(const Tensor<>& grad_output) override;
        virtual void update_params(const float lr) override;

        void randomizeParams();

        inline void setWeights(const Tensor<>& desiredWeights) { this->weights_ = desiredWeights; };
        inline void setBiases(const Tensor<>& desiredBiases) { this->biases_ = desiredBiases; }

        inline const Tensor<>& getWeights() const { return this->weights_; }
        inline const Tensor<>& getBiases() const { return this->biases_; }

    private:
        size_t in_features_;
        size_t out_features_;
        bool bias_;
        Tensor<> weights_;
        Tensor<> biases_;
        Tensor<> grad_weights_;
        Tensor<> grad_biases_;
    };  

}