#pragma once
#include "module.hpp"

namespace nn {

class Linear : public Module{
    public:
        Linear(int in_features, int out_features, bool bias);
        
        virtual vector<vector<float>> forward(const vector<vector<float>>& input) override;
        virtual vector<vector<float>> backward(const vector<vector<float>>& grad_output) override;
        virtual void update_params(const float lr) override;

        void randomizeParams();

        inline void setWeights(const vector<vector<float>>& desiredWeights) { this->weights_ = desiredWeights; };
        inline void setBiases(const vector<vector<float>>& desiredBiases) { this->biases_ = desiredBiases; }

        inline const vector<vector<float>>& getWeights() const { return this->weights_; }
        inline const vector<vector<float>>& getBiases() const { return this->biases_; }

    private:
        int in_features_;
        int out_features_;
        bool bias_;
        vector<vector<float>> weights_;
        vector<vector<float>> biases_;
        vector<vector<float>> grad_weights_;
        vector<vector<float>> grad_biases_;
    };  

}