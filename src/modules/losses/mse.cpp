#include "modules/losses/mse.hpp"
using namespace nn;

MSE::MSE() {}

float MSE::forward(const vector<vector<float>>& Y, const vector<vector<float>>& Y_hat) {
    // Y R^B x M, Y_hat R^B x M , B is the batch size and M is the output dimension

    // 1 / (B * M) * ||(Y - Y_hat)||^2

    this->Y_cache_ = Y;
    this->Y_hat_cache_ = Y_hat;
    
    const int B  = Y.size(), M = Y[0].size();
    if (Y_hat.size() != B || Y_hat[0].size() != M) {
        cout << "Error: Matrix dimensions are not compatible for subtraction." << endl;
        return 0.0f;
    }

    float loss = 0.0f;
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < M; j++) {
            const float diff = Y[i][j] - Y_hat[i][j];
            loss += diff * diff;
        }
    }

    return loss / (B * M);
}

vector<vector<float>> MSE::backward() {
    // dL/dY_hat should have the same shape as Y_hat

    // 2 / (B * M) * (Y - Y_hat)

    const int B = this->Y_cache_.size(), M = this->Y_cache_[0].size();
    vector<vector<float>> grad_output(B, vector<float>(M, 0.0f));
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < M; j++) {
            grad_output[i][j] = 2.0f * (this->Y_cache_[i][j] - this->Y_hat_cache_[i][j]) / (B * M);
        }
    }
    return grad_output;
}