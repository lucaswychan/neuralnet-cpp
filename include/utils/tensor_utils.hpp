#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <variant>
#include <string>
#include <sstream>
#include <functional>
#include <climits>

using namespace std;

// ================================================declaration================================================

template <typename T>
class Tensor;

// Convert tensor to different data type
template<typename U, typename V>
Tensor<V> dtype_impl(const Tensor<U>& tensor);

// for max, min ,argmax, argmin reduction
enum class ReduceOp {
    MAX,
    MIN,
    ARGMAX,
    ARGMIN
};

// for add, subtract, multiply, divide
enum class ArithmeticOp {
    ADD,
    SUB,
    MUL,
    DIV
};

// Slice struct to handle Python-like slicing
struct Slice {
    int start;
    int stop;
    int step;
    
    Slice(int start_ = 0, int stop_ = -1, int step_ = 1)
        : start(start_), stop(stop_), step(step_) {}
    
    static Slice parse(const string& slice_str) {
        Slice result;
        istringstream ss(slice_str);
        string token;
        vector<int> values;
        vector<uint8_t> is_empty;
        
        while (getline(ss, token, ':')) {
            // cout << "token: " << token << endl;
            if (token.empty()) {
                values.push_back(-1);
                is_empty.push_back(1);
            } else {
                values.push_back(stoi(token));
                is_empty.push_back(0);
            }
        }

        result.start = is_empty[0] == 1 ? 0 : values[0];
        result.stop = is_empty[1] == 1 ? INT_MAX : values[1];

        if (values.size() == 3) {
            result.step = is_empty[2] == 1 ? 1 : values[2];
        }
        
        // cout << "start: " << result.start << " stop: " << result.stop << " step: " << result.step << endl;
        return result;
    }
};

// ================================================definition================================================

template<typename U, typename V>
Tensor<V> dtype_impl(const Tensor<U>& tensor) {
    Tensor<V> result;
    result.shapes_ = tensor.shapes_;
    result.data_.resize(tensor.data_.size());
    
    std::transform(tensor.data_.begin(), tensor.data_.end(), result.data_.begin(),
        [](const U& val) { return static_cast<V>(val); });
        
    return result;
}