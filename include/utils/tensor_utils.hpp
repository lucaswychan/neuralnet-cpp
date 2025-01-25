#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <variant>
#include <string>
#include <sstream>
#include <functional>

using namespace std;

// ================================================declaration================================================

template <typename T>
class Tensor;

// Convert tensor to different data type
template<typename U, typename V>
Tensor<V> dtype_impl(const Tensor<U>& tensor);

// TensorView class to provide a reference-like view into tensor data
template<typename U>
class TensorView {
    Tensor<U>& tensor_;  // Reference to the original tensor
    vector<size_t> indices_;
    vector<size_t> strides_;
    size_t slice_dim_;
    size_t size_;
    
public:
    // Constructor
    TensorView(Tensor<U>& tensor, const vector<size_t>& indices, const vector<size_t>& strides, size_t slice_dim, size_t size)
        : tensor_(tensor)
        , indices_(indices)
        , strides_(strides)
        , slice_dim_(slice_dim)
        , size_(size) {}
    
    // Indexing operator
    U& operator[](size_t idx);
    
    inline size_t size() const { return this->size_; }

    // Iterator support
    inline U* begin() { return &operator[](0); }
    inline U* end() { return &operator[](this->size_); }
    inline const U* begin() const { return &operator[](0); }
    inline const U* end() const { return &operator[](this->size_); }
};

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
    
    static Slice parse(const std::string& slice_str) {
        Slice result;
        std::istringstream ss(slice_str);
        std::string token;
        std::vector<int> values;
        
        while (std::getline(ss, token, ':')) {
            if (token.empty()) {
                values.push_back(-1);
            } else {
                values.push_back(std::stoi(token));
            }
        }
        
        switch (values.size()) {
            case 2:
                result.start = values[0] == -1 ? 0 : values[0];
                result.stop = values[1] == -1 ? INT_MAX : values[1];
                break;
            case 3:
                result.start = values[0] == -1 ? 0 : values[0];
                result.stop = values[1] == -1 ? INT_MAX : values[1];
                result.step = values[2] == -1 ? 1 : values[2];
                break;
            default:
                throw std::invalid_argument("Invalid slice format");
        }
        
        cout << "start: " << result.start << " stop: " << result.stop << " step: " << result.step << endl;
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

// TensorView class to provide a reference-like view into tensor data
template<typename U>    // Indexing operator
U& TensorView<U>::operator[](size_t idx) {
    if (idx >= this->size_) throw std::out_of_range("TensorView: Index out of bounds");
    
    vector<size_t> full_indices = this->indices_;
    size_t remaining = idx;
    
    // Convert linear index back to multidimensional indices
    size_t curr_dim = 0;
    for (size_t i = 0; i < this->tensor_.ndim(); ++i) {
        if (i != this->slice_dim_) {
            full_indices[i] = remaining / strides_[curr_dim];
            remaining %= this->strides_[curr_dim];
            ++curr_dim;
        }
    }
    
    // Calculate final linear index in original data
    size_t final_idx = this->tensor_.calculateIndex(full_indices);
    
    return this->tensor_.data_[final_idx];
}