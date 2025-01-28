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
#include <cstdint>

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
    
    Slice(int start_ = 0, int stop_ = -1, int step_ = 1) : start(start_), stop(stop_), step(step_) {}
    
    static Slice parse(const string& slice_str);
};

// Helper function to convert negative indices to positive
size_t normalize_index(int idx, size_t dim_size);

// Helper function to apply slice to a dimension
vector<size_t> apply_slice(const Slice& slice, size_t dim_size);

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