#pragma once
#include <iostream>
#include <cmath>
#include <numeric>
#include <vector>
#include <initializer_list>
#include <algorithm>
#include <stdexcept>
#include <variant>
#include <string>
#include <sstream>
#include <functional>
#include <climits>
#include <cstdint>
#include <type_traits>
#include <memory>
#include <unordered_set>

using namespace std;

// ================================================declaration================================================

template <typename T>
class Tensor;

// Convert tensor to different data type
template <typename U, typename V>
Tensor<V> dtype_impl(const Tensor<U> &tensor);

// for max, min ,argmax, argmin reduction
enum class ReduceOp
{
    MAX,
    MIN,
    ARGMAX,
    ARGMIN
};

// for add, subtract, multiply, divide
enum class ArithmeticOp
{
    ADD,
    SUB,
    MUL,
    DIV
};

// Slice struct to handle Python-like slicing
struct Slice
{
    int start;
    int stop;
    int step;

    Slice(int start_ = 0, int stop_ = -1, int step_ = 1) : start(start_), stop(stop_), step(step_) {}

    static Slice parse(const string &slice_str);
};

// Helper function to convert negative indices to positive
size_t normalize_index(int idx, size_t dim_size);

// Helper function to apply slice to a dimension
vector<size_t> apply_slice(const Slice &slice, size_t dim_size);

// Helper function to calculate the offset of the tensor given a single index
vector<size_t> linear_to_multi_idxs(size_t idx, const vector<size_t> &shape);

// Type trait to check if a type is a std::vector
template <typename>
struct is_vector : public std::false_type
{
};

template <typename T, typename A>
struct is_vector<std::vector<T, A>> : public std::true_type
{
};

// Type trait to check if a type is a std::vector
template <typename>
struct is_initializer_list : public std::false_type
{
};

template <typename T>
struct is_initializer_list<std::initializer_list<T>> : public std::true_type
{
};

// ================================================definition================================================

template <typename U, typename V>
Tensor<V> dtype_impl(const Tensor<U> &tensor)
{
    Tensor<V> result;

    result.shape_ = tensor.shape_;
    result.data_ = make_shared<vector<V>>();
    result.data_->resize(tensor.data_->size());
    result.strides_ = tensor.strides_;
    result.offset_ = tensor.offset_;
    result.size_ = tensor.size_;

    std::transform(tensor.data_->begin(), tensor.data_->end(), result.data_->begin(),
                   [](const U &val)
                   { return static_cast<V>(val); });

    return result;
}