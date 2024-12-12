#pragma once
#include <vector>
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

// ================================================definition================================================

template<typename U, typename V>
Tensor<V> dtype_impl(const Tensor<U>& tensor) {
    Tensor<V> result;
    result.shapes_ = tensor.shapes_;
    result.size_ = tensor.size_;
    result.data_.resize(tensor.data_.size());
    
    std::transform(tensor.data_.begin(), tensor.data_.end(), result.data_.begin(),
        [](const U& val) { return static_cast<V>(val); });
        
    return result;
}

// TensorView class to provide a reference-like view into tensor data
template<typename U>    // Indexing operator
U& TensorView<U>::operator[](size_t idx) {
    if (idx >= this->size_) throw out_of_range("TensorView: Index out of bounds");
    
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