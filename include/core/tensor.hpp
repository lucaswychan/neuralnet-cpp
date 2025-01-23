#pragma once
#include <iostream>
#include <initializer_list>
#include "tensor_utils.hpp"
using namespace std;


template <typename T = double>
class Tensor {
    private:
        vector<size_t> shapes_; // store the dimensions of the tensor
        size_t size_; // store the number of elements in the tensor
        vector<T> data_; // data is stored as a 1D vector

        // Helper function to calculate the index in the 1D vector for a given set of indices expressed in the form of N-D vector
        size_t calculateIndex(const vector<size_t>& idxs) const {
            size_t index = 0;
            size_t multiplier = 1;
            for (size_t i = this->ndim(); i-- > 0;) {
                index += idxs[i] * multiplier;
                multiplier *= this->shapes_[i];
            }
            return index;
        }

        // Helper function for printing since we don't know the number of dimensions
        void printRecursive(size_t dim, size_t offset) const {
            if (dim == this->ndim() - 1) { // Last dimension
                cout << "[";
                for (size_t i = 0; i < this->shapes_[dim]; ++i) {
                    cout << data_[offset + i];
                    if (i < this->shapes_[dim] - 1) cout << ", ";
                }
                cout << "]";
            } else {
                cout << "[";
                size_t stride = 1;
                for (size_t i = dim + 1; i < this->ndim(); ++i) {
                    stride *= this->shapes_[i];
                }
                for (size_t i = 0; i < this->shapes_[dim]; ++i) {
                    printRecursive(dim + 1, offset + i * stride);
                    if (i < this->shapes_[dim] - 1) cout << ", " << endl;
                }
                cout << "]" << endl;
            }
        }

        // Helper function for operator[] overloading
        template<typename... Indices>
        const vector<size_t> getIdxs(Indices... indices) const {
            // Convert variadic arguments to vector
            vector<size_t> idxs({static_cast<size_t>(indices)...});

            // Check bounds
            for (size_t i = 0; i < idxs.size(); ++i) {
                if (idxs[i] < 0 || idxs[i] >= this->shapes_[i]) {
                    throw std::out_of_range("Tensor: Index out of bounds");
                }
            }

            return idxs;
        }

        template <typename U = T>
        Tensor<U> reduce(ReduceOp op) const {        
            if (this->ndim() > 2) {
                throw runtime_error("Only 1D and 2D tensors are supported for reduce");
            }

            const size_t num_rows = (this->ndim() == 2)? this->shapes_[0] : 1;
            const size_t num_cols = (this->ndim() == 2)? this->shapes_[1] : this->shapes_[0];

            
            // Result will be a tensor of shape (num_rows, 1)
            vector<U> result(num_rows);
            
            for (size_t i = 0; i < num_rows; i++) {
                size_t start_idx = i * num_cols;
                
                // Initialize with first element in row
                T extreme_val = this->data_[start_idx];
                size_t extreme_idx = 0;
                
                // Process remaining elements in the row
                for (size_t j = 1; j < num_cols; j++) {
                    size_t curr_idx = start_idx + j;
                    bool update = false;
                    
                    switch (op) {
                        case ReduceOp::MAX:
                        case ReduceOp::ARGMAX:
                            update = this->data_[curr_idx] > extreme_val;
                            break;
                        case ReduceOp::MIN:
                        case ReduceOp::ARGMIN:
                            update = this->data_[curr_idx] < extreme_val;
                            break;
                    }
                    
                    if (update) {
                        extreme_val = this->data_[curr_idx];
                        extreme_idx = j;
                    }
                }
                
                // Store the result
                switch (op) {
                    case ReduceOp::MAX:
                    case ReduceOp::MIN:
                        result[i] = extreme_val;
                        break;
                    case ReduceOp::ARGMAX:
                    case ReduceOp::ARGMIN:
                        result[i] = extreme_idx;
                        break;
                }
            }
            
            return Tensor<U>(result);
        }

        Tensor<T> arithmetic_operation_impl(ArithmeticOp op, const Tensor<T>& other) const {
            if (other.shapes_ != this->shapes_) {
                throw runtime_error("Shape mismatch in arithmetic operation");
            }

            Tensor<T> result(this->shapes_, static_cast<T>(0));

            for (size_t i = 0; i < this->size_; i++) {
                switch (op) {
                    case ArithmeticOp::ADD:
                        result.data_[i] = this->data_[i] + other.data_[i];
                        break;
                    case ArithmeticOp::SUB:
                        result.data_[i] = this->data_[i] - other.data_[i];
                        break;
                    case ArithmeticOp::MUL:
                        result.data_[i] = this->data_[i] * other.data_[i];
                        break;
                    case ArithmeticOp::DIV:
                        result.data_[i] = this->data_[i] / other.data_[i];
                        break;
                }
            }
            return result;
        }

        // Declare friendship so that TensorView can access private members of Tensor
        template<typename U, typename V>
        friend Tensor<V> dtype_impl(const Tensor<U>& tensor);

        template<typename U>
        friend class TensorView;

    public:
        Tensor() = default;

        // Scaler constructor
        Tensor(const T& value) {
            this->shapes_ = vector<size_t> {1};
            this->data_ = vector<T>(1, value);
        }

        // 1D tensor constructor
        Tensor(const initializer_list<T>& data) : size_(data.size()), data_(data.begin(), data.end()) {
            this->shapes_ = vector<size_t> { data.size() };
        }

        Tensor(const vector<T>& data) : size_(data.size()), data_(data.begin(), data.end()) {
            this->shapes_ = vector<size_t> { data.size() };
        }

        // 2D tensor constructor
        Tensor(const initializer_list<initializer_list<T>>& data_2d) {
            const size_t n = data_2d.size(), m = data_2d.begin()->size();
            this->size_ = n * m;

            this->shapes_ = vector<size_t> { n, m };

            for (const initializer_list<T>& row : data_2d) {
                this->data_.insert(this->data_.end(), row.begin(), row.end());
            }
        }

        Tensor(const vector<vector<T>>& data_2d) {
            const size_t n = data_2d.size(), m = data_2d.begin()->size();
            this->size_ = n * m;

            this->shapes_ = vector<size_t> { n, m };

            for (const vector<T>& row : data_2d) {
                this->data_.insert(this->data_.end(), row.begin(), row.end());
            }
        }

        // 3D tensor constructor
        Tensor(const initializer_list<initializer_list<initializer_list<T>>>& data_3d) {
            const size_t n = data_3d.size(), m = data_3d.begin()->size(), l = data_3d.begin()->begin()->size();
            this->size_ = n * m * l;

            this->shapes_ = vector<size_t> { n, m, l };

            for (const initializer_list<initializer_list<T>>& matrix : data_3d) {
                for (const initializer_list<T>& row : matrix) {
                    this->data_.insert(this->data_.end(), row.begin(), row.end());
                }
            }
        }

        Tensor(const vector<vector<vector<T>>>& data_3d) {
            const size_t n = data_3d.size(), m = data_3d.begin()->size(), l = data_3d.begin()->begin()->size();
            this->size_ = n * m * l;

            this->shapes_ = vector<size_t> { n, m, l };

            for (const vector<vector<T>>& matrix : data_3d) {
                for (const vector<T>& row : matrix) {
                    this->data_.insert(this->data_.end(), row.begin(), row.end());
                }
            }
        }

        // certin value constructor
        Tensor(const vector<size_t>& shape, const T& value) {
            this->shapes_ = shape;
            this->size_ = 1;
            for (const size_t& s : shape) {
                this->size_ *= s;
            }

            this->data_.resize(this->size_, value);
        }

        // copy constructor
        Tensor(const Tensor<T>& other) {
            // already overloading operator=
            *this = other;
        }

        // Add two tensors with same shape, element-wise
        inline Tensor<T> add(const Tensor& other) const {
            return arithmetic_operation_impl(ArithmeticOp::ADD, other);
        }

        // Subtract two tensors with same shape, element-wise
        inline Tensor<T> sub(const Tensor<T>& other) const {
            return arithmetic_operation_impl(ArithmeticOp::SUB, other);
        }

        // Multiply two tensors with same shape, element-wise
        inline Tensor<T> mul(const Tensor<T>& other) const {
            return arithmetic_operation_impl(ArithmeticOp::MUL, other);
        }

        // Divide two tensors with same shape, element-wise
        inline Tensor<T> div(const Tensor<T>& other) const {
            return arithmetic_operation_impl(ArithmeticOp::DIV, other);
        }

        // Multiply all elements of tensor with the given scaler
        Tensor<T> mul(const T& scaler) const {
            Tensor<T> result(this->shapes_, static_cast<T>(0));

            for (size_t i = 0; i < this->size_; i++) {
                result.data_[i] = this->data_[i] * scaler;
            }
            return result;
        }

        /// @brief Matrix multiplication.
        /// @details This function supports matrix multiplication between two 2D tensors.
        /// The first tensor is of shape (n, m) and the second tensor is of shape (m, p).
        /// The resulting tensor is of shape (n, p).
        /// @param other The second tensor to multiply with.
        /// @return A new tensor that is the result of the matrix multiplication.
        Tensor<T> matmul(const Tensor<T>& other) const {
            // this->data_ R^n x m, other.data_ R^m x p
            const size_t n = this->shapes_[0], m = this->shapes_[1], p = other.shapes_[1];

            if (m != other.shapes_[0]) {
                throw runtime_error("Shape mismatch in matrix multiplication");
            }
            if (this->ndim() != 2 || other.ndim() != 2) {
                throw runtime_error("Only 2D tensors are supported for matrix multiplication");
            }

            Tensor<T> result({ n, p }, static_cast<T>(0));

            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < p; j++) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < m; k++) {
                        sum += this->data_[i * m + k] * other.data_[k * p + j];
                    }
                    result[i, j] = static_cast<T>(sum);
                }
            }
            
            return result;
        }

        
        /// @brief Transpose the tensor.
        /// @details This function supports transposing 1D and 2D tensors.
        /// 1D tensors are transposed from shape (1, n) to (n, 1).
        /// For 2D tensors, it swaps rows and columns.
        /// @return A new tensor that is the transpose of the original tensor.
        /// @throws runtime_error if the tensor has more than 2 dimensions.

        Tensor<T> transpose() const {
            if (this->ndim() > 2) {
                throw runtime_error("Only 1D and 2D tensors are supported for transpose");
            }

            // We want 1D vector to be transposed from 1xn to nx1
            const size_t n = (this->ndim() == 2)? this->shapes_[0] : 1;
            const size_t m = (this->ndim() == 2)? this->shapes_[1] : this->shapes_[0];

            Tensor result({m, n}, static_cast<T>(0));

            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    result.data_[j * n + i] = data_[i * m + j];
                }
            }
            return result;
        }


        
        /// @brief Flatten the tensor into 1D.
        /// @return A new 1D tensor with the same elements as the original tensor.
        Tensor<T> flatten() const {
            Tensor<T> result({ this->size_ }, static_cast<T>(0));

            for (size_t i = 0; i < this->size_; i++) {
                result.data_[i] = this->data_[i];
            }

            return result;
        }

        /// @brief Flatten the tensor into 1D in-place.
        /// @details This function only changes the shape of the tensor, and does not modify the underlying data.
        /// @post The shape of the tensor is changed to 1D, with the same elements as the original tensor.
        void flatten() {
            this->shapes_ = { this->size_ };
            return;
        }
        

        // Assign this tensor with other tensor. Exactly the same as copy constructor
        Tensor<T>& assign(const Tensor<T>& other) {
            if (this == &other) return *this;

            this->shapes_ = other.shapes_;
            this->size_ = other.size_;
            this->data_ = other.data_;
            return *this;
        }

        /// @brief Calculate the absolute value of each element in the tensor
        /// @return a new tensor with the same shape as the original, but with each element replaced by its absolute value
        Tensor<T> abs() const {
            Tensor<T> result(this->shapes_, static_cast<T>(0));

            for (size_t i = 0; i < this->size_; i++) {
                result.data_[i] = std::abs(this->data_[i]);
            }
            return result;
        }

        /// @brief Filter the tensor with the given function
        /// @param func a function to test each element of the tensor. It should return true if the element passes the test
        /// @return a new tensor with the same shape as the original, but all elements that fail the test are set to 0
        Tensor<T> filter(bool (*func)(T)) const {
            Tensor<T> result(this->shapes_, static_cast<T>(0));

            for (size_t i = 0; i < this->size_; i++) {
                if (func(this->data_[i])) {
                    result.data_[i] = this->data_[i];
                }
            }
            return result;
        }

        /// @brief Perform element-wise transformation with a function
        /// @param func a function perform element-wise transformation to the tensor
        /// @return a new tensor with the same shape as the original, but with each element transformed by the given func
        Tensor<T> map(T (*func)(T)) const {
            Tensor<T> result(this->shapes_, static_cast<T>(0));

            for (size_t i = 0; i < this->size_; i++) {
                result.data_[i] = func(this->data_[i]);
            }
            return result;
        }

        /// @brief Calculate the sum of all elements in the tensor
        /// @return The sum of all elements in the tensor, regardless of the dimension
        T sum() const {
            T sum = static_cast<T>(0);

            for (size_t i = 0; i < this->size_; i++) {
                sum += this->data_[i];
            }
            
            return sum;
        }

        /// @brief Check if all elements of two tensors are equal
        /// @param other Tensor to compare
        /// @return Tensor of integers where each element is 1 if the two tensors are equal at the same index, 0 otherwise
        Tensor<int> equal(const Tensor& other) const{
            if (other.shapes_ != this->shapes_) {
                throw runtime_error("Shape mismatch");
            }

            Tensor<T> result(this->shapes_, static_cast<T>(0));

            for (size_t i = 0; i < this->size_; i++) {
                result.data_[i] = this->data_[i] == other.data_[i];
            }

            return result.dtype<int>();
        }

        /// @brief Check if all elements of two tensors are equal
        /// @param other Tensor to compare
        /// @return true if all elements are equal, false otherwise
        bool compare(const Tensor& other) const {
            if (other.shapes_ != this->shapes_) {
                throw runtime_error("Shape mismatch");
            }

            for (size_t i = 0; i < this->size_; i++) {
                if (this->data_[i] != other.data_[i]) {
                    return false;
                }
            }
            return true;
        }

        /// @brief Reduce the tensor to the maximum value of all elements
        /// @return a tensor with a single element, the maximum of all elements in the tensor
        inline Tensor<> max() const {
            return reduce(ReduceOp::MAX);
        }


        /// @brief Reduce the tensor to the indices of the maximum values along each row
        /// @return a tensor with indices of the maximum values for each row
        inline Tensor<size_t> argmax() const {
            return reduce<size_t>(ReduceOp::ARGMAX);
        }

        /// @brief Reduce the tensor to the minimum value of all elements
        /// @return a tensor with a single element, the minimum of all elements in the tensor
        inline Tensor<> min() const {
            return reduce(ReduceOp::MIN);
        }

        /// @brief Reduce the tensor to the indices of the minimum values along each row
        /// @return a tensor with indices of the minimum values for each row
        inline Tensor<size_t> argmin() const {
            return reduce<size_t>(ReduceOp::ARGMIN);
        }

        
        /// @brief Convert the tensor to a tensor of a different type.
        /// @details If U is not provided, it defaults to double.
        /// @param U the type to convert to
        /// @return a tensor with the same shape and data, but with the type U
        template<typename U = double>
        Tensor<U> dtype() const {
            return dtype_impl<T, U>(*this);
        }


        /**
         * @brief Get a view of a specific slice along a specified dimension.
         * 
         * This function creates a `TensorView` that represents a reference-like view 
         * into the tensor data along a specified dimension. The view does not contain 
         * actual data but provides access to the underlying tensor data.
         * 
         * @tparam Dim The dimension along which the slice is taken.
         * @param index The index of the slice along the specified dimension.
         * @return A `TensorView<T>` representing the view of the specified slice.
         * @throws std::std::out_of_range if the index is out of bounds for the given dimension.
         */

        template<size_t Dim>
        TensorView<T> slice(size_t index) {
            if (index >= this->shapes_[Dim]) throw std::out_of_range("Index out of bounds");

            // Calculate strides for the resulting view
            vector<size_t> view_strides;
            size_t slice_size = 1;
            for (size_t i = 0; i < this->ndim(); ++i) {
                if (i != Dim) {
                    view_strides.push_back(slice_size);
                    slice_size *= this->shapes_[i];
                }
            }

            // Create initial indices with the fixed dimension
            vector<size_t> indices(this->ndim(), 0);
            indices[Dim] = index;

            // It will only return a view, not the real row or col data
            return TensorView<T>(*this, indices, view_strides, Dim, slice_size);
        }


        /**
         * @brief Get a view of a specific row in the tensor.
         * 
         * This function creates a `TensorView` that represents a reference-like view 
         * into the tensor data for a specific row. The view does not contain actual data 
         * but provides access to the underlying tensor data.
         * 
         * @param index The index of the row to view.
         * @return A `TensorView<T>` representing the view of the specified row.
         * @throws std::std::out_of_range if the index is out of bounds for the row dimension.
         */
        inline TensorView<T> row(size_t index) {
            return this->slice<0>(index);
        }

        /**
         * @brief Get a view of a specific column in the tensor.
         * 
         * This function creates a `TensorView` that represents a reference-like view 
         * into the tensor data for a specific column. The view does not contain actual data 
         * but provides access to the underlying tensor data.
         * 
         * @param index The index of the column to view.
         * @return A `TensorView<T>` representing the view of the specified column.
         * @throws std::std::out_of_range if the index is out of bounds for the column dimension.
         */
        inline TensorView<T> col(size_t index) {
            return this->slice<1>(index);
        }
        
        // Get the dimension of the tensor
        inline size_t ndim() const {
            return this->shapes_.size();
        }

        inline const vector<size_t>& shapes() const { return this->shapes_; }

        inline const size_t size() const { return this->size_; }

        inline void print() const { printRecursive(0, 0); }

        // ========================================operators overloading========================================
        inline Tensor<T> operator+(const Tensor<T>& other) const { return this->add(other); }
        inline Tensor<T> operator-(const Tensor<T>& other) const { return this->sub(other); }
        inline Tensor<T> operator*(const Tensor<T>& other) const { return this->mul(other); }
        inline Tensor<T> operator*(const T& scaler) const { return this->mul(scaler); }
        inline Tensor<T>& operator=(const Tensor<T>& other) { return this->assign(other); }
        inline bool operator==(const Tensor<T>& other) const { return this->compare(other); }

        const Tensor<T> operator+=(const Tensor<T>& other) { 
            *this = *this + other;
            return *this;
        }

        const Tensor<T> operator-=(const Tensor<T>& other) {
            *this = *this - other;
            return *this;
        }

        const Tensor<T> operator*=(const Tensor<T>& other) {
            *this = *this * other;
            return *this;
        }

        const Tensor<T> operator*=(const T& other) {
            *this = *this * other;
            return *this;
        }

        // lvalue operator overloading
        template<typename... Indices>
        T& operator[](Indices... indices) {
            // ((cout << ',' << std::forward<Indices>(indices)), ...);
            // cout << endl;

            vector<size_t> idxs = this->getIdxs(indices...);
            return this->data_[calculateIndex(idxs)]; 
        }

        // rvalue operator overloading
        template<typename... Indices>
        T operator[](Indices... indices) const {
            vector<size_t> idxs = this->getIdxs(indices...);
            return this->data_[calculateIndex(idxs)]; 
        }
};