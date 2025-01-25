#pragma once
#include <iostream>
#include <initializer_list>
#include "tensor_utils.hpp"
using namespace std;


template <typename T = double>
class Tensor {
    private:
        vector<T> data_; // data is stored as a 1D vector
        vector<size_t> shapes_; // store the dimensions of the tensor
        mutable size_t cached_size_ = 0; // store the number of elements in the tensor

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

            for (size_t i = 0; i < this->size(); i++) {
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

        // Helper function to convert negative indices to positive
        size_t normalize_index(size_t idx, size_t dim_size) const {
            if (idx < 0) idx += dim_size;
            cout << "dix in normalize_index: " << idx << endl;
            if (idx < 0 || idx > dim_size) {
                throw std::out_of_range("Index out of bounds");
            }
            return idx;
        }

        // Helper function to apply slice to a dimension
        std::vector<size_t> apply_slice(const Slice& slice, size_t dim_size) const {
            std::vector<size_t> indices;
            size_t start = normalize_index(slice.start, dim_size);
            size_t stop = slice.stop == INT_MAX ? dim_size : normalize_index(slice.stop, dim_size);
            size_t step = slice.step;
            
            cout << "start applying slice" << endl;
            for (size_t i = start; i < stop; i += step) {
                cout << "i: " << i << endl;
                indices.push_back(i);
            }
            return indices;
        }

        // Declare friendship so that TensorView can access private members of Tensor
        template<typename U, typename V>
        friend Tensor<V> dtype_impl(const Tensor<U>& tensor);

    public:
        Tensor() = default;

        // Scaler constructor
        Tensor(const T& value) {
            this->shapes_ = vector<size_t> {1};
            this->data_ = vector<T>(1, value);
        }

        // 1D tensor constructor
        Tensor(const initializer_list<T>& data_1d) {
            this->data_ = vector<T>(data_1d.begin(), data_1d.end());
            this->shapes_ = vector<size_t> { data_1d.size() };
        }

        Tensor(const vector<T>& data_1d) {
            this->data_ = data_1d;
            this->shapes_ = vector<size_t> { data_1d.size() };
        }

        // 2D tensor constructor
        Tensor(const initializer_list<initializer_list<T>>& data_2d) {
            const size_t n = data_2d.size(), m = data_2d.begin()->size();

            this->shapes_ = vector<size_t> { n, m };

            this->data_.reserve(n * m);  // Optimize memory allocation
            for (const initializer_list<T>& row : data_2d) {
                this->data_.insert(this->data_.end(), row.begin(), row.end());
            }
        }

        Tensor(const vector<vector<T>>& data_2d) {
            const size_t n = data_2d.size(), m = data_2d.begin()->size();

            this->shapes_ = vector<size_t> { n, m };

            this->data_.reserve(n * m);  // Optimize memory allocation
            for (const vector<T>& row : data_2d) {
                this->data_.insert(this->data_.end(), row.begin(), row.end());
            }
        }

        // 3D tensor constructor
        Tensor(const initializer_list<initializer_list<initializer_list<T>>>& data_3d) {
            const size_t n = data_3d.size(), m = data_3d.begin()->size(), l = data_3d.begin()->begin()->size();

            this->shapes_ = vector<size_t> { n, m, l };

            this->data_.reserve(n * m * l);  // Optimize memory allocation
            for (const initializer_list<initializer_list<T>>& matrix : data_3d) {
                for (const initializer_list<T>& row : matrix) {
                    this->data_.insert(this->data_.end(), row.begin(), row.end());
                }
            }
        }

        Tensor(const vector<vector<vector<T>>>& data_3d) {
            const size_t n = data_3d.size(), m = data_3d.begin()->size(), l = data_3d.begin()->begin()->size();

            this->shapes_ = vector<size_t> { n, m, l };

            this->data_.reserve(n * m * l);  // Optimize memory allocation
            for (const vector<vector<T>>& matrix : data_3d) {
                for (const vector<T>& row : matrix) {
                    this->data_.insert(this->data_.end(), row.begin(), row.end());
                }
            }
        }

        // certin value constructor
        Tensor(const vector<size_t>& shape, const T& value) {
            this->shapes_ = shape;
            size_t size = 1;
            for (const size_t& dim : shape) {
                size *= dim;
            }
            this->data_.resize(size, value);
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

            for (size_t i = 0; i < this->size(); i++) {
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
            Tensor<T> result({ this->size() }, static_cast<T>(0));

            for (size_t i = 0; i < this->size(); i++) {
                result.data_[i] = this->data_[i];
            }

            return result;
        }

        /// @brief Flatten the tensor into 1D in-place.
        /// @details This function only changes the shape of the tensor, and does not modify the underlying data.
        /// @post The shape of the tensor is changed to 1D, with the same elements as the original tensor.
        void flatten() {
            this->shapes_ = { this->size() };
            return;
        }
        

        // Assign this tensor with other tensor. Exactly the same as copy constructor
        Tensor<T>& assign(const Tensor<T>& other) {
            if (this == &other) return *this;

            this->shapes_ = other.shapes_;
            this->data_ = other.data_;
            return *this;
        }

        /// @brief Calculate the absolute value of each element in the tensor
        /// @return a new tensor with the same shape as the original, but with each element replaced by its absolute value
        Tensor<T> abs() const {
            Tensor<T> result(this->shapes_, static_cast<T>(0));

            for (size_t i = 0; i < this->size(); i++) {
                result.data_[i] = std::abs(this->data_[i]);
            }
            return result;
        }

        /// @brief Filter the tensor with the given function
        /// @param func a function to test each element of the tensor. It should return true if the element passes the test
        /// @return a new tensor with the same shape as the original, but all elements that fail the test are set to 0
        Tensor<T> filter(bool (*func)(T)) const {
            Tensor<T> result(this->shapes_, static_cast<T>(0));

            for (size_t i = 0; i < this->size(); i++) {
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

            for (size_t i = 0; i < this->size(); i++) {
                result.data_[i] = func(this->data_[i]);
            }
            return result;
        }

        /// @brief Calculate the sum of all elements in the tensor
        /// @return The sum of all elements in the tensor, regardless of the dimension
        T sum() const {
            T sum = static_cast<T>(0);

            for (size_t i = 0; i < this->size(); i++) {
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

            for (size_t i = 0; i < this->size(); i++) {
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

            for (size_t i = 0; i < this->size(); i++) {
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
        
        // Get the dimension of the tensor
        inline size_t ndim() const {
            return this->shapes_.size();
        }

        inline const size_t size() const {
            return this->data_.size();
        }
        

        inline const vector<size_t>& shapes() const { return this->shapes_; }

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

        T& operator[](const vector<size_t>& indices) {
            return this->data_[calculateIndex(indices)]; 
        }

        // rvalue operator overloading
        template<typename... Indices>
        const T& operator[](Indices... indices) const {
            vector<size_t> idxs = this->getIdxs(indices...);
            return this->data_[calculateIndex(idxs)]; 
        }

        const T& operator[](const vector<size_t>& indices) const {
            return this->data_[calculateIndex(indices)]; 
        }


        using IndexType = std::variant<size_t, std::string, Slice>;
        Tensor<T> index(const std::vector<IndexType>& indices) const {
            std::vector<std::vector<size_t>> expanded_indices;
            int ellipsis_pos = -1;
            
            // Handle ellipsis and expand slices
            cout << "Start expanding indices" << endl;
            for (size_t i = 0; i < indices.size(); ++i) {
                cout << "Index: " << i << endl;
                const auto& idx = indices[i];
                
                if (std::holds_alternative<std::string>(idx)) {
                    cout << "Inside string" << endl;
                    const std::string& str_idx = std::get<std::string>(idx);
                    if (str_idx == "...") {
                        if (ellipsis_pos != -1) {
                            throw std::invalid_argument("Only one ellipsis allowed");
                        }
                        ellipsis_pos = i;
                    } 
                    else if (str_idx == "None") {
                        expanded_indices.push_back({static_cast<size_t>(-1)}); // -1 represents None/newaxis
                    } 
                    else {
                        // Handle slice
                        Slice slice = Slice::parse(str_idx);
                        expanded_indices.push_back(apply_slice(slice, this->shapes_[i]));
                    }
                } 
                else if (std::holds_alternative<size_t>(idx)) {
                    cout << "Inside size_t" << endl;
                    expanded_indices.push_back({normalize_index(std::get<size_t>(idx), this->shapes_[i])});
                } 
                else if (std::holds_alternative<Slice>(idx)) {
                    cout << "Inside Slice" << endl;
                    expanded_indices.push_back(apply_slice(std::get<Slice>(idx), this->shapes_[i]));
                }
                else {
                    throw std::invalid_argument("Invalid index type");
                }
            }
            
            // Handle ellipsis
            if (ellipsis_pos != -1) {
                size_t missing_dims = this->shapes_.size() - (indices.size() - 1);
                std::vector<std::vector<size_t>> new_expanded_indices;
                
                for (size_t i = 0; i < ellipsis_pos; ++i) {
                    new_expanded_indices.push_back(expanded_indices[i]);
                }
                
                for (size_t i = 0; i < missing_dims; ++i) {
                    std::vector<size_t> full_range;
                    for (size_t j = 0; j < this->shapes_[ellipsis_pos + i]; ++j) {
                        full_range.push_back(j);
                    }
                    new_expanded_indices.push_back(full_range);
                }
                
                for (size_t i = ellipsis_pos + 1; i < expanded_indices.size(); ++i) {
                    new_expanded_indices.push_back(expanded_indices[i]);
                }
                
                expanded_indices = std::move(new_expanded_indices);
            }
            
            // Calculate new dimensions
            std::vector<size_t> new_dims;
            for (const vector<size_t>& expanded_idx : expanded_indices) {
                if (expanded_idx[0] != -1) { // Not None/newaxis
                    if (expanded_idx.size() > 1) {
                        new_dims.push_back(expanded_idx.size());
                    }
                } 
                else {
                    new_dims.push_back(1);
                }
            }

            cout << "Start printing new_dims" << endl;
            cout << "new_dims size: " << new_dims.size() << endl;
            for (size_t i = 0; i < new_dims.size(); ++i) {
                cout << new_dims[i] << " ";
            }
            
            // Create result tensor
            Tensor<T> result(new_dims, static_cast<T>(0));
            
            // Fill result tensor
            std::vector<size_t> current_indices(expanded_indices.size());
            std::vector<size_t> result_indices;
            
            // Recursive lambda to fill result tensor
            std::function<void(size_t)> fill_tensor = [&](size_t depth) {
                if (depth == expanded_indices.size()) {
                    result_indices.clear();
                    for (int i = 0; i < expanded_indices.size(); ++i) {
                        if (expanded_indices[i][0] != -1 && expanded_indices[i].size() > 1) {
                            result_indices.push_back(current_indices[i]);
                        }
                    }
                    
                    std::vector<size_t> original_indices;
                    for (int i = 0; i < expanded_indices.size(); ++i) {
                        if (expanded_indices[i][0] != -1) {
                            original_indices.push_back(expanded_indices[i][current_indices[i]]);
                        }
                    }
                    
                    result[result_indices] = (*this)[original_indices];
                    return;
                }
                
                for (int i = 0; i < expanded_indices[depth].size(); ++i) {
                    current_indices[depth] = i;
                    fill_tensor(depth + 1);
                }
            };
            
            fill_tensor(0);
            return result;
        }
};