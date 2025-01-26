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

        // Helper function to calculate the index in the 1D vector for a given set of indices expressed in the form of N-D vector
        size_t calculate_idx(const vector<size_t>& idxs) const {
            size_t idx = 0;
            size_t multiplier = 1;
            for (size_t i = this->ndim(); i-- > 0;) {
                idx += idxs[i] * multiplier;
                multiplier *= this->shapes_[i];
            }
            return idx;
        }

        // Helper function for printing since we don't know the number of dimensions
        void print_recursive(size_t dim, size_t offset) const {
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
                    print_recursive(dim + 1, offset + i * stride);
                    if (i < this->shapes_[dim] - 1) cout << ", " << endl;
                }
                cout << "]" << endl;
            }
        }

        // Helper function for operator[] overloading
        template<typename... Indices>
        const vector<size_t> get_idxs(Indices... indices) const {
            // Convert variadic arguments to vector
            vector<int> idxs({static_cast<int>(indices)...});
            vector<size_t> normalized_idxs;

            // for better performance, reserve the size of the vector
            normalized_idxs.reserve(idxs.size());

            // Check bounds
            for (size_t i = 0; i < idxs.size(); ++i) {
                size_t normalized_idx = this->normalize_index(idxs[i], this->shapes_[i]);
                normalized_idxs.push_back(normalized_idx);
            }

            return normalized_idxs;
        }

        /**
         * Reduces a 1D or 2D tensor along its rows using the specified reduction operation.
         *
         * @tparam U The data type of the resulting tensor. Defaults to the type of the current tensor.
         * @param op The reduction operation to perform. Supported operations are MAX, MIN, ARGMAX, and ARGMIN.
         * @return A Tensor<U> of shape (num_rows, 1) containing the reduced values or indices.
         * @throws runtime_error if the tensor's number of dimensions is greater than 2.
         */

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
        size_t normalize_index(int idx, size_t dim_size) const {
            if (idx < 0) idx += dim_size;
            if (idx < 0 || idx >= dim_size) {
                throw std::out_of_range("Index out of bounds after index normalization");
            }
            return idx;
        }

        // Helper function to apply slice to a dimension
        vector<size_t> apply_slice(const Slice& slice, size_t dim_size) const {
            vector<size_t> indices;
            // cout << "In apply_slice, start: " << slice.start << " stop: " << slice.stop << " step: " << slice.step << endl;
            size_t start = normalize_index(slice.start, dim_size);
            size_t stop = slice.stop == INT_MAX ? dim_size : normalize_index(slice.stop - 1, dim_size) + 1;
            size_t step = slice.step;
            
            // cout << "start applying slice" << endl;
            for (size_t i = start; i < stop; i += step) {
                // cout << "i: " << i << endl;
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

        /// @brief Reshape the tensor to the specified new shape.
        /// @details This function changes the shape of the tensor without altering the data.
        /// The total number of elements must remain the same; otherwise, an exception is thrown.
        /// @param new_shape The desired shape for the tensor.
        /// @throws runtime_error if the new shape is not compatible with the current number of elements.
        void reshape(const vector<size_t>& new_shape) {
            size_t new_size = 1;
            for (const size_t& dim : new_shape) {
                new_size *= dim;
            }

            if (new_size != this->size()) {
                throw runtime_error("New shape must be compatible with the original shape");
            }

            this->shapes_ = new_shape;
        }
        
        // Get the dimension of the tensor
        inline size_t ndim() const {
            return this->shapes_.size();
        }

        inline const size_t size() const {
            return this->data_.size();
        }
        

        inline const vector<size_t>& shapes() const { return this->shapes_; }

        inline void print() const { print_recursive(0, 0); }

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
            vector<size_t> idxs = this->get_idxs(indices...);
            return this->data_[calculate_idx(idxs)]; 
        }

        T& operator[](const vector<size_t>& indices) {
            return this->data_[calculate_idx(indices)]; 
        }

        // rvalue operator overloading
        template<typename... Indices>
        const T& operator[](Indices... indices) const {
            vector<size_t> idxs = this->get_idxs(indices...);
            return this->data_[calculate_idx(idxs)]; 
        }

        const T& operator[](const vector<size_t>& indices) const {
            return this->data_[calculate_idx(indices)]; 
        }

        /**
         * @brief Advanced indexing using a combination of integers, strings, and slices.
         * 
         * This function allows for flexible indexing into the tensor, similar to Python's
         * advanced indexing. It supports integer indices, string-based slices, and the ellipsis
         * ("...") for automatic dimension completion. The function expands slices and handles
         * ellipsis to generate the appropriate sub-tensor.
         * 
         * @param indices A vector of indices where each index can be an integer, a string
         *                representing a slice, or a special ellipsis ("...").
         * @return A new tensor that is indexed from the current tensor according to the given indices.
         * 
         * @throw std::invalid_argument if an index type is invalid or if more than one ellipsis is used.
         */
        using IndexType = variant<size_t, string, Slice>;
        Tensor<T> index(const vector<IndexType>& indices) const {
            vector<vector<size_t>> expanded_indices;
            
            // Handle ellipsis and expand slices
            // cout << "Start expanding indices" << endl;
            for (size_t i = 0; i < indices.size(); ++i) {
                const auto& idx = indices[i];

                if (auto str_idx = get_if<string>(&idx)) {
                    Slice slice = Slice::parse(*str_idx);
                    expanded_indices.push_back(apply_slice(slice, this->shapes_[i]));
                } 
                else if (auto int_idx = get_if<size_t>(&idx)) {
                    expanded_indices.push_back({normalize_index(*int_idx, this->shapes_[i])});
                } 
                else if (auto slice_idx = get_if<Slice>(&idx)) {
                    expanded_indices.push_back(apply_slice(*slice_idx, this->shapes_[i]));
                }
                else {
                    throw std::invalid_argument("Invalid index type");
                }
            }
            
            // Calculate new dimensions
            vector<size_t> new_dims;
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

            // cout << "Start printing new_dims" << endl;
            // cout << "new_dims size: " << new_dims.size() << endl;
            // for (size_t i = 0; i < new_dims.size(); ++i) {
            //     cout << new_dims[i] << " ";
            // }
            
            // Create result tensor
            Tensor<T> result(new_dims, static_cast<T>(0));
            
            // Fill result tensor
            vector<size_t> current_indices(expanded_indices.size());
            vector<size_t> result_indices;
            
            // Recursive lambda to fill result tensor
            function<void(size_t)> fill_tensor = [&](size_t depth) {
                if (depth == expanded_indices.size()) {
                    result_indices.clear();
                    for (int i = 0; i < expanded_indices.size(); ++i) {
                        if (expanded_indices[i][0] != -1 && expanded_indices[i].size() > 1) {
                            result_indices.push_back(current_indices[i]);
                        }
                    }
                    
                    vector<size_t> original_indices;
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