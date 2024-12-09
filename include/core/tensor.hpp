#include <iostream>
#include <vector>
#include <initializer_list>
using namespace std;


template <typename T = double>
class Tensor {
    private:
        vector<size_t> shapes_;
        size_t size_;
        vector<T> data_; // data is stored as a 1D vector

        size_t calculateIndex(const vector<size_t>& idxs) const {
            size_t index = 0;
            size_t multiplier = 1;
            for (size_t i = this->ndim(); i-- > 0;) {
                index += idxs[i] * multiplier;
                multiplier *= this->shapes_[i];
            }
            return index;
        }

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
                cout << "]" << endl;;
            }
        }

        // Declare friendship
        template<typename U>
        friend class TensorView;

        // TensorView class to provide a reference-like view into tensor data
        template<typename U>
        class TensorView {
            Tensor<U>& tensor_;  // Reference to the original tensor
            vector<size_t> indices_;
            vector<size_t> strides_;
            size_t slice_dim_;
            size_t size_;
            
        public:
            TensorView(Tensor<U>& tensor, const vector<size_t>& indices, const vector<size_t>& strides, size_t slice_dim, size_t size)
                : tensor_(tensor)
                , indices_(indices)
                , strides_(strides)
                , slice_dim_(slice_dim)
                , size_(size) {}
            
            U& operator[](size_t idx) {
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
            
            size_t size() const { return this->size_; }

            // Iterator support
            U* begin() { return &operator[](0); }
            U* end() { return &operator[](this->size_); }
            const U* begin() const { return &operator[](0); }
            const U* end() const { return &operator[](this->size_); }
        };

    public:
        Tensor() = default;

        // Scaler
        Tensor(const T& value) {
            this->shapes_ = vector<size_t>(1, 1);
            this->data_ = vector<T>(1, value);
        }

        // 1D tensor
        Tensor(const initializer_list<T>& data) : size_(data.size()), data_(data.begin(), data.end()) {
            this->shapes_ = vector<size_t>(1, data.size());
        }

        // 2D tensor
        Tensor(const initializer_list<initializer_list<T>>& data_2d) {
            const size_t n = data_2d.size(), m = data_2d.begin()->size();
            this->size_ = n * m;

            this->shapes_ = vector<size_t> { n, m };

            for (const initializer_list<T>& row : data_2d) {
                this->data_.insert(this->data_.end(), row.begin(), row.end());
            }
        }

        // 3D tensor
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

        // certin value initializer
        Tensor(const vector<size_t>& shape, const T& value) {
            this->shapes_ = shape;
            this->size_ = 1;
            for (const size_t& s : shape) {
                this->size_ *= s;
            }

            this->data_.resize(this->size_, value);
        }

        Tensor(const Tensor<T>& other) {
            // already overloading operator=
            *this = other;
        }

        Tensor<T> add(const Tensor& other) const {
            if (other.shapes_ != this->shapes_) {
                throw runtime_error("Shape mismatch");
            }

            // default initialization (the default parameter is the first element of the other tensor)
            Tensor<T> result(this->shapes_, other.data_[0]);

            for (size_t i = 0; i < this->size_; i++) {
                result.data_[i] = this->data_[i] + other.data_[i];
            }
            return result;
        }

        Tensor<T> sub(const Tensor<T>& other) const {
            if (other.shapes_ != this->shapes_) {
                throw runtime_error("Shape mismatch");
            }

            // default initialization (the default parameter is the first element of the other tensor)
            Tensor<T> result(this->shapes_, other.data_[0]);

            for (size_t i = 0; i < this->size_; i++) {
                result.data_[i] = this->data_[i] - other.data_[i];
            }
            return result;
        }

        Tensor<T> mul(const Tensor<T>& other) const {
            if (other.shapes_ != this->shapes_) {
                throw runtime_error("Shape mismatch");
            }

            // default initialization (the default parameter is the first element of the other tensor)
            Tensor<T> result(this->shapes_, other.data_[0]);

            for (size_t i = 0; i < this->size_; i++) {
                result.data_[i] = this->data_[i] * other.data_[i];
            }
            return result;
        }

        // scaler multiplication
        Tensor<T> mul(const T& scaler) const {
            // default initialization (the default parameter is the first element of this tensor)
            Tensor<T> result(this->shapes_, this->data_[0]);

            for (size_t i = 0; i < this->size_; i++) {
                result.data_[i] = this->data_[i] * scaler;
            }
            return result;
        }

        Tensor<T> matmul(const Tensor<T>& other) const {
            // this->data_ R^n x m, other.data_ R^m x p
            const size_t n = this->shapes_[0], m = this->shapes_[1], p = other.shapes_[1];

            if (m != other.shapes_[0]) {
                throw runtime_error("Shape mismatch");
            }
            if (this->ndim() != 2 || other.ndim() != 2) {
                throw runtime_error("Only 2D tensors are supported for matrix multiplication");
            }

            Tensor<T> result({ n, p }, other.data_[0]);

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

        Tensor<T> transpose() const {
            if (this->ndim() > 2) {
                throw runtime_error("Only 1D and 2D tensors are supported for transpose");
            }

            // We want 1D vector to be transposed from 1xn to nx1
            const size_t n = (this->ndim() == 2)? this->shapes_[0] : 1;
            const size_t m = (this->ndim() == 2)? this->shapes_[1] : this->shapes_[0];

            Tensor result({m, n}, this->data_[0]);

            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    result.data_[j * n + i] = data_[i * m + j];
                }
            }
            return result;
        }

        Tensor<T> flatten() const {
            this->shapes_ = vector<size_t> { 1, this->size_ };
            return *this;
        }
        

        Tensor<T>& assign(const Tensor<T>& other) {
            if (this == &other) return *this;

            this->shapes_ = other.shapes_;
            this->size_ = other.size_;
            this->data_ = other.data_;
            return *this;
        }

        Tensor<T> abs() const {
            Tensor<T> result(this->shapes_, this->data_[0]);

            for (size_t i = 0; i < this->size_; i++) {
                result.data_[i] = std::abs(this->data_[i]);
            }
            return result;
        }

        // Get a vector representing a slice (row/column/etc)
        // Slice with reference return
        template<size_t Dim>
        TensorView<T> slice(size_t index) {
            if (index >= this->shapes_[Dim]) throw out_of_range("Index out of bounds");

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

        // Convenience methods for common operations
        inline TensorView<T> row(size_t index) {
            return this->slice<0>(index);
        }

        inline TensorView<T> col(size_t index) {
            return this->slice<1>(index);
        }
        
        // Get the dimension of the tensor
        inline size_t ndim() const {
            return this->shapes_.size();
        }

        inline const vector<size_t>& shapes() const { return this->shapes_; }

        inline void print() const { printRecursive(0, 0); }

        // operators overloading
        inline Tensor<T> operator+(const Tensor<T>& other) const { return this->add(other); }
        inline Tensor<T> operator-(const Tensor<T>& other) const { return this->sub(other); }
        inline Tensor<T> operator*(const Tensor<T>& other) const { return this->mul(other); }
        inline Tensor<T> operator*(const T& scaler) const { return this->mul(scaler); }
        inline Tensor<T>& operator=(const Tensor<T>& other) { return this->assign(other); }

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

        // lvalue operator overloading
        template<typename... Indices>
        T& operator[](Indices... indices) {
            // ((cout << ',' << std::forward<Indices>(indices)), ...);
            // cout << endl;

            // Convert variadic arguments to vector
            vector<size_t> idxs({static_cast<size_t>(indices)...});

            // Check bounds
            for (size_t i = 0; i < idxs.size(); ++i) {
                if (idxs[i] < 0 || idxs[i] >= this->shapes_[i]) {
                    throw out_of_range("Tensor: Index out of bounds");
                }
            }

            return this->data_[calculateIndex(idxs)]; 
        }

        
};
