#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <initializer_list>
#include <cassert>
using namespace std;


template <typename T>
class Tensor {
    public:
        Tensor() = default;

        // Scaler
        Tensor(const T& value) {
            this->shapes_ = vector<size_t>(1, 1);
            this->data_ = vector<T>(1, value);
        }

        // 1D tensor
        Tensor(const initializer_list<T>& data) : dim_(1), size_(data.size()), data_(data.begin(), data.end()) {
            this->shapes_ = vector<size_t>(1, data.size());
        }

        // 2D tensor
        Tensor(const initializer_list<initializer_list<T>>& data_2d) : dim_(2) {
            const size_t n = data_2d.size(), m = data_2d.begin()->size();
            this->size_ = n * m;

            this->shapes_ = vector<size_t> { n, m };

            for (const initializer_list<T>& row : data_2d) {
                this->data_.insert(this->data_.end(), row.begin(), row.end());
            }
        }

        // 3D tensor
        Tensor(const initializer_list<initializer_list<initializer_list<T>>>& data_3d) : dim_(3) {
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
            this->data_ = vector<T>(calculateSize(shape), value);
        }

        Tensor<T> add(const Tensor& other) const {
            if (other.shape_ != this->shapes_) {
                throw std::runtime_error("Shape mismatch");
            }

            // default initialization (the default parameter is the first element of the other tensor)
            Tensor<T> result(this->shapes_, other.data_[0]);

            for (size_t i = 0; i < this->size_; i++) {
                result.data_[i] = this->data_[i] + other.data_[i];
            }
            return result;
        }

        Tensor<T> sub(const Tensor<T>& other) const {
            if (other.shape_ != this->shapes_) {
                throw std::runtime_error("Shape mismatch");
            }

            // default initialization (the default parameter is the first element of the other tensor)
            Tensor<T> result(this->shapes_, other.data_[0]);

            for (size_t i = 0; i < this->size_; i++) {
                result.data_[i] = this->data_[i] - other.data_[i];
            }
            return result;
        }

        Tensor<T> mul(const Tensor<T>& other) const {
            if (other.shape_ != this->shapes_) {
                throw std::runtime_error("Shape mismatch");
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
                throw std::runtime_error("Shape mismatch");
            }
            if (this->shapes_.size() != 2 || other.shapes_.size() != 2) {
                throw std::runtime_error("Only 2D tensors are supported for matrix multiplication");
            }

            Tensor<T> result({ n, p }, other.data_[0]);

            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < p; j++) {
                    for (size_t k = 0; k < m; k++) {
                        result.data_[i * p + j] += this->data_[i * n + k] * other.data_[k * p + j];;
                    }
                }
            }
            
            return result;
        }

        Tensor<T> transpose() const {
            if (this->shapes_.size() != 2) {
                throw std::runtime_error("Only 2D tensors are supported for transpose");
            }

            const size_t n = this->shapes_[0], m = this->shapes_[1];

            Tensor result({m, n}, this->data_[0]);

            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    result.data_[j * n + i] = data_[i * m + j];
                }
            }
            return result;
        }

        Tensor<T> flatten() const {
            this->dim_ = 1;
            this->shapes_ = { this->size_ };
            return *this;
        }
        
        inline void print() const { printRecursive(0, 0); }

        // operators overloading
        inline Tensor<T> operator+(const Tensor<T>& other) const { return this->add(other); }
        inline Tensor<T> operator-(const Tensor<T>& other) const { return this->sub(other); }
        inline Tensor<T> operator*(const Tensor<T>& other) const { return this->mul(other); }
        inline Tensor<T> operator*(const T& scaler) const { return this->mul(scaler); }

        Tensor<T>& operator=(const Tensor<T>& other);
        Tensor<T> operator()(const vector<size_t>& indices);
    
    private:
        vector<size_t> shapes_;

        size_t dim_;
        size_t size_;

        // data is stored as a 1D vector
        vector<T> data_;

        size_t calculateSize(const vector<size_t>& shape) const {
            size_t size = 1;
            for (const size_t& dim : shape) {
                size *= dim;
            }
            return size;
        }

        size_t calculateIndex(const vector<size_t>& indices) const {
            assert(indices.size() == this->shapes_.size());

            size_t index = 0;
            size_t multiplier = 1;
            for (size_t i = this->shapes_.size(); i-- > 0;) {
                index += indices[i] * multiplier;
                multiplier *= this->shapes_[i];
            }
            return index;
        }

        void printRecursive(size_t dim, size_t offset) const {
            if (dim == this->shapes_.size() - 1) { // Last dimension
                cout << "[";
                for (size_t i = 0; i < this->shapes_[dim]; ++i) {
                    cout << data_[offset + i];
                    if (i < this->shapes_[dim] - 1) cout << ", ";
                }
                cout << "]";
            } else {
                cout << "[";
                size_t stride = 1;
                for (size_t i = dim + 1; i < this->shapes_.size(); ++i) {
                    stride *= this->shapes_[i];
                }
                for (size_t i = 0; i < this->shapes_[dim]; ++i) {
                    printRecursive(dim + 1, offset + i * stride);
                    if (i < this->shapes_[dim] - 1) cout << ", " << endl;
                }
                cout << "]";
            }
        }
};
