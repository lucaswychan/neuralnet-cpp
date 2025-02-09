#pragma once
#include "tensor_utils.hpp"
using namespace std;

template <typename T = double>
class Tensor
{
private:
    shared_ptr<vector<T>> data_ = nullptr; // data is stored as a 1D vector // shared_ptr is used to avoid copying data
    vector<size_t> shapes_;                // store the dimensions of the tensor
    vector<size_t> strides_;               // store the strides of the tensor
    size_t offset_ = 0;                    // offset for slicing
    mutable int64_t size_ = -1;            // it can be changed by const member functions (in size() function)

    // Helper function to calculate the index in the 1D vector for a given set of indices expressed in the form of N-D vector
    size_t calculate_idx(const vector<size_t> &idxs) const
    {
        size_t idx = this->offset_;
        for (size_t i = 0; i < idxs.size(); ++i)
        {
            idx += idxs[i] * this->strides_[i];
        }
        return idx;
    }

    // Helper function for printing since we don't know the number of dimensions
    void print_recursive_impl(size_t dim, size_t offset, int indent = 0) const
    {
        const string indent_str(indent, ' ');

        // Handle empty dimensions
        if (this->shapes_[dim] == 0)
        {
            cout << indent_str << "[]";
            return;
        }

        cout << indent_str << "[";

        if (dim == this->ndim() - 1)
        { // Last dimension
            for (size_t i = 0; i < this->shapes_[dim]; ++i)
            {
                cout << (*this->data_)[offset + i * this->strides_[dim]];
                if (i < this->shapes_[dim] - 1)
                    cout << ", ";
            }
        }
        else
        {
            cout << "\n";
            for (size_t i = 0; i < this->shapes_[dim]; ++i)
            {
                print_recursive_impl(dim + 1, offset + i * this->strides_[dim], indent + 2);
                if (i < this->shapes_[dim] - 1)
                    cout << ",\n";
            }
            cout << "\n"
                 << indent_str;
        }
        cout << "]";
    }

    // Helper function for operator[] overloading
    template <typename... Indices>
    const vector<size_t> get_idxs(Indices... indices) const
    {
        // Convert variadic arguments to vector
        vector<int64_t> idxs({static_cast<int64_t>(indices)...});
        vector<size_t> normalized_idxs;

        // for better performance, reserve the size of the vector
        normalized_idxs.reserve(idxs.size());

        // Check bounds
        for (size_t i = 0; i < idxs.size(); ++i)
        {
            size_t normalized_idx = normalize_index(idxs[i], this->shapes_[i]);
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
    Tensor<U> reduce_impl(ReduceOp op) const
    {
        const size_t ndim = this->ndim();

        if (ndim > 2)
        {
            throw std::runtime_error("Only 1D and 2D tensors are supported for reduce");
        }

        // Determine tensor dimensions
        const size_t num_rows = (ndim == 2) ? this->shapes_[0] : 1;
        const size_t num_cols = (ndim == 2) ? this->shapes_[1] : this->shapes_[0];

        vector<U> result(num_rows);

        for (size_t i = 0; i < num_rows; ++i)
        {
            // Calculate base offset for current row
            size_t row_offset = this->offset_;
            if (ndim == 2)
            {
                row_offset += i * this->strides_[0];
            }

            T extreme_val = (*this->data_)[row_offset];
            size_t extreme_idx = 0;

            // Process elements using stride-aware indexing
            for (size_t j = 1; j < num_cols; ++j)
            {
                size_t elem_offset = row_offset;
                if (ndim == 2)
                {
                    elem_offset += j * this->strides_[1];
                }
                else
                {
                    elem_offset += j * this->strides_[0];
                }

                bool update = false;
                switch (op)
                {
                case ReduceOp::MAX:
                case ReduceOp::ARGMAX:
                    update = (*this->data_)[elem_offset] > extreme_val;
                    break;
                case ReduceOp::MIN:
                case ReduceOp::ARGMIN:
                    update = (*this->data_)[elem_offset] < extreme_val;
                    break;
                }

                if (update)
                {
                    extreme_val = (*this->data_)[elem_offset];
                    extreme_idx = j;
                }
            }

            switch (op)
            {
            case ReduceOp::MAX:
            case ReduceOp::MIN:
                result[i] = static_cast<U>(extreme_val);
                break;
            case ReduceOp::ARGMAX:
            case ReduceOp::ARGMIN:
                result[i] = static_cast<U>(extreme_idx);
                break;
            }
        }

        return Tensor<U>(result);
    }

    Tensor<T> arithmetic_operation_impl(ArithmeticOp op, const Tensor<T> &other) const
    {
        if (other.shapes_ != this->shapes_)
        {
            throw runtime_error("Shape mismatch in arithmetic operation");
        }

        size_t ndim = this->ndim();

        Tensor<T> result(this->shapes_, static_cast<T>(0));

        // Precompute result's contiguous strides for index calculation
        const vector<size_t> &result_strides = result.strides_;

        for (size_t i = 0; i < this->size(); i++)
        {
            auto [a_offset, b_offset] = calculate_tensors_offsets(i, ndim, result_strides, other);

            switch (op)
            {
            case ArithmeticOp::ADD:
                (*result.data_)[i] = (*this->data_)[a_offset] + (*other.data_)[b_offset];
                break;
            case ArithmeticOp::SUB:
                (*result.data_)[i] = (*this->data_)[a_offset] - (*other.data_)[b_offset];
                break;
            case ArithmeticOp::MUL:
                (*result.data_)[i] = (*this->data_)[a_offset] * (*other.data_)[b_offset];
                break;
            case ArithmeticOp::DIV:
                (*result.data_)[i] = (*this->data_)[a_offset] / (*other.data_)[b_offset];
                break;
            }
        }
        return result;
    }

    // Helper function to cacluate the stride of the tensor
    void calculate_strides()
    {
        this->strides_.resize(this->ndim(), 0);
        vector<size_t> strides(this->ndim());

        int64_t stride = 1;

        for (int64_t i = this->ndim() - 1; i >= 0; --i)
        {
            this->strides_[i] = stride;
            stride *= this->shapes_[i];
        }
    }

    std::tuple<size_t, size_t> calculate_tensors_offsets(const size_t idx, const size_t ndim, const vector<size_t> &result_strides, const Tensor<T> &other) const
    {
        vector<size_t> indices(ndim);

        size_t remaining = idx;

        for (int dim = 0; dim < ndim; ++dim)
        {
            indices[dim] = remaining / result_strides[dim];
            remaining %= result_strides[dim];
        }

        // Calculate offsets using original tensors' strides
        size_t a_offset = this->offset_;
        size_t b_offset = other.offset_;

        for (int dim = 0; dim < ndim; ++dim)
        {
            a_offset += indices[dim] * this->strides_[dim];
            b_offset += indices[dim] * other.strides_[dim];
        }

        return {a_offset, b_offset};
    }

    // Declare friendship so that TensorView can access private members of Tensor
    template <typename U, typename V>
    friend Tensor<V> dtype_impl(const Tensor<U> &tensor);

public:
    Tensor() = default;

    // Helper to recursively flatten nested vectors and compute shapes
    template <typename V>
    void flatten_vector(const std::vector<V> &vec, size_t depth = 0)
    {
        // Add current level's size to shapes
        if (depth == this->shapes_.size())
        {
            // First encounter with this depth: record size
            this->shapes_.push_back(vec.size());
        }
        else
        {
            // Verify size matches the existing dimension
            if (vec.size() != this->shapes_[depth])
            {
                throw std::invalid_argument("Inconsistent shape at depth " + std::to_string(depth));
            }
        }

        if constexpr (is_vector<V>::value)
        {
            // Ensure nested vectors have consistent sizes at this level
            if (!vec.empty())
            {
                size_t expected_size = vec[0].size();
                for (const auto &elem : vec)
                {
                    if (elem.size() != expected_size)
                    {
                        throw std::invalid_argument("Inconsistent shape in nested vectors");
                    }
                }
            }

            // Recurse into nested vectors
            for (const auto &elem : vec)
            {
                flatten_vector(elem, depth + 1);
            }
        }
        else
        {
            // Ensure leaf elements match the Tensor's data type
            // static_assert(std::is_same_v<V, T>, "Element type must match Tensor type");
            this->data_->reserve(this->data_->size() + vec.size());
            for (const auto &elem : vec)
            {
                this->data_->emplace_back(static_cast<T>(elem));
            }
        }
    }

    // Constructor for nested vectors
    template <typename V>
    Tensor(const std::vector<V> &input)
    {
        this->data_ = make_shared<vector<T>>();
        flatten_vector(input);
        this->calculate_strides();
    }

    // // Recursive helper to process nested initializer lists
    // template<typename U>
    // void flatten_list(const std::initializer_list<U>& list, size_t depth = 0) {
    //     // Handle the current dimension
    //     if (depth == shapes_.size()) {
    //         // First encounter with this depth: record size
    //         shapes_.push_back(list.size());
    //     } else {
    //         // Verify size matches the existing dimension
    //         if (list.size() != shapes_[depth]) {
    //             throw std::invalid_argument("Inconsistent shape at depth " + std::to_string(depth));
    //         }
    //     }

    //     // Recurse or add data
    //     if constexpr (is_list<U>::value) {
    //         // Process nested lists
    //         for (const auto& elem : list) {
    //             flatten_list(elem, depth + 1);
    //         }
    //     } else {
    //         // Ensure element type matches Tensor type
    //         // static_assert(std::is_same_v<U, T>, "Element type must match Tensor type");
    //         for (const auto& elem : list) {
    //             data_.push_back(static_cast<T>(elem));
    //         }
    //     }
    // }

    // Scaler constructor
    Tensor(const T &value)
    {
        this->shapes_ = vector<size_t>{1};
        this->data_ = make_shared<vector<T>>(1, value);
        this->calculate_strides();
    }

    // 1D tensor constructor
    Tensor(const initializer_list<T> &data_1d)
    {
        this->data_ = make_shared<vector<T>>(data_1d.begin(), data_1d.end());
        this->shapes_ = vector<size_t>{data_1d.size()};
        this->calculate_strides();
    }

    // 2D tensor constructor
    Tensor(const initializer_list<initializer_list<T>> &data_2d)
    {
        const size_t n = data_2d.size(), m = data_2d.begin()->size();

        this->shapes_ = vector<size_t>{n, m};

        this->data_ = make_shared<vector<T>>();
        this->data_->reserve(n * m); // Optimize memory allocation

        for (const initializer_list<T> &row : data_2d)
        {
            this->data_->insert(this->data_->end(), row.begin(), row.end());
        }
        this->calculate_strides();
    }

    // 3D tensor constructor
    Tensor(const initializer_list<initializer_list<initializer_list<T>>> &data_3d)
    {
        const size_t n = data_3d.size(), m = data_3d.begin()->size(), l = data_3d.begin()->begin()->size();

        this->shapes_ = vector<size_t>{n, m, l};

        this->data_ = make_shared<vector<T>>();
        this->data_->reserve(n * m * l); // Optimize memory allocation

        for (const initializer_list<initializer_list<T>> &matrix : data_3d)
        {
            for (const initializer_list<T> &row : matrix)
            {
                this->data_->insert(this->data_->end(), row.begin(), row.end());
            }
        }
        this->calculate_strides();
    }

    // 4D tensor constructor
    Tensor(const initializer_list<initializer_list<initializer_list<initializer_list<T>>>> &data_4d)
    {
        const size_t n = data_4d.size(), m = data_4d.begin()->size(), l = data_4d.begin()->begin()->size(), k = data_4d.begin()->begin()->begin()->size();

        this->shapes_ = vector<size_t>{n, m, l, k};

        this->data_ = make_shared<vector<T>>();
        this->data_->reserve(n * m * l * k); // Optimize memory allocation

        for (const initializer_list<initializer_list<initializer_list<T>>> &tensor : data_4d)
        {
            for (const initializer_list<initializer_list<T>> &matrix : tensor)
            {
                for (const initializer_list<T> &row : matrix)
                {
                    this->data_->insert(this->data_->end(), row.begin(), row.end());
                }
            }
        }
        this->calculate_strides();
    }

    // certin value constructor
    Tensor(const vector<size_t> &shape, const T &value)
    {
        this->shapes_ = shape;
        size_t size = 1;
        for (const size_t &dim : shape)
        {
            size *= dim;
        }

        this->data_ = make_shared<vector<T>>(size, value);
        this->calculate_strides();
    }

    // copy constructor
    Tensor(const Tensor<T> &other)
    {
        // already overload the = operator
        *this = other;
    }

    // template <typename V>
    // Tensor(const Tensor<V> &other)
    // {
    //     // use dtype function to convert the data type
    //     *this = other.dtype<>();
    // }

    // Add two tensors with same shape, element-wise
    inline Tensor<T> add(const Tensor &other) const
    {
        return arithmetic_operation_impl(ArithmeticOp::ADD, other);
    }

    // Subtract two tensors with same shape, element-wise
    inline Tensor<T> sub(const Tensor<T> &other) const
    {
        return arithmetic_operation_impl(ArithmeticOp::SUB, other);
    }

    // Multiply two tensors with same shape, element-wise
    inline Tensor<T> mul(const Tensor<T> &other) const
    {
        return arithmetic_operation_impl(ArithmeticOp::MUL, other);
    }

    // Divide two tensors with same shape, element-wise
    inline Tensor<T> div(const Tensor<T> &other) const
    {
        return arithmetic_operation_impl(ArithmeticOp::DIV, other);
    }

    // Multiply all elements of tensor with the given scaler
    Tensor<T> mul(const T &scaler) const
    {
        Tensor<T> result(this->shapes_, static_cast<T>(0));

        for (size_t i = 0; i < this->size(); i++)
        {
            (*result.data_)[i] = (*this->data_)[i] * scaler;
        }
        return result;
    }

    /**
     * Matrix multiplication of two tensors.
     *
     * The two tensors must have at least two dimensions. The leading dimensions (all except last two) must be equal.
     * The last two dimensions must match the matrix multiplication dimensions.
     * For example, if the first tensor has shape [a, b, n, m] and the second tensor has shape [a, b, m, p], the result will have shape [a, b, n, p].
     *
     * The result is a tensor with the leading dimensions of the first tensor and the matrix multiplication result as the last two dimensions.
     *
     * The total number of batches is the product of the leading dimensions.
     *
     * The matrix multiplication is performed batched, i.e., for each batch, a matrix multiplication is performed.
     *
     * @param other The tensor to multiply with.
     * @return The result of the matrix multiplication.
     */
    Tensor<T> matmul(const Tensor<T> &other) const
    {
        // Ensure both tensors have at least 2 dimensions
        size_t A_ndim = this->ndim(), B_ndim = other.ndim();

        if (A_ndim < 2 || B_ndim < 2)
        {
            throw std::runtime_error("Tensors must have at least 2 dimensions for matrix multiplication");
        }

        // Check leading dimensions (all except last two) are equal
        const size_t A_leading_ndim = A_ndim - 2;
        const size_t B_leading_ndim = B_ndim - 2;

        if (A_leading_ndim != B_leading_ndim)
        {
            throw std::runtime_error("Number of leading dimensions must match");
        }

        vector<size_t> A_leading_shape(this->shapes_.begin(), this->shapes_.end() - 2);
        vector<size_t> B_leading_shape(other.shapes_.begin(), other.shapes_.end() - 2);

        if (A_leading_shape != B_leading_shape)
        {
            throw invalid_argument("Batch dimensions must match");
        }

        // Extract matrix dimensions
        const size_t n = this->shapes_[A_ndim - 2];
        const size_t m = this->shapes_[A_ndim - 1];
        const size_t m_other = other.shapes_[B_ndim - 2];
        const size_t p = other.shapes_[B_ndim - 1];

        if (m != m_other)
        {
            throw std::invalid_argument("Matrix dimension mismatch: last dimension of first tensor must match second last of second tensor");
        }

        // Determine result shape: leading dimensions + [n, p]
        vector<size_t> result_shapes = A_leading_shape;
        result_shapes.push_back(n);
        result_shapes.push_back(p);

        Tensor<T> result(result_shapes, static_cast<T>(0));

        // Compute total number of batches (product of leading dimensions)
        // may be we can use divisoin in stride to have O(1) time
        size_t total_batches = 1;
        for (const size_t &dim : A_leading_shape)
        {
            total_batches *= dim;
        }

        for (size_t batch = 0; batch < total_batches; ++batch)
        {
            // Get multi_dimensional indices for this batch
            vector<size_t> indices = linear_to_multi_idxs(batch, A_leading_shape);

            // Compute offsets for A, B, and result
            size_t A_offset = this->offset_;
            size_t B_offset = other.offset_;
            size_t result_offset = 0;

            for (size_t i = 0; i < A_leading_ndim; ++i)
            {
                A_offset += indices[i] * this->strides_[i];
                B_offset += indices[i] * other.strides_[i];
                result_offset += indices[i] * result.strides_[i];
            }

            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j < p; ++j)
                {
                    T sum = static_cast<T>(0);

                    for (size_t k = 0; k < m; ++k)
                    {
                        // Calculate offsets in A and B
                        size_t a_idx = A_offset +
                                       i * this->strides_[A_leading_ndim] +
                                       k * this->strides_[A_leading_ndim + 1];

                        size_t b_idx = B_offset +
                                       k * other.strides_[B_leading_ndim] +
                                       j * other.strides_[B_leading_ndim + 1];

                        sum += (*this->data_)[a_idx] * (*other.data_)[b_idx];
                    }
                    // Write to result
                    size_t out_idx = result_offset +
                                     i * result.strides_[result.ndim() - 2] +
                                     j * result.strides_.back();
                    (*result.data_)[out_idx] = sum;
                }
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

    Tensor<T> transpose(int64_t dim0 = -2, int64_t dim1 = -1) const
    {
        const size_t ndim = this->ndim();

        if (ndim == 1 && dim0 == -2 && dim1 == -1)
        {
            Tensor result = *this;
            result.reshape({this->size(), 1});
            return result;
        }

        if (dim0 == dim1)
        {
            return *this; // No-op if dimensions are the same
        }

        if (dim0 < 0)
        {
            dim0 += ndim;
        }

        if (dim1 < 0)
        {
            dim1 += ndim;
        }

        if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim)
        {
            throw out_of_range("Transpose dimensions out of range");
        }

        // Create new tensor with swapped dimensions
        Tensor<T> result = *this;
        swap(result.shapes_[dim0], result.shapes_[dim1]);
        swap(result.strides_[dim0], result.strides_[dim1]);

        // cout << "result.shapes_: " << result.shapes_[0] << " " << result.shapes_[1] << endl;
        // cout << "result.strides_: " << result.strides_[0] << " " << result.strides_[1] << endl;

        return result;
    }

    Tensor<T> permute(const initializer_list<int64_t> &dims)
    {
        size_t ndim = this->ndim();

        if (dims.size() != ndim)
        {
            throw std::invalid_argument("Number of dimensions in permutation must match tensor's number of dimensions");
        }

        unordered_set<int64_t> seen_dims;
        for (int64_t dim : dims)
        {
            if (dim < 0 || dim >= ndim)
            {
                throw out_of_range("Permute dimension out of range");
            }
            if (seen_dims.count(dim))
            {
                throw invalid_argument("Duplicate dimension in permute");
            }
            seen_dims.insert(dim);
        }

        vector<int64_t> new_shapes(ndim);
        vector<int64_t> new_strides(ndim);

        int64_t i = 0;
        for (int64_t dim : dims)
        {
            if (dim >= ndim)
            {
                throw std::out_of_range("Permutation dimension out of range");
            }

            new_shapes[i] = this->shapes_[dim];
            new_strides[i] = this->strides_[dim];
            ++i;
        }

        Tensor<T> result = *this;
        result.shapes_ = new_shapes;
        result.strides_ = new_strides;

        return result;
    }

    /// @brief Flatten the tensor into 1D in-place.
    /// @details This function only changes the shape of the tensor, and does not modify the underlying data.
    /// @post The shape of the tensor is changed to 1D, with the same elements as the original tensor.
    void flatten()
    {
        this->shapes_ = {this->size()};
        this->calculate_strides();
        return;
    }

    /// @brief Calculate the absolute value of each element in the tensor
    /// @return a new tensor with the same shape as the original, but with each element replaced by its absolute value
    Tensor<T> abs() const
    {
        Tensor<T> result = *this;

        for (size_t i = 0; i < this->size(); i++)
        {
            (*result.data_)[i] = std::abs((*this->data_)[i]);
        }

        return result;
    }

    /// @brief Filter the tensor with the given function
    /// @param func a function to test each element of the tensor. It should return true if the element passes the test
    /// @return a new tensor with the same shape as the original, but all elements that fail the test are set to 0.
    Tensor<T> filter(bool (*func)(T)) const
    {
        Tensor<T> result = *this;

        for (size_t i = 0; i < this->size(); i++)
        {
            if (func((*this->data_)[i]))
            {
                (*result.data_)[i] = (*this->data_)[i];
            }
        }

        return result;
    }

    /// @brief Perform element-wise transformation with a function
    /// @param func a function to perform element-wise transformation to the tensor
    /// @return a new tensor with the same shape as the original, but with each element transformed by the given func
    Tensor<T> map(T (*func)(T)) const
    {
        Tensor<T> result = *this;

        for (size_t i = 0; i < this->size(); i++)
        {
            (*result.data_)[i] = func((*this->data_)[i]);
        }

        return result;
    }

    /// @brief Calculate the sum of all elements in the tensor
    /// @return The sum of all elements in the tensor, regardless of the dimension
    T sum() const
    {
        T sum = static_cast<T>(0);

        for (size_t i = 0; i < this->size(); i++)
        {
            sum += (*this->data_)[i];
        }

        return sum;
    }

    /// @brief Check if all elements of two tensors are equal
    /// @param other Tensor to compare
    /// @return Tensor of integers where each element is 1 if the two tensors are equal at the same index, 0 otherwise
    Tensor<int> equal(const Tensor<T> &other) const
    {
        if (other.shapes_ != this->shapes_)
        {
            throw runtime_error("Shape mismatch");
        }

        Tensor<T> result(this->shapes_, static_cast<T>(0));
        const vector<size_t> &result_strides = result.strides_;

        for (size_t i = 0; i < this->size(); i++)
        {
            auto [a_offset, b_offset] = calculate_tensors_offsets(i, this->ndim(), result_strides, other);

            (*result.data_)[i] = (*this->data_)[a_offset] == (*other.data_)[b_offset];
        }

        return result.dtype<int>();
    }

    /// @brief Check if all elements of two tensors are equal
    /// @param other Tensor to compare
    /// @return true if all elements are equal, false otherwise
    bool compare(const Tensor<T> &other) const
    {
        if (other.shapes_ != this->shapes_)
        {
            throw runtime_error("Shape mismatch");
        }

        for (size_t i = 0; i < this->size(); i++)
        {
            auto [a_offset, b_offset] = calculate_tensors_offsets(i, this->ndim(), this->strides_, other);

            if ((*this->data_)[a_offset] != (*other.data_)[b_offset])
            {
                return false;
            }
        }
        return true;
    }

    /// @brief Reduce the tensor to the maximum value of all elements
    /// @return a tensor with a single element, the maximum of all elements in the tensor
    inline Tensor<> max() const
    {
        return reduce_impl(ReduceOp::MAX);
    }

    /// @brief Reduce the tensor to the indices of the maximum values along each row
    /// @return a tensor with indices of the maximum values for each row
    inline Tensor<size_t> argmax() const
    {
        return reduce_impl<size_t>(ReduceOp::ARGMAX);
    }

    /// @brief Reduce the tensor to the minimum value of all elements
    /// @return a tensor with a single element, the minimum of all elements in the tensor
    inline Tensor<> min() const
    {
        return reduce_impl(ReduceOp::MIN);
    }

    /// @brief Reduce the tensor to the indices of the minimum values along each row
    /// @return a tensor with indices of the minimum values for each row
    inline Tensor<size_t> argmin() const
    {
        return reduce_impl<size_t>(ReduceOp::ARGMIN);
    }

    /// @brief Convert the tensor to a tensor of a different type.
    /// @details If U is not provided, it defaults to double.
    /// @param U the type to convert to
    /// @return a tensor with the same shape and data, but with the type U
    template <typename U = double>
    Tensor<U> dtype() const
    {
        return dtype_impl<T, U>(*this);
    }

    /// @brief Reshape the tensor to the specified new shape.
    /// @details This function changes the shape of the tensor without altering the data.
    /// The total number of elements must remain the same; otherwise, an exception is thrown.
    /// @param new_shape The desired shape for the tensor.
    /// @throws runtime_error if the new shape is not compatible with the current number of elements.
    void reshape(const vector<size_t> &new_shape)
    {
        size_t new_size = 1;
        for (const size_t &dim : new_shape)
        {
            new_size *= dim;
        }

        if (new_size != this->size())
        {
            throw runtime_error("New shape must be compatible with the original shape");
        }

        this->shapes_ = new_shape;
        this->calculate_strides();

        return;
    }

    /// @brief Return a deep copy of the tensor. Actually the same as the copy constructor.
    /// @details This function will create a new tensor with the same shape and data as the current tensor.
    /// @return a new tensor which is a deep copy of the current tensor
    Tensor<T> clone() const
    {
        Tensor<T> result;

        result.shapes_ = this->shapes_;
        result.data_ = make_shared<vector<T>>(*(this->data_));
        result.calculate_strides();

        // Copy data from original tensor's view to the new contiguous storage
        for (size_t i = 0; i < this->size(); ++i)
        {
            vector<size_t> indices = linear_to_multi_idxs(i, result.shapes_);
            size_t src_offset = this->offset_;

            for (size_t dim = 0; dim < indices.size(); ++dim)
            {
                src_offset += indices[dim] * this->strides_[dim];
            }

            (*result.data_)[i] = (*this->data_)[src_offset];
        }

        return result;
    }

    vector<T> to_vector() const { return (*this->data_); }

    // Get the dimension of the tensor
    inline size_t ndim() const
    {
        return this->shapes_.size();
    }

    const size_t size() const
    {
        if (this->offset_ == 0)
        {
            return this->data_->size();
        }

        if (this->size_ != -1)
        {
            return this->size_;
        }

        this->size_ = 1;
        for (const size_t &s : this->shapes_)
        {
            this->size_ *= s;
        }

        return this->size_;
    }

    /// @brief Print the tensor to console.
    /// @details This function will print the tensor in a nested array style.
    void print() const
    {
        print_recursive_impl(0, 0, 0);
        cout << endl; // flush the output
        return;
    }

    inline const vector<size_t> &shapes() const { return this->shapes_; }

    // ========================================operators overloading========================================
    inline Tensor<T> operator+(const Tensor<T> &other) const { return this->add(other); }
    inline Tensor<T> operator-(const Tensor<T> &other) const { return this->sub(other); }
    inline Tensor<T> operator*(const Tensor<T> &other) const { return this->mul(other); }
    inline Tensor<T> operator*(const T &scaler) const { return this->mul(scaler); }
    inline bool operator==(const Tensor<T> &other) const { return this->compare(other); }

    /*
    Instead of returning a new tensor, we modify the current tensor in place.

    Besides, it is slightly different from method clone(), in which it will not modify data_ to make all the elements stored contiguously.
    */
    Tensor<T> &operator=(const Tensor<T> &other)
    {
        if (this == &other)
            return *this;

        this->shapes_ = other.shapes_;
        this->data_ = make_shared<vector<T>>(*(other.data_));
        this->strides_ = other.strides_;
        this->offset_ = other.offset_;
        this->size_ = other.size_;

        return *this;
    }

    const Tensor<T> operator+=(const Tensor<T> &other)
    {
        *this = *this + other;
        return *this;
    }

    const Tensor<T> operator-=(const Tensor<T> &other)
    {
        *this = *this - other;
        return *this;
    }

    const Tensor<T> operator*=(const Tensor<T> &other)
    {
        *this = *this * other;
        return *this;
    }

    const Tensor<T> operator*=(const T &other)
    {
        *this = *this * other;
        return *this;
    }

    // lvalue operator overloading
    template <typename... Indices>
    T &operator[](Indices... indices)
    {
        vector<size_t> idxs = this->get_idxs(indices...);
        return (*this->data_)[this->calculate_idx(idxs)];
    }

    // Using vector to index the tensor
    T &operator[](const vector<size_t> &indices)
    {
        return (*this->data_)[this->calculate_idx(indices)];
    }

    // rvalue operator overloading
    template <typename... Indices>
    const T &operator[](Indices... indices) const
    {
        vector<size_t> idxs = this->get_idxs(indices...);
        return (*this->data_)[this->calculate_idx(idxs)];
    }

    // Using vector to index the tensor
    const T &operator[](const vector<size_t> &indices) const
    {
        return (*this->data_)[this->calculate_idx(indices)];
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
    Tensor<T> index(const vector<IndexType> &indices) const
    {
        vector<vector<size_t>> expanded_indices;

        // Handle ellipsis and expand slices
        // cout << "Start expanding indices" << endl;
        for (size_t i = 0; i < indices.size(); ++i)
        {
            const auto &idx = indices[i];

            if (auto str_idx = get_if<string>(&idx))
            {
                Slice slice = Slice::parse(*str_idx);
                expanded_indices.push_back(apply_slice(slice, this->shapes_[i]));
            }
            else if (auto int_idx = get_if<size_t>(&idx))
            {
                expanded_indices.push_back({normalize_index(*int_idx, this->shapes_[i])});
            }
            else if (auto slice_idx = get_if<Slice>(&idx))
            {
                expanded_indices.push_back(apply_slice(*slice_idx, this->shapes_[i]));
            }
            else
            {
                throw std::invalid_argument("Invalid index type");
            }
        }

        // Calculate new dimensions
        vector<size_t> new_dims;
        for (const vector<size_t> &expanded_idx : expanded_indices)
        {
            if (expanded_idx[0] != -1)
            { // Not None/newaxis
                if (expanded_idx.size() > 1)
                {
                    new_dims.push_back(expanded_idx.size());
                }
            }
            else
            {
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
        function<void(size_t)> fill_tensor = [&](size_t depth)
        {
            if (depth == expanded_indices.size())
            {
                result_indices.clear();
                for (int i = 0; i < expanded_indices.size(); ++i)
                {
                    if (expanded_indices[i][0] != -1 && expanded_indices[i].size() > 1)
                    {
                        result_indices.push_back(current_indices[i]);
                    }
                }

                vector<size_t> original_indices;
                for (int i = 0; i < expanded_indices.size(); ++i)
                {
                    if (expanded_indices[i][0] != -1)
                    {
                        original_indices.push_back(expanded_indices[i][current_indices[i]]);
                    }
                }

                result[result_indices] = (*this)[original_indices];
                return;
            }

            for (int i = 0; i < expanded_indices[depth].size(); ++i)
            {
                current_indices[depth] = i;
                fill_tensor(depth + 1);
            }
        };

        fill_tensor(0);
        return result;
    }
};