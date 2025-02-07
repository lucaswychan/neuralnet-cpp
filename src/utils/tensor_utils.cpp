#include "tensor_utils.hpp"
#include "tensor.hpp"

Slice Slice::parse(const string& slice_str) {
    Slice result;
    istringstream ss(slice_str);
    string token;
    vector<int> values;
    vector<uint8_t> is_empty;
    
    while (getline(ss, token, ':')) {
        // cout << "token: " << token << endl;
        if (token.empty()) {
            values.push_back(-1);
            is_empty.push_back(1);
        } else {
            values.push_back(stoi(token));
            is_empty.push_back(0);
        }
    }

    // handle the case of ":"
    if (values.size() == 1) {
        values.push_back(-1);
        is_empty.push_back(1);
    }

    result.start = is_empty[0] == 1 ? 0 : values[0];
    result.stop = is_empty[1] == 1 ? INT_MAX : values[1];

    if (values.size() == 3) {
        result.step = is_empty[2] == 1 ? 1 : values[2];
    }
    
    // cout << "start: " << result.start << " stop: " << result.stop << " step: " << result.step << endl;
    return result;
}

// Helper function to convert negative indices to positive
size_t normalize_index(int idx, size_t dim_size) {
    if (idx < 0) idx += dim_size;
    if (idx < 0 || idx >= dim_size) {
        throw std::out_of_range("Index out of bounds after index normalization with index " + to_string(idx));
    }
    return idx;
}

// Helper function to apply slice to a dimension
vector<size_t> apply_slice(const Slice& slice, size_t dim_size) {
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

vector<size_t> linear_to_multi_idxs(size_t idx, const vector<size_t>& shape) {
    vector<size_t> indices(shape.size());
    for (int64_t i = shape.size() - 1; i >= 0; --i) {
        indices[i] = idx % shape[i];
        idx /= shape[i];
    }
    return indices;
}