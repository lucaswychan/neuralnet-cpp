#include "mnist.hpp"

bool MNIST::load_data(const string& image_file, const string& label_file) {
    if (!this->read_images(image_file) || !this->read_labels(label_file)) {
        return false;
    }

    // Calculate number of batches (ceiling division)
    this->num_batches = (this->images.size() + this->batch_size - 1) / this->batch_size;
    
    if (this->verbose) {
        cout << "Dataset loaded successfully:\n";
        cout << "Total images: " << images.size() << "\n";
        cout << "Batch size: " << this->batch_size << "\n";
        cout << "Number of batches: " << num_batches << "\n";
    }
    
    return true;
}


Batch MNIST::get_next_batch() {
    Batch batch;
    size_t start_idx = this->current_batch_idxs * this->batch_size;
    size_t end_idx = std::min(start_idx + this->batch_size, this->images.size());
    size_t actual_batch_size = end_idx - start_idx;

    batch.batch_data.resize(actual_batch_size);
    batch.batch_labels.resize(actual_batch_size);

    // Copy data for this batch
    for (size_t i = 0; i < actual_batch_size; i++) {
        batch.batch_data[i] = this->images[start_idx + i];
        batch.batch_labels[i] = this->labels[start_idx + i]; // the labels are not one-hot encoded
    }

    // Update batch index
    this->current_batch_idxs = (this->current_batch_idxs + 1) % this->num_batches;

    return batch;
}

template<typename T>
T MNIST::reverse_int(T value) {
    T result = 0;
    for(size_t i = 0; i < sizeof(T); i++) {
        result = (result << 8) | ((value >> (i * 8)) & 0xFF);
    }
    return result;
}

double MNIST::normalize(double value) {
    double scaled = value / 255.0f;
    return (scaled - this->MNIST_MEAN) / this->MNIST_STD;
}

bool MNIST::read_images(const string& path) {
    ifstream file(path, ios::binary);
    if(!file.is_open()) {
        cerr << "Failed to open file: " << path << endl;
        return false;
    }

    int32_t magic, numImages, numRows, numCols;

    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
    file.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
    file.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));

    magic = reverse_int(magic);
    numImages = reverse_int(numImages);
    numRows = reverse_int(numRows);
    numCols = reverse_int(numCols);

    this->images.resize(numImages, vector<double>(numRows * numCols));

    for(int i = 0; i < numImages; i++) {
        for(int j = 0; j < numRows * numCols; j++) {
            unsigned char temp = 0;
            file.read(reinterpret_cast<char*>(&temp), sizeof(temp));
            this->images[i][j] = normalize(static_cast<double>(temp));
        }
    }

    return true;
}

bool MNIST::read_labels(const string& path) {
    ifstream file(path, ios::binary);
    if(!file.is_open()) {
        cerr << "Failed to open file: " << path << endl;
        return false;
    }

    int32_t magic, numLabels;

    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));

    magic = this->reverse_int(magic);
    numLabels = this->reverse_int(numLabels);

    labels.resize(numLabels);

    for(int i = 0; i < numLabels; i++) {
        unsigned char temp;
        file.read(reinterpret_cast<char*>(&temp), 1);
        this->labels[i] = static_cast<int>(temp);
    }

    return true;
}

tuple<Tensor<>, Tensor<>> Batch::to_tensor() {
    Tensor<> data = this->batch_data;
    Tensor<> labels = this->batch_labels;

    return make_tuple(data, labels);
}