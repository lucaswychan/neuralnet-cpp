#include <numeric>
#include "mlp.hpp"
#include "mnist.hpp"
#include "cross_entropy.hpp"
#include "accuracy.hpp"
#include "utils.hpp"
using namespace nn;

int main() {

    // Define the hyperparameters

    const double LR = 0.01;
    const double EPOCH = 10;
    const double BATCH_SIZE = 64;
    const double DROPOUT_P = 0.3;

    MNIST dataset(BATCH_SIZE);

    const string mnist_image_file = "../data/mnist/train-images.idx3-ubyte";
    const string mnist_label_file = "../data/mnist/train-labels.idx1-ubyte";

    // load MNIST data
    if (!dataset.loadData(mnist_image_file, mnist_label_file)) {
        cerr << "Failed to load dataset" << endl;
        return 1;
    }

    // Initialize the model
    MLP model = MLP({784, 128, 64, 10}, DROPOUT_P);

    // Define the loss function
    CrossEntropyLoss criterion = CrossEntropyLoss();

    double loss = 0.0;
    double acc = 0.0;
    vector<double> loss_list;
    vector<double> accuracy_list;

    // // Train the model
    // Example of iterating through all batches
    for (size_t e = 0; e < EPOCH; e++) {
        cout << "\nEpoch " << e + 1 << ":\n";
        dataset.reset();  // Reset batch counter at the start of each epoch
        loss_list.clear();
        accuracy_list.clear();
        
        for (size_t i = 0; i < dataset.getNumBatches(); i++) {
            auto batch = dataset.getNextBatch();
            auto [data, labels] = batch.toTensor();

            // forward propagation
            Tensor<> output = model.forward(data);

            loss = criterion.forward(output, labels);
            // cout << "After loss" << endl;
            acc = metrics::accuracy(output, labels);
            // cout << "After acc" << endl;

            accuracy_list.push_back(acc);
            loss_list.push_back(loss);

            // backward propagation
            Tensor<> grad = criterion.backward();
            model.backward(grad);
            model.update_params(LR);

            // print the training stats
            print_training_stats_line(i, loss, acc);
        }

        double total_loss = accumulate(loss_list.begin(), loss_list.end(), 0.0) / loss_list.size();
        double total_acc = accumulate(accuracy_list.begin(), accuracy_list.end(), 0.0) / accuracy_list.size() * 100;

        cout << "------------------------------------" << endl;
        cout << "Total Loss in Epoch " << e + 1 << " = " << total_loss << "" << endl;
        cout << "Total Accuracy in Epoch " << e + 1 << " = " << total_acc<< "%" << endl;
        cout << "------------------------------------" << endl;
    }

    return 0;
}
