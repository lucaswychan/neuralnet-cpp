#include <numeric>
#include <fstream>
#include "mlp.hpp"
#include "mnist.hpp"
#include "cross_entropy.hpp"
#include "adam.hpp"
#include "sgd.hpp"
#include "accuracy.hpp"
#include "utils.hpp"
using namespace nn;

int main()
{

    // Define the hyperparameters

    const float LR = 0.01;
    const float EPOCH = 1;
    const float BATCH_SIZE = 64;
    const float DROPOUT_P = 0.2;

    MNIST dataset(BATCH_SIZE);

    const string mnist_image_file = "../data/mnist/train-images.idx3-ubyte";
    const string mnist_label_file = "../data/mnist/train-labels.idx1-ubyte";

    // load MNIST data
    if (!dataset.load_data(mnist_image_file, mnist_label_file))
    {
        cerr << "Failed to load dataset" << endl;
        return 1;
    }

    // Initialize the model
    bool bias = true;
    MLP model = MLP(784, {128, 64, 10}, bias, DROPOUT_P);

    cout << "Finished model initialization" << endl;

    // Define the loss function
    CrossEntropyLoss criterion = CrossEntropyLoss();

    cout << "Finished loss initialization" << endl;

    // Define the optimizer
    Adam optimizer = Adam(model, LR);

    cout << "Finished optimizer initialization" << endl;

    float loss = 0.0;
    float acc = 0.0;
    vector<float> loss_list;
    vector<float> accuracy_list;
    
    // Create files to save metrics
    ofstream loss_file("training_loss.txt");
    ofstream acc_file("training_accuracy.txt");
    
    if (!loss_file.is_open() || !acc_file.is_open()) {
        cerr << "Failed to open files for saving metrics" << endl;
        return 1;
    }
    
    // Write headers to files
    loss_file << "Epoch,Loss" << endl;
    acc_file << "Epoch,Accuracy" << endl;

    cout << "Training started..." << endl;

    // ============================ Training ====================================

    // Example of iterating through all batches
    for (size_t e = 0; e < EPOCH; e++)
    {
        cout << "\nEpoch " << e + 1 << ":\n";
        dataset.reset(); // Reset batch counter at the start of each epoch
        loss_list.clear();
        accuracy_list.clear();

        for (size_t i = 0; i < dataset.get_num_batches(); i++)
        {
            auto batch = dataset.get_next_batch();
            auto [data, labels] = batch.to_tensor();

            optimizer.zero_grad();

            // forward propagation
            Tensor<> output = model(data);

            loss = criterion(output, labels);
            acc = metrics::accuracy(output, labels);

            accuracy_list.push_back(acc);
            loss_list.push_back(loss);

            // backward propagation
            Tensor<> grad = criterion.backward();
            model.backward(grad);

            optimizer.step();

            // print the training stats
            print_stats_line(i, loss, acc);
        }

        float total_loss = accumulate(loss_list.begin(), loss_list.end(), 0.0) / loss_list.size();
        float total_acc = accumulate(accuracy_list.begin(), accuracy_list.end(), 0.0) / accuracy_list.size() * 100;

        // Save metrics to files
        loss_file << e + 1 << "," << total_loss << endl;
        acc_file << e + 1 << "," << total_acc << endl;

        cout << "------------------------------------" << endl;
        cout << "Total Loss in Epoch " << e + 1 << " = " << total_loss << "" << endl;
        cout << "Total Accuracy in Epoch " << e + 1 << " = " << total_acc << "%" << endl;
        cout << "------------------------------------" << endl;
    }

    // Close metrics files
    loss_file.close();
    acc_file.close();
    cout << "Training metrics saved to training_loss.txt and training_accuracy.txt" << endl;

    // ============================ Inference ====================================

    model.eval();

    const string mnist_image_file_test = "../data/mnist/t10k-images.idx3-ubyte";
    const string mnist_label_file_test = "../data/mnist/t10k-labels.idx1-ubyte";

    MNIST test_dataset(BATCH_SIZE);

    if (!test_dataset.load_data(mnist_image_file_test, mnist_label_file_test))
    {
        cerr << "Failed to load test dataset" << endl;
        return 1;
    }

    cout << "\n------------------------------------" << endl;
    cout << "Testing started..." << endl;

    loss = 0.0;
    acc = 0.0;
    loss_list.clear();
    accuracy_list.clear();

    for (size_t i = 0; i < test_dataset.get_num_batches(); i++)
    {
        auto batch = test_dataset.get_next_batch();
        auto [data, labels] = batch.to_tensor();

        // forward propagation
        Tensor<> output = model(data);

        loss = criterion(output, labels);
        acc = metrics::accuracy(output, labels);

        accuracy_list.push_back(acc);
        loss_list.push_back(loss);

        // print the testing stats
        print_stats_line(i, loss, acc);
    }

    float total_loss = accumulate(loss_list.begin(), loss_list.end(), 0.0) / loss_list.size();
    float total_acc = accumulate(accuracy_list.begin(), accuracy_list.end(), 0.0) / accuracy_list.size() * 100;

    // Save test metrics to files
    ofstream test_loss_file("test_loss.txt");
    ofstream test_acc_file("test_accuracy.txt");
    
    if (test_loss_file.is_open() && test_acc_file.is_open()) {
        test_loss_file << "Loss," << total_loss << endl;
        test_acc_file << "Accuracy," << total_acc << endl;
        test_loss_file.close();
        test_acc_file.close();
        cout << "Test metrics saved to test_loss.txt and test_accuracy.txt" << endl;
    }

    cout << "Average Loss on Test Data = " << total_loss << "" << endl;
    cout << "Average Accuracy on Test Data = " << total_acc << "%" << endl;

    cout << "------------------------------------" << endl;

    return 0;
}
