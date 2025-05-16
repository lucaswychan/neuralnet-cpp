#include <numeric>
#include <fstream>
#include <chrono>
#include "mlp.hpp"
#include "mnist.hpp"
#include "cross_entropy.hpp"
#include "accuracy.hpp"
#include "utils.hpp"
#include "sgd.hpp"
#include "adam.hpp"
#include "model_utils.hpp"
using namespace nn;

// Training function to avoid code duplication
void train_model(MLP& model, MNIST& dataset, const string& optimizer_name, Optimizer& optimizer, 
                const size_t epochs, const float dropout_p) {
    // Define the loss function
    CrossEntropyLoss criterion = CrossEntropyLoss();
    
    cout << "\nTraining with " << optimizer_name << " optimizer..." << endl;

    float loss = 0.0;
    float acc = 0.0;
    vector<float> loss_list;
    vector<float> accuracy_list;
    
    // Create files to save metrics
    ofstream loss_file(optimizer_name + "_loss.txt");
    ofstream acc_file(optimizer_name + "_accuracy.txt");
    
    if (!loss_file.is_open() || !acc_file.is_open()) {
        cerr << "Failed to open files for saving metrics" << endl;
        return;
    }
    
    // Write headers to files
    loss_file << "Epoch,Loss" << endl;
    acc_file << "Epoch,Accuracy" << endl;

    // Record training start time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Training loop
    for (size_t e = 0; e < epochs; e++) {
        cout << "\nEpoch " << e + 1 << ":\n";
        dataset.reset(); // Reset batch counter at the start of each epoch
        loss_list.clear();
        accuracy_list.clear();

        for (size_t i = 0; i < dataset.get_num_batches(); i++) {
            auto batch = dataset.get_next_batch();
            auto [data, labels] = batch.to_tensor();

            // Zero gradients before forward pass
            optimizer.zero_grad();

            // Forward propagation
            Tensor<> output = model(data);

            // Compute loss and accuracy
            loss = criterion(output, labels);
            acc = metrics::accuracy(output, labels);

            accuracy_list.push_back(acc);
            loss_list.push_back(loss);

            // Backward propagation
            Tensor<> grad = criterion.backward();
            model.backward(grad);
            
            // Update parameters using the optimizer
            optimizer.step();

            // Print the training stats
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

    // Record training end time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

    // Close metrics files
    loss_file.close();
    acc_file.close();
    cout << "Training metrics saved to " << optimizer_name << "_loss.txt and " 
         << optimizer_name << "_accuracy.txt" << endl;
    cout << "Training completed in " << duration << " seconds." << endl;
    
    // Test the model
    const string mnist_image_file_test = "../data/mnist/t10k-images.idx3-ubyte";
    const string mnist_label_file_test = "../data/mnist/t10k-labels.idx1-ubyte";

    MNIST test_dataset(dataset.get_batch_size());
    if (!test_dataset.load_data(mnist_image_file_test, mnist_label_file_test)) {
        cerr << "Failed to load test dataset" << endl;
        return;
    }

    cout << "\n------------------------------------" << endl;
    cout << "Testing started..." << endl;

    // Set model to evaluation mode
    model.eval();
    
    loss = 0.0;
    acc = 0.0;
    loss_list.clear();
    accuracy_list.clear();

    for (size_t i = 0; i < test_dataset.get_num_batches(); i++) {
        auto batch = test_dataset.get_next_batch();
        auto [data, labels] = batch.to_tensor();

        // Forward propagation
        Tensor<> output = model(data);

        loss = criterion(output, labels);
        acc = metrics::accuracy(output, labels);

        accuracy_list.push_back(acc);
        loss_list.push_back(loss);

        // Print the testing stats
        print_stats_line(i, loss, acc);
    }

    float total_loss = accumulate(loss_list.begin(), loss_list.end(), 0.0) / loss_list.size();
    float total_acc = accumulate(accuracy_list.begin(), accuracy_list.end(), 0.0) / accuracy_list.size() * 100;

    // Save test metrics to files
    ofstream test_loss_file(optimizer_name + "_test_loss.txt");
    ofstream test_acc_file(optimizer_name + "_test_accuracy.txt");
    
    if (test_loss_file.is_open() && test_acc_file.is_open()) {
        test_loss_file << "Loss," << total_loss << endl;
        test_acc_file << "Accuracy," << total_acc << endl;
        test_loss_file.close();
        test_acc_file.close();
        cout << "Test metrics saved to " << optimizer_name << "_test_loss.txt and " 
             << optimizer_name << "_test_accuracy.txt" << endl;
    }

    cout << "Average Loss on Test Data = " << total_loss << "" << endl;
    cout << "Average Accuracy on Test Data = " << total_acc << "%" << endl;
    cout << "------------------------------------" << endl;
}

int main() {
    // Define the hyperparameters
    const float LR_SGD = 0.01f;
    const float LR_ADAM = 0.001f;
    const float MOMENTUM = 0.9f;
    const size_t EPOCH = 5;  // Reduced for example
    const size_t BATCH_SIZE = 64;
    const float DROPOUT_P = 0.2f;

    // Load the dataset
    MNIST dataset(BATCH_SIZE);
    const string mnist_image_file = "../data/mnist/train-images.idx3-ubyte";
    const string mnist_label_file = "../data/mnist/train-labels.idx1-ubyte";

    if (!dataset.load_data(mnist_image_file, mnist_label_file)) {
        cerr << "Failed to load dataset" << endl;
        return 1;
    }

    // --------------------- Train with SGD ---------------------
    {
        // Initialize the model
        bool bias = true;
        MLP model = MLP(784, {128, 64, 10}, bias, DROPOUT_P);
        cout << "Model initialized for SGD training" << endl;

        // Initialize SGD optimizer with momentum
        SGD optimizer(model, LR_SGD, MOMENTUM);
        
        // Train the model with SGD
        train_model(model, dataset, "sgd", optimizer, EPOCH, DROPOUT_P);
    }

    // --------------------- Train with Adam ---------------------
    {
        // Initialize a new model
        bool bias = true;
        MLP model = MLP(784, {128, 64, 10}, bias, DROPOUT_P);
        cout << "Model initialized for Adam training" << endl;

        // Initialize Adam optimizer
        Adam optimizer(model, LR_ADAM);
        
        // Train the model with Adam
        train_model(model, dataset, "adam", optimizer, EPOCH, DROPOUT_P);
    }

    cout << "\nOptimizer comparison complete. Check the output files for performance comparison." << endl;

    return 0;
} 