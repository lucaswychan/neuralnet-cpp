#include "sequential.hpp"
#include <stdexcept>

namespace nn {

// Default constructor
Sequential::Sequential() = default;

// Constructor with initializer_list
Sequential::Sequential(std::initializer_list<Module*> modules) {
    for (Module* module : modules) {
        this->modules_.push_back(module);
    }
}

// Constructor with vector
Sequential::Sequential(const std::vector<Module*>& modules) {
    for (Module* module : modules) {
        this->modules_.push_back(module);
    }
}

// Move constructor
Sequential::Sequential(Sequential&& other) noexcept : modules_(std::move(other.modules_)) {
    // Clear the other container's vector after moving
    other.modules_.clear();
}

// Destructor
Sequential::~Sequential() {
    for (Module* module : this->modules_) {
        delete module;
    }
}

// Forward pass
Tensor<> Sequential::forward(const Tensor<>& input) {
    Tensor<> x = input;
    
    for (Module* module : this->modules_) {
        x = module->forward(x);
    }
    
    return x;
}

// Backward pass
Tensor<> Sequential::backward(const Tensor<>& grad_output) {
    Tensor<> grad = grad_output;
    
    for (int i = this->modules_.size() - 1; i >= 0; i--) {
        grad = this->modules_[i]->backward(grad);
    }
    
    return grad;
}

// Add a module
Sequential& Sequential::add(Module* module) {
    this->modules_.push_back(module);
    return *this;
}

// Get module at index
Module* Sequential::get(size_t index) const {
    if (index >= this->modules_.size()) {
        throw std::out_of_range("Index out of range");
    }
    return this->modules_[index];
}

// Move assignment operator
Sequential& Sequential::operator=(Sequential&& other) noexcept {
    if (this != &other) {
        // First clean up existing modules
        for (Module* module : this->modules_) {
            delete module;
        }
        
        // Transfer ownership
        this->modules_ = std::move(other.modules_);
        
        // Clear the other container's vector
        other.modules_.clear();
    }
    return *this;
}

// Apply function to children
void Sequential::apply_to_children(const function<void(Module&)>& fn) {
    for (Module* module : this->modules_) {
        fn(*module);
    }
}

void Sequential::register_parameters(
    unordered_map<string, Tensor<> *> &params,
    unordered_map<string, Tensor<> *> &grads,
    const string &prefix) const
{

    for (size_t i = 0; i < this->modules_.size(); ++i)
    {
        string module_prefix = prefix.empty() ? "layer" + to_string(i) : prefix + ".layer" + to_string(i);
        cout << "Getting parameters for " << module_prefix << endl;

        this->modules_[i]->register_parameters(params, grads, module_prefix);
    }
}

} // namespace nn 