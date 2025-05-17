#pragma once
#include "module.hpp"
#include <initializer_list>
#include <vector>
#include <unordered_map>

namespace nn
{

    /**
     * A sequential container for modules.
     * Modules are added in the order they should be called during forward pass.
     * Propagates training state to all contained modules.
     */
    class Sequential : public Module
    {
    public:
        /**
         * Default constructor for an empty sequential container.
         */
        Sequential();

        /**
         * Constructor that takes a list of modules.
         *
         * @param modules A list of Module pointers to add to the container.
         */
        Sequential(initializer_list<Module *> modules);

        /**
         * Constructor that takes a vector of modules.
         *
         * @param modules A vector of Module pointers to add to the container.
         */
        Sequential(const vector<Module *> &modules);

        /**
         * Copy constructor is deleted to prevent unintended copying
         * of modules which would lead to memory issues with ownership.
         */
        Sequential(const Sequential &) = delete;

        /**
         * Move constructor that transfers ownership of modules from
         * one Sequential container to another.
         */
        Sequential(Sequential &&other) noexcept;

        /**
         * Destructor that deletes all owned modules.
         */
        virtual ~Sequential();

        /**
         * Passes the input tensor through all modules in sequence.
         *
         * @param input The input tensor to pass through the modules.
         * @return The output from the last module.
         */
        virtual Tensor<> forward(const Tensor<> &input) override;

        /**
         * Propagates the gradient backward through all modules in reverse order.
         *
         * @param grad_output The gradient of the loss with respect to the output.
         * @return The gradient with respect to the input.
         */
        virtual Tensor<> backward(const Tensor<> &grad_output) override;

        /**
         * Adds a module to the end of the sequence.
         *
         * @param module The module to add.
         * @return A reference to this Sequential container for chaining.
         */
        Sequential &add(Module *module);

        /**
         * Gets the number of modules in the sequence.
         *
         * @return The number of modules.
         */
        inline size_t size() const
        {
            return this->modules_.size();
        }

        /**
         * Gets a module at a specific index.
         *
         * @param index The index of the module to get.
         * @return The module at the specified index.
         */
        Module *get(size_t index) const;

        /**
         * Get all parameters of contained modules for optimization
         *
         * @param params Map to store parameters
         * @param grads Map to store gradients
         * @param prefix Prefix for parameter names
         */
        virtual void register_parameters(
            unordered_map<string, Tensor<> *> &params,
            unordered_map<string, Tensor<> *> &grads,
            const string &prefix = "") const override;

        /**
         * Copy assignment operator is deleted to prevent unintended copying
         * of modules which would lead to memory issues with ownership.
         *
         * Consider using move semantics or implementing a clone method if
         * copying is needed.
         */
        Sequential &operator=(const Sequential &) = delete;

        /**
         * Move assignment operator to transfer ownership of modules from one
         * Sequential container to another.
         */
        Sequential &operator=(Sequential &&other) noexcept;

    protected:
        /**
         * Applies a function to all child modules.
         *
         * @param fn A function that takes a Module reference and returns void.
         */
        virtual void apply_to_children(const function<void(Module &)> &fn) override;

    private:
        vector<Module *> modules_;
    };

} // namespace nn