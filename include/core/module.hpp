using namespace std;

class Module {
    public:
        virtual ~Module() = default;
        virtual float** forward(const float** input) = 0;
        virtual float** backward(const float** grad_output) = 0;
        virtual void update_params(const float lr) = 0;

        float** operator()(const float** input) {
            return this->forward(input);
        }
    
    protected:
        float** input_cache_;
};