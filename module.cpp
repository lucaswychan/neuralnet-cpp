using namespace std;

class Module {
    public:
        virtual float** forward(float** input) = 0;
        virtual float** backward(float** grad_output) = 0;
};