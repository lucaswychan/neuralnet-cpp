#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <initializer_list>
using namespace std;


template <typename T>
class Tensor {
    public:
        Tensor() = default;

        Tensor(vector<int> shape, T value);

        Tensor operator

    
    private:
        vector<int> shape_;

        // data is stored as a 1D vector
        vector<T> data_;
};
