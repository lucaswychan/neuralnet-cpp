#include <vector>
using namespace std;

vector<vector<float>> allocateMatrix(int rows, int cols, const float value = 0.0f);
vector<vector<float>> matrixAddition(const vector<vector<float>>& A, const vector<vector<float>>& B);
vector<vector<float>> matrixSubtraction(const vector<vector<float>>& A, const vector<vector<float>>& B);
vector<vector<float>> matrixMultiplication(const vector<vector<float>>& A, const vector<vector<float>>& B);
vector<vector<float>> matrixTranspose(const vector<vector<float>>& matrix);
void printMatrix(const vector<vector<float>>& matrix);
