#include "utils/matrix_utils.hpp"
#include <vector>
#include <iostream>
using namespace std;

// Function to allocate a matrix dynamically
vector<vector<float>> allocateMatrix(const int rows, const int cols) {
    vector<vector<float>> matrix(rows, vector<float>(cols, 0.0f));
    return matrix;
}

// Function for matrix addition
vector<vector<float>> matrixAddition(const vector<vector<float>>& A, const vector<vector<float>>& B) {
    // A R^n x m, B R^n x m
    const int n = A.size(), m = A[0].size();
    if (n != B.size() || m != B[0].size()) {
        cout << "Error: Matrix dimensions are not compatible for subtraction." << endl;
        return vector<vector<float>>();
    }

    vector<vector<float>> result(n, vector<float>(m, 0.0f));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }
    return result;
}

// Function for matrix subtraction
vector<vector<float>> matrixSubtraction(const vector<vector<float>>& A, const vector<vector<float>>& B) {
    // A R^n x m, B R^n x m
    const int n = A.size(), m = m;
    if (n != B.size() || m != B[0].size()) {
        cout << "Error: Matrix dimensions are not compatible for subtraction." << endl;
        return vector<vector<float>>();
    }

    vector<vector<float>> result(n, vector<float>(m, 0.0f));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            result[i][j] = A[i][j] - B[i][j];
        }
    }
    return result;
}

// Function for matrix multiplication
vector<vector<float>> matrixMultiplication(const vector<vector<float>>& A, const vector<vector<float>>& B) {
    // A R^n x m, B R^m x k

    const int n = A.size(), m = A[0].size(), l = B[0].size();
    if (m != B.size()) {
        cout << "Error: Matrix dimensions are not compatible for multiplication." << endl;
        return vector<vector<float>>();
    }

    vector<vector<float>> result(n, vector<float>(l, 0.0f));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < l; j++) {
            for (int k = 0; k < m; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

// Function to transpose a matrix
vector<vector<float>> matrixTranspose(const vector<vector<float>>& A) {
    // A R^n x m
    const int n = A.size(), m = A[0].size();

    vector<vector<float>> result(m, vector<float>(n, 0.0f));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            result[j][i] = A[i][j];
        }
    }
    return result;
}

// Function to print a matrix
void printMatrix(const vector<vector<float>>& matrix) {
    const int n = matrix.size(), m = matrix[0].size();
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }
}
