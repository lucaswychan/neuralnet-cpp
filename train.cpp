#include <vector>
#include <iostream>
using namespace std;

int main() {

    vector<int> v = {1, 2, 3, 4, 5};
    for (auto it = v.begin(); it != v.end(); it++) {
        cout << *it << " ";
    }
    cout << endl;
    return 0;
}