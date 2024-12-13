#include "utils.hpp"

void print_training_stats(int batch, float loss, float accuracy) {
    cout << "\rBatch " << setw(4) << batch << " "
                << "Loss: " << fixed << setprecision(5) << setw(8) << loss << " "
                << "Accuracy: " << fixed << setprecision(2) << setw(6) << accuracy << "%"
                << flush;
}

void print_training_stats_line(int batch, float loss, float accuracy) {
    cout << "Batch " << setw(4) << batch << " "
                << "Loss: " << fixed << setprecision(5) << setw(8) << loss << " "
                << "Accuracy: " << fixed << setprecision(2) << setw(6) << accuracy << "%"
                << endl;
}