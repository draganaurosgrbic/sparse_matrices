#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <fstream>
#include <string>
#include <algorithm>
#include <cmath>

// Function to generate a random sparse matrix in COO format
void generateRandomMatrixCOO(int num_rows, int num_cols, double sparsity,
                            std::vector<int> &row_indices,
                            std::vector<int> &col_indices,
                            std::vector<double> &values) {
    int num_nonzeros = static_cast<int>(num_rows * num_cols * (1 - sparsity));
    row_indices.clear();
    col_indices.clear();
    values.clear();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> row_dist(0, num_rows - 1);
    std::uniform_int_distribution<> col_dist(0, num_cols - 1);
    std::uniform_real_distribution<> val_dist(-1.0, 1.0);

    // Generate non-zero entries
    for (int i = 0; i < num_nonzeros; ++i) {
        int row = row_dist(gen);
        int col = col_dist(gen);
        double val = val_dist(gen);
        row_indices.push_back(row);
        col_indices.push_back(col);
        values.push_back(val);
    }
    // Remove duplicate entries
    std::vector<std::tuple<int, int, double>> entries;
    for (size_t i = 0; i < row_indices.size(); ++i) {
        entries.emplace_back(row_indices[i], col_indices[i], values[i]);
    }
    std::sort(entries.begin(), entries.end());
    entries.erase(std::unique(entries.begin(), entries.end()), entries.end());

    row_indices.clear();
    col_indices.clear();
    values.clear();
    for (const auto &entry : entries) {
        row_indices.push_back(std::get<0>(entry));
        col_indices.push_back(std::get<1>(entry));
        values.push_back(std::get<2>(entry));
    }
}

// Function to generate a banded sparse matrix in COO format
void generateBandedMatrixCOO(int num_rows, int num_cols, int bandwidth,
                            std::vector<int> &row_indices,
                            std::vector<int> &col_indices,
                            std::vector<double> &values) {
    row_indices.clear();
    col_indices.clear();
    values.clear();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> val_dist(-1.0, 1.0);

    for (int row = 0; row < num_rows; ++row) {
        int start_col = std::max(0, row - bandwidth);
        int end_col = std::min(num_cols - 1, row + bandwidth);
        for (int col = start_col; col <= end_col; ++col) {
            double val = val_dist(gen);
            row_indices.push_back(row);
            col_indices.push_back(col);
            values.push_back(val);
        }
    }
}

// Function to generate a block diagonal sparse matrix in COO format
void generateBlockDiagonalMatrixCOO(int num_blocks, int block_size,
                                    int num_rows, int num_cols,
                                    std::vector<int> &row_indices,
                                    std::vector<int> &col_indices,
                                    std::vector<double> &values) {
    row_indices.clear();
    col_indices.clear();
    values.clear();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> val_dist(-1.0, 1.0);

    for (int block = 0; block < num_blocks; ++block) {
        int row_offset = block * block_size;
        int col_offset = block * block_size;
        for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j < block_size; ++j) {
                int row = row_offset + i;
                int col = col_offset + j;
                if (row < num_rows && col < num_cols) { // Check边界
                    double val = val_dist(gen);
                    row_indices.push_back(row);
                    col_indices.push_back(col);
                    values.push_back(val);
                }
            }
        }
    }
}

// Function to convert COO to CSR format
void cooToCSR(const std::vector<int> &row_indices,
             const std::vector<int> &col_indices,
             const std::vector<double> &values, int num_rows,
             std::vector<int> &row_ptrs, std::vector<int> &csr_col_indices,
             std::vector<double> &csr_values) {
    row_ptrs.clear();
    csr_col_indices.clear();
    csr_values.clear();

    row_ptrs.resize(num_rows + 1, 0);
    // Count non-zeros per row
    for (int row : row_indices) {
        row_ptrs[row + 1]++;
    }
    // Compute prefix sum of row pointers
    for (int i = 0; i < num_rows; ++i) {
        row_ptrs[i + 1] += row_ptrs[i];
    }
    // Fill in column indices and values
    csr_col_indices.resize(values.size());
    csr_values.resize(values.size());
    std::vector<int> row_pos(num_rows, 0);
    for (size_t i = 0; i < row_indices.size(); ++i) {
        int row = row_indices[i];
        int dest_pos = row_ptrs[row] + row_pos[row];
        csr_col_indices[dest_pos] = col_indices[i];
        csr_values[dest_pos] = values[i];
        row_pos[row]++;
    }
}

// Function to convert COO to ELLPACK format
void cooToELLPACK(const std::vector<int> &row_indices,
                  const std::vector<int> &col_indices,
                  const std::vector<double> &values, int num_rows,
                  int num_cols, std::vector<int> &ell_col_indices,
                  std::vector<double> &ell_values, int &max_row_length) {
    ell_col_indices.clear();
    ell_values.clear();
    max_row_length = 0;

    std::vector<int> row_lengths(num_rows, 0);
    // Calculate row lengths
    for (int row : row_indices) {
        row_lengths[row]++;
    }
    // Find maximum row length
    for (int length : row_lengths) {
        max_row_length = std::max(max_row_length, length);
    }
    ell_col_indices.resize(num_rows * max_row_length, 0);
    ell_values.resize(num_rows * max_row_length, 0.0);

    std::vector<int> row_pos(num_rows, 0);
    for (size_t i = 0; i < row_indices.size(); ++i) {
        int row = row_indices[i];
        int pos = row * max_row_length + row_pos[row];
        ell_col_indices[pos] = col_indices[i];
        ell_values[pos] = values[i];
        row_pos[row]++;
    }
}

// Function to convert COO to SELL format
void cooToSELL(const std::vector<int> &row_indices,
              const std::vector<int> &col_indices,
              const std::vector<double> &values, int num_rows, int num_cols,
              std::vector<int> &slice_boundaries,
              std::vector<int> &slice_lengths,
              std::vector<int> &sell_col_indices,
              std::vector<double> &sell_values, int slice_size) {
    slice_boundaries.clear();
    slice_lengths.clear();
    sell_col_indices.clear();
    sell_values.clear();

    // Calculate the number of slices
    int num_slices = (num_rows + slice_size - 1) / slice_size;
    slice_boundaries.push_back(0); // First slice starts at row 0

    // Group rows into slices
    std::vector<std::vector<int>> slices_rows(num_slices);
    for (int i = 0; i < num_rows; ++i) {
        int slice_id = i / slice_size;
        slices_rows[slice_id].push_back(i);
    }

    // Process each slice
    for (int slice_id = 0; slice_id < num_slices; ++slice_id) {
        int slice_start = slice_boundaries[slice_id];
        int slice_end =
            (slice_id == num_slices - 1) ? num_rows
                                        : slice_start + slice_size; // exclusive
        slice_boundaries.push_back(
            slice_end); // Store the end row of the slice as the start of the next

        // Find the maximum row length for the current slice
        int max_row_length = 0;
        for (int row_in_slice : slices_rows[slice_id]) {
            int row_length = 0;
            for (size_t i = 0; i < row_indices.size(); ++i) {
                if (row_indices[i] == row_in_slice) {
                    row_length++;
                }
            }
            max_row_length = std::max(max_row_length, row_length);
        }
        slice_lengths.push_back(max_row_length);

        // Store the values and column indices for the slice
        for (int row_in_slice : slices_rows[slice_id]) {
            for (size_t i = 0; i < row_indices.size(); ++i) {
                if (row_indices[i] == row_in_slice) {
                    sell_values.push_back(values[i]);
                    sell_col_indices.push_back(col_indices[i]);
                }
            }
        }
    }
}

// Function to convert COO to Padded SELL format
void cooToPaddedSELL(const std::vector<int> &row_indices,
                    const std::vector<int> &col_indices,
                    const std::vector<double> &values, int num_rows,
                    int num_cols, std::vector<int> &slice_boundaries,
                    std::vector<int> &slice_lengths,
                    std::vector<int> &sellp_col_indices,
                    std::vector<double> &sellp_values, int slice_size,
                    int padding_factor) {
    slice_boundaries.clear();
    slice_lengths.clear();
    sellp_col_indices.clear();
    sellp_values.clear();

    // Calculate the number of slices
    int num_slices = (num_rows + slice_size - 1) / slice_size;
    slice_boundaries.push_back(0); // First slice starts at row 0

    // Group rows into slices
    std::vector<std::vector<int>> slices_rows(num_slices);
    for (int i = 0; i < num_rows; ++i) {
        int slice_id = i / slice_size;
        slices_rows[slice_id].push_back(i);
    }

    // Process each slice
    for (int slice_id = 0; slice_id < num_slices; ++slice_id) {
        int slice_start = slice_boundaries[slice_id];
        int slice_end =
            (slice_id == num_slices - 1) ? num_rows
                                        : slice_start + slice_size; // exclusive
        slice_boundaries.push_back(
            slice_end); // Store the end row of the slice as the start of the next

        // Find the maximum row length for the current slice
        int max_row_length = 0;
        for (int row_in_slice : slices_rows[slice_id]) {
            int row_length = 0;
            for (size_t i = 0; i < row_indices.size(); ++i) {
                if (row_indices[i] == row_in_slice) {
                    row_length++;
                }
            }
            max_row_length = std::max(max_row_length, row_length);
        }

        // Calculate padded length
        int padded_length =
            (int)ceil((double)max_row_length / padding_factor) * padding_factor;
        slice_lengths.push_back(padded_length);

        // Store the values and column indices for the slice, with padding
        for (int row_in_slice : slices_rows[slice_id]) {
            int row_count = 0;
            for (size_t i = 0; i < row_indices.size(); ++i) {
                if (row_indices[i] == row_in_slice) {
                    sellp_values.push_back(values[i]);
                    sellp_col_indices.push_back(col_indices[i]);
                    row_count++;
                }
            }
            // Add padding
            for (int j = row_count; j < padded_length; ++j) {
                sellp_values.push_back(0.0);
                sellp_col_indices.push_back(0);
            }
        }
    }
}

void writeMatrixToFile(const std::vector<int> &row_indices,
                      const std::vector<int> &col_indices,
                      const std::vector<double> &values,
                      const std::string &filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "COO Format\n";
        file << "num_rows, num_cols, num_nonzeros\n";
        int num_rows = 0;
        int num_cols = 0;
        for (int row : row_indices) {
            num_rows = std::max(num_rows, row + 1);
        }
        for (int col : col_indices) {
            num_cols = std::max(num_cols, col + 1);
        }

        file << num_rows << ", " << num_cols << ", " << row_indices.size()
             << "\n";
        file << "row_index, col_index, value\n";
        for (size_t i = 0; i < row_indices.size(); ++i) {
            file << row_indices[i] << ", " << col_indices[i] << ", "
                 << std::fixed << std::setprecision(6) << values[i] << "\n";
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

int main() {
    // Experiment parameters
    std::vector<std::tuple<int, int, double, std::string>>
        experiment_parameters; // num_rows, num_cols, sparsity, matrix_type

    // 1. Varying Matrix Size, Constant Sparsity, Random Distribution
    for (int size = 1000; size <= 10000; size += 3000) {
        experiment_parameters.emplace_back(size, size, 0.9,
                                           "random_" + std::to_string(size));
    }

    // 2. Varying Sparsity, Constant Size, Random Distribution
    for (double sparsity = 0.9; sparsity >= 0.1; sparsity -= 0.2) {
        int size = 5000;
        experiment_parameters.emplace_back(size, size, sparsity,
                                           "random_" + std::to_string((int)(sparsity * 100)));
    }

    // 3. Banded Matrix with Varying Bandwidth
    for (int bandwidth = 10; bandwidth <= 100; bandwidth += 30) {
        int size = 5000;
        experiment_parameters.emplace_back(size, size, 0.9,
                                           "banded_" + std::to_string(bandwidth));
    }

    // 4. Block Diagonal Matrix with Varying Block Size
    for (int block_size = 10; block_size <= 100; block_size += 30) {
        int size = 5000;
        int num_blocks = size / block_size;
        experiment_parameters.emplace_back(size, size, 0.9,
                                           "block_" + std::to_string(block_size));
    }

    // Generate matrices and write to files
    for (const auto &params : experiment_parameters) {
        int num_rows = std::get<0>(params);
        int num_cols = std::get<1>(params);
        double sparsity = std::get<2>(params);
        std::string matrix_type = std::get<3>(params);

        std::vector<int> row_indices, col_indices;
        std::vector<double> values;

        if (matrix_type.find("random") != std::string::npos) {
            generateRandomMatrixCOO(num_rows, num_cols, sparsity, row_indices,
                                    col_indices, values);
        } else if (matrix_type.find("banded") != std::string::npos) {
            int bandwidth = std::stoi(matrix_type.substr(matrix_type.find("_") + 1));
            generateBandedMatrixCOO(num_rows, num_cols, bandwidth, row_indices,
                                    col_indices, values);
        } else if (matrix_type.find("block") != std::string::npos) {
            int block_size = std::stoi(matrix_type.substr(matrix_type.find("_") + 1));
            int num_blocks = num_rows / block_size;
            generateBlockDiagonalMatrixCOO(num_blocks, block_size, num_rows, num_cols,
                                            row_indices, col_indices, values);
        }

        std::string filename = "matrix_" + matrix_type + ".txt";
        writeMatrixToFile(row_indices, col_indices, values, filename);

        std::cout << "Generated matrix: " << filename << std::endl;
    }

    std::cout << "Matrix generation complete.  You can now use these files to run your SpMV experiments." << std::endl;

    return 0;
}

