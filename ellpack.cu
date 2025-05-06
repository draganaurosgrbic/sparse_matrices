#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm> // for std::max

// CUDA kernel for ELLPACK SpMV
__global__
void ellpack_spmv(long num_rows, long max_row_length, int block_size,
                  const int *col_indices, const double *values, const double *x,
                  double *y) {
  long row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < num_rows) {
    double result = 0.0;
    // Iterate through the row, processing 'block_size' elements at a time.
    for (long j = 0; j < max_row_length; j++) { // Changed j to long
        int col = col_indices[row * max_row_length + j];
        result += values[row * max_row_length + j] * x[col];
    }
    y[row] = result;
  }
}

// Function to generate a random dense vector
std::vector<double> generateRandomVector(long size) { // Changed size to long
  std::vector<double> vec(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  for (long i = 0; i < size; ++i) {
    vec[i] = dist(gen);
  }
  return vec;
}

// Function to generate a random sparse matrix in ELLPACK format
void generateRandomELLPACKMatrix(long num_rows, long num_cols, long num_nonzeros, // Changed to long
                                 int block_size,
                                 std::vector<std::vector<int>> &col_indices,
                                 std::vector<std::vector<double>> &values) {
    // Use Mersenne Twister for random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<long> col_dist(0, num_cols - 1); // Changed to long
    std::uniform_real_distribution<> val_dist(-1.0, 1.0);

    // Calculate the maximum row length.
    long max_row_length = (num_nonzeros + num_rows - 1) / num_rows; // Changed to long
    max_row_length = std::max(1L, max_row_length); // Ensure max_row_length is at least 1.  Changed 1 to 1L

    // Calculate the padded row length.
    long padded_row_length = (max_row_length + block_size - 1) / block_size * block_size; // Changed to long

    // Initialize ELLPACK arrays.  Important:  Use padded length.
    col_indices.assign(num_rows, std::vector<int>(padded_row_length, 0)); // Initialize with 0
    values.assign(num_rows, std::vector<double>(padded_row_length, 0.0));   // Initialize with 0.0

    long nonzeros_generated = 0; // Changed to long
    for (long i = 0; i < num_rows; ++i) {
        long nonzeros_in_row = 0; // Changed to long
        while (nonzeros_in_row < max_row_length && nonzeros_generated < num_nonzeros) {
             long col = col_dist(gen); // Changed to long
             double val = val_dist(gen);
            col_indices[i][nonzeros_in_row] = col;
            values[i][nonzeros_in_row] = val;
            nonzeros_in_row++;
            nonzeros_generated++;
        }
    }
}

int main() {
    // Define the matrix dimensions and number of non-zero elements
    long num_rows = 100; // Changed to long
    long num_cols = 100; // Changed to long
    long num_nonzeros = 100; // Changed to long
    int block_size = 32; // Choose a block size for ELLPACK

    // ELLPACK data structures
    std::vector<std::vector<int>> col_indices_ellpack;
    std::vector<std::vector<double>> values_ellpack;

    // Generate the random ELLPACK matrix
    auto start_gen = std::chrono::high_resolution_clock::now();
    generateRandomELLPACKMatrix(num_rows, num_cols, num_nonzeros, block_size,
                                 col_indices_ellpack, values_ellpack);
    auto end_gen = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_gen = end_gen - start_gen;
    std::cout << "Matrix generation time: " << duration_gen.count() << " seconds"
              << std::endl;

    // Generate a random vector
    std::vector<double> x = generateRandomVector(num_cols);
    std::vector<double> y(num_rows, 0.0); // Output vector

     // Determine the maximum row length.
    long max_row_length = 0; // Changed to long
    for (const auto& row : values_ellpack)
    {
        max_row_length = std::max(max_row_length, (long)row.size()); // Changed to long
    }

    // Print the first few non-zero elements from the first row
    std::cout << "\nFirst 10 Non-Zero Elements of First Row:\n";
    for (int i = 0; i < std::min(10, (int)values_ellpack[0].size()); ++i) {
        std::cout << "Col: " << col_indices_ellpack[0][i]
                  << ", Value: " << std::fixed << std::setprecision(6)
                  << values_ellpack[0][i] << std::endl;
    }

    // Print the first 10 values of the input vector
    std::cout << "\nInput Vector (x):\n";
    //for (long i = 0; i < std::min(10, (long)x.size()); ++i) {
        //std::cout << std::fixed << std::setprecision(6) << x[i] << " ";
    //}
    std::cout << std::endl;

    // 1. Copy data to the GPU
    int *d_col_indices;
    double *d_values, *d_x, *d_y;

    // Flatten the 2D ELLPACK arrays into 1D arrays for copying to the GPU
    std::vector<int> col_indices_flat;
    std::vector<double> values_flat;
    for (long i = 0; i < num_rows; ++i) { // Changed to long
        for (int j = 0; j < values_ellpack[i].size(); ++j) {
            col_indices_flat.push_back(col_indices_ellpack[i][j]);
            values_flat.push_back(values_ellpack[i][j]);
        }
    }
    size_t ellpack_array_size = num_rows * max_row_length; // Changed to long
    // Allocate memory on the GPU
    cudaMalloc((void **)&d_col_indices,
               ellpack_array_size * sizeof(int));
    cudaMalloc((void **)&d_values,
               ellpack_array_size * sizeof(double));
    cudaMalloc((void **)&d_x, x.size() * sizeof(double));
    cudaMalloc((void **)&d_y, y.size() * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(d_col_indices, col_indices_flat.data(),
               ellpack_array_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values_flat.data(),
               ellpack_array_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.data(), y.size() * sizeof(double),
               cudaMemcpyHostToDevice);

    // 2. Configure and launch the kernel
    int threads_per_block = 256;
    int num_blocks = (num_rows + threads_per_block - 1) / threads_per_block;

    // Record the start time
    auto start_kernel = std::chrono::high_resolution_clock::now();
    ellpack_spmv<<<num_blocks, threads_per_block>>>(
        num_rows, max_row_length, block_size, d_col_indices, d_values, d_x, d_y);
    cudaDeviceSynchronize(); // Ensure kernel finishes before timing
    // Record the end time
    auto end_kernel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_kernel = end_kernel - start_kernel;

    // 3. Copy the result back from the GPU
    cudaMemcpy(y.data(), d_y, y.size() * sizeof(double),
               cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Output Vector (y):" << std::endl;
    //for (long i = 0; i < num_rows; ++i) {
        //std::cout << std::fixed << std::setprecision(1) << y[i] << " ";
    //}
    std::cout << std::endl;

    // Print the execution time
    std::cout << "CUDA kernel execution time: " << duration_kernel.count()
              << " seconds" << std::endl;

    // 4. Free GPU memory
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
