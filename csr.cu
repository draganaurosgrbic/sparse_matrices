#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <iomanip>
#include <random>
#include <chrono>
#include <algorithm>

__global__
void csr_spmv(long num_rows, long num_cols,
             int *rowptr, int *colind,
             double *values, double *x,
             double *y) {
  // Get the row index for this thread.
  long row = blockIdx.x * blockDim.x + threadIdx.x;

  // Make sure we don't go out of bounds.
  if (row < num_rows) {
    double result = 0.0;
    // Iterate over the non-zero elements in this row.
    for (int i = rowptr[row]; i < rowptr[row + 1]; ++i) {
      long col = colind[i];
      result += values[i] * x[col];
    }
    y[row] = result;
  }
}

// Function to generate a random dense vector
std::vector<double> generateRandomVector(long size) {
  std::vector<double> vec(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (long i = 0; i < size; ++i) {
    vec[i] = dist(gen);
  }
  return vec;
}

// Function to generate a random sparse matrix in CSR format (Optimized)
void generateRandomCSRMatrix(long num_rows, long num_cols, long num_nonzeros,
                             std::vector<int> &row_ptrs,
                             std::vector<int> &col_indices,
                             std::vector<double> &values) {
    row_ptrs.clear();
    col_indices.clear();
    values.clear();

    // Ensure num_nonzeros is within reasonable bounds
    num_nonzeros = std::min(num_nonzeros, num_rows * num_cols);

    // Use a Mersenne Twister engine
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<long> col_dist(0, num_cols - 1);
    std::uniform_real_distribution<double> val_dist(-1.0, 1.0);

    // Generate row pointers.  The last element is the number of nonzeros.
    row_ptrs.resize(num_rows + 1);
    long avg_nonzeros_per_row = num_nonzeros / num_rows;
    long remaining_nonzeros = num_nonzeros;
    for (long i = 0; i < num_rows; ++i) {
        // Distribute nonzeros, ensuring at least 0 and not more than remaining.
        long nonzeros_in_row = (i < num_rows - 1) ? std::min(col_dist(gen), remaining_nonzeros) : remaining_nonzeros;
        row_ptrs[i] = values.size(); // Current offset into values array
        remaining_nonzeros -= nonzeros_in_row;
        for (long j = 0; j < nonzeros_in_row; ++j) {
            long col = col_dist(gen);
            double val = val_dist(gen);
            col_indices.push_back(col);
            values.push_back(val);
        }
    }
    row_ptrs[num_rows] = values.size();
}

int main() {
  // Define the matrix dimensions and number of non-zero elements
  long num_rows = 100;
  long num_cols = 100;
  long num_nonzeros = 100;

  // CSR data structures
  std::vector<int> row_ptrs_csr, col_indices_csr;
  std::vector<double> values_csr;

  // Generate the random CSR matrix
  auto start_gen = std::chrono::high_resolution_clock::now();
  generateRandomCSRMatrix(num_rows, num_cols, num_nonzeros, row_ptrs_csr,
                           col_indices_csr, values_csr);
  auto end_gen = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration_gen = end_gen - start_gen;
  std::cout << "Matrix generation time: " << duration_gen.count() << " seconds"
            << std::endl;

  // Generate a random vector
  std::vector<double> x = generateRandomVector(num_cols);
  std::vector<double> y(num_rows, 0.0); // Output vector

  // Print the first few non-zero elements
  std::cout << "\nFirst 10 Non-Zero Elements:\n";
  for (int i = 0; i < std::min(10, (int)col_indices_csr.size()); ++i) {
    int row_index = -1;
        for (long j = 0; j < num_rows; ++j) {
            if (row_ptrs_csr[j] <= i && i < row_ptrs_csr[j + 1]) {
                row_index = j;
                break;
            }
        }
    std::cout << "Row: " << row_index << ", Col: " << col_indices_csr[i]
              << ", Value: " << std::fixed << std::setprecision(6)
              << values_csr[i] << std::endl;
  }

  // Print the first 10 row pointers
    std::cout << "\nRow Pointers:\n";
    for (int i = 0; i < std::min(11, (int)row_ptrs_csr.size()); ++i) {
        std::cout << row_ptrs_csr[i] << " ";
    }
    std::cout << std::endl;

  // Print the input vector
  std::cout << "\nInput Vector (x):\n";
  for (long i = 0; i < std::min(10L, (long)x.size()); ++i) {
    std::cout << std::fixed << std::setprecision(6) << x[i] << " ";
  }
  std::cout << std::endl;

  // 1. Copy data to the GPU
  int *d_row_ptrs, *d_col_indices;
  double *d_values, *d_x, *d_y;

  // Allocate memory on the GPU
  cudaMalloc((void **)&d_row_ptrs,
           (row_ptrs_csr.size()) * sizeof(int));
  cudaMalloc((void **)&d_col_indices, col_indices_csr.size() * sizeof(int));
  cudaMalloc((void **)&d_values, values_csr.size() * sizeof(double));
  cudaMalloc((void **)&d_x, x.size() * sizeof(double));
  cudaMalloc((void **)&d_y, y.size() * sizeof(double));

  // Copy data from host to device
  cudaMemcpy(d_row_ptrs, row_ptrs_csr.data(),
           row_ptrs_csr.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_indices, col_indices_csr.data(),
           col_indices_csr.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, values_csr.data(), values_csr.size() * sizeof(double),
           cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x.data(), x.size() * sizeof(double),
           cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y.data(), y.size() * sizeof(double),
           cudaMemcpyHostToDevice);

  // 2. Configure and launch the kernel
  int threads_per_block = 256;
  int num_blocks = (num_rows + threads_per_block - 1) / threads_per_block;

  // Record the start time
  auto start_kernel = std::chrono::high_resolution_clock::now();
  csr_spmv<<<num_blocks, threads_per_block>>>(
      num_rows, num_cols, d_row_ptrs, d_col_indices, d_values, d_x, d_y);
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
  cudaFree(d_row_ptrs);
  cudaFree(d_col_indices);
  cudaFree(d_values);
  cudaFree(d_x);
  cudaFree(d_y);

  return 0;
}
