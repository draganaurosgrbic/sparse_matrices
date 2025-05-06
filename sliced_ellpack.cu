#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm> // for std::max and std::min

// CUDA kernel for Sliced ELLPACK SpMV
__global__
void sell_spmv(long num_slices, int slice_size, int *slice_boundaries,
               int *slice_lengths, int *col_indices, double *values, double *x,
               double *y) {
  int slice_id = blockIdx.x;
  int row_offset_in_slice = threadIdx.x;

  if (slice_id < num_slices) {
    int slice_start = slice_boundaries[slice_id];
    int slice_end = slice_boundaries[slice_id + 1];
    int slice_length = slice_lengths[slice_id];

    for (int row_in_slice = slice_start + row_offset_in_slice;
         row_in_slice < slice_end; row_in_slice += blockDim.x) {
      double result = 0.0;
      int row_index_in_slice = row_in_slice - slice_start;
      if (row_index_in_slice <
          (slice_end -
           slice_start)) { // make sure the row_index_in_slice is valid
        for (int j = 0; j < slice_length; ++j) {
          int global_col =
              col_indices[slice_id * slice_length * slice_size +
                          row_index_in_slice * slice_length + j];
          double val =
              values[slice_id * slice_length * slice_size +
                     row_index_in_slice * slice_length + j];
          result += val * x[global_col];
        }
        y[row_in_slice] = result;
      }
    }
  }
}

// Function to generate a random dense vector
std::vector<double> generateRandomVector(long size) {
  std::vector<double> vec(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  for (long i = 0; i < size; ++i) {
    vec[i] = dist(gen);
  }
  return vec;
}

// Function to generate a random sparse matrix in Sliced ELLPACK format
void generateRandomSELLMatrix(long num_rows, long num_cols, long num_nonzeros, int slice_size,
                               std::vector<int>& slice_boundaries, std::vector<int>& slice_lengths,
                               std::vector<int>& col_indices, std::vector<double>& values) {
    // Use Mersenne Twister
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<long> col_dist(0, num_cols - 1);
    std::uniform_real_distribution<> val_dist(-1.0, 1.0);

    // Calculate the number of slices
    long num_slices = (num_rows + slice_size - 1) / slice_size;

    // Initialize slice_boundaries array
    slice_boundaries.resize(num_slices + 1);
    for (long i = 0; i < num_slices; ++i) {
        slice_boundaries[i] = i * slice_size;
    }
    slice_boundaries[num_slices] = num_rows;

    // Determine maximum row length for each slice
    slice_lengths.resize(num_slices);
    for (long i = 0; i < num_slices; ++i) {
        long slice_start = slice_boundaries[i];
        long slice_end = slice_boundaries[i + 1];
        int max_len = 0;
        for (long j = slice_start; j < slice_end; ++j) {
             long nonzeros_in_row = num_nonzeros / num_rows;
            if (j < num_nonzeros % num_rows)
                nonzeros_in_row++;
            max_len = std::max(max_len, (int)nonzeros_in_row);
        }
        slice_lengths[i] = max_len;
    }

    // Calculate the total number of elements needed for col_indices and values
    long total_elements = 0;
    for (long i = 0; i < num_slices; ++i) {
        total_elements += slice_lengths[i] * slice_size;
    }
    col_indices.resize(total_elements);
    values.resize(total_elements);

    // Generate random data for each slice
    long current_index = 0;
     for (long slice_id = 0; slice_id < num_slices; ++slice_id) {
        long slice_start = slice_boundaries[slice_id];
        long slice_end = slice_boundaries[slice_id + 1];
        int slice_len = slice_lengths[slice_id];

        for (int row_in_slice = 0; row_in_slice < (slice_end - slice_start); ++row_in_slice) {
            long row_global = slice_start + row_in_slice;
             long nonzeros_in_row = num_nonzeros / num_rows;
            if (row_global < num_nonzeros % num_rows)
                nonzeros_in_row++;
            for (int j = 0; j < slice_len; ++j) {
                if (j < nonzeros_in_row)
                {
                    long col = col_dist(gen);
                    double val = val_dist(gen);
                    col_indices[current_index] = col;
                    values[current_index] = val;
                }
                else{
                    col_indices[current_index] = 0;
                    values[current_index] = 0.0;
                }
                current_index++;
            }
        }
    }
}

int main() {
    // Define the matrix dimensions, number of non-zero elements, and slice size
    long num_rows = 100;
    long num_cols = 100;
    long num_nonzeros = 100;
    int slice_size = 1;

    // SELL data structures
    std::vector<int> slice_boundaries;
    std::vector<int> slice_lengths;
    std::vector<int> col_indices_sell;
    std::vector<double> values_sell;

    // Generate the random SELL matrix
    auto start_gen = std::chrono::high_resolution_clock::now();
    generateRandomSELLMatrix(num_rows, num_cols, num_nonzeros, slice_size,
                             slice_boundaries, slice_lengths, col_indices_sell, values_sell);
    auto end_gen = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_gen = end_gen - start_gen;
    std::cout << "Matrix generation time: " << duration_gen.count() << " seconds" << std::endl;

    // Generate a random vector
    std::vector<double> x = generateRandomVector(num_cols);
    std::vector<double> y(num_rows, 0.0);

    // 1. Copy data to the GPU
    int* d_slice_boundaries;
    int* d_slice_lengths;
    int* d_col_indices_sell;
    double* d_values_sell;
    double* d_x;
    double* d_y;

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_slice_boundaries, slice_boundaries.size() * sizeof(int));
    cudaMalloc((void**)&d_slice_lengths, slice_lengths.size() * sizeof(int));
    cudaMalloc((void**)&d_col_indices_sell, col_indices_sell.size() * sizeof(int));
    cudaMalloc((void**)&d_values_sell, values_sell.size() * sizeof(double));
    cudaMalloc((void**)&d_x, x.size() * sizeof(double));
    cudaMalloc((void**)&d_y, y.size() * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(d_slice_boundaries, slice_boundaries.data(), slice_boundaries.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_slice_lengths, slice_lengths.data(), slice_lengths.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices_sell, col_indices_sell.data(), col_indices_sell.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values_sell, values_sell.data(), values_sell.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.data(), y.size() * sizeof(double), cudaMemcpyHostToDevice);

    // 2. Configure and launch the kernel
    int threads_per_block = 256;
    int num_blocks = slice_boundaries.size() - 1;

    auto start_kernel = std::chrono::high_resolution_clock::now();
    sell_spmv<<<num_blocks, threads_per_block>>>(
        slice_boundaries.size() - 1, slice_size, d_slice_boundaries,
        d_slice_lengths, d_col_indices_sell, d_values_sell, d_x, d_y);
    cudaDeviceSynchronize();
    auto end_kernel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_kernel = end_kernel - start_kernel;

    // 3. Copy the result back from the GPU
    cudaMemcpy(y.data(), d_y, y.size() * sizeof(double), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Output Vector (y):" << std::endl;
    //for (long i = 0; i < num_rows; ++i) {
        //std::cout << std::fixed << std::setprecision(1) << y[i] << " ";
    //}
    std::cout << std::endl;
    std::cout << "Kernel execution time: " << duration_kernel.count() << " seconds" << std::endl;

    // 4. Free GPU memory
    cudaFree(d_slice_boundaries);
    cudaFree(d_slice_lengths);
    cudaFree(d_col_indices_sell);
    cudaFree(d_values_sell);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
