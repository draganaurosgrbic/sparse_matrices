#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <iomanip>
#include <random>
#include <algorithm> // for std::max
#include <chrono>

// CUDA kernel for SELL-P SpMV
__global__
void sellp_spmv(long num_rows,
               int slice_size,
               int *slice_boundaries,
               int *slice_lengths,
               int *colind,
               double *values,
               double *x,
               double *y,
               int padding_factor) {
  // Get the row index for this thread.
  long row = blockIdx.x * blockDim.x + threadIdx.x;

  // Check if the row is within the valid range.
  if (row < num_rows) {
    double result = 0.0;
    // Find the slice this row belongs to.
    int slice_id;
    for (slice_id = 0; slice_id < (num_rows + slice_size - 1) / slice_size; ++slice_id) { // Corrected slice_id calculation
      if (row >= slice_boundaries[slice_id] &&
          row < slice_boundaries[slice_id + 1]) {
        break;
      }
    }
    // Calculate the starting row index for the slice.
    int slice_start = slice_boundaries[slice_id];
    // Calculate the offset of the current row within the slice.
    long row_offset_in_slice = row - slice_start; // Changed to long
    // Get the padded row length for the current slice.
    int padded_row_length = slice_lengths[slice_id];

    // Iterate over the non-zero elements in the row.
    for (int j = 0; j < padded_row_length; ++j) {
      int global_col_index =
          colind[(slice_start + row_offset_in_slice) * padded_row_length + j];
      double val =
          values[(slice_start + row_offset_in_slice) * padded_row_length + j];
      result += val * x[global_col_index];
    }
    y[row] = result;
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

// Function to generate a random sparse matrix in SELL-P format
void generateRandomSELLPMatrix(long num_rows, long num_cols, long num_nonzeros,
                               int slice_size, int padding_factor,
                               std::vector<int> &slice_boundaries,
                               std::vector<int> &slice_lengths,
                               std::vector<double> &values,
                               std::vector<int> &colind) {
  // Use Mersenne Twister for random number generation
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long> col_dist(0, num_cols - 1);
  std::uniform_real_distribution<> val_dist(-1.0, 1.0);

  // Calculate slice boundaries.
  int num_slices = (num_rows + slice_size - 1) / slice_size; // Calculate num_slices
  slice_boundaries.resize(num_slices + 1);
  slice_boundaries[0] = 0;
  for (int i = 1; i <= num_slices; ++i) {
    slice_boundaries[i] = std::min(num_rows, (long)i * slice_size); // Cast i to long to avoid overflow
  }

  // Calculate slice lengths and maximum row length.
    slice_lengths.resize(num_slices);
    std::vector<int> row_lengths(num_rows, 0);
    long nonzeros_generated = 0; // Changed to long
     for (long i = 0; i < num_rows; ++i) { // Changed to long
        int nonzeros_in_row = 0;
        while (nonzeros_in_row < (num_nonzeros / num_rows) && nonzeros_generated < num_nonzeros)
        {
            long col = col_dist(gen); // Changed to long
            double val = val_dist(gen);
             row_lengths[i]++;
             nonzeros_generated++;
             nonzeros_in_row++;
        }
    }

    for (int i = 0; i < num_slices; ++i)
    {
        int max_len = 0;
        for (int j = slice_boundaries[i]; j < slice_boundaries[i+1]; ++j)
        {
            max_len = std::max(max_len, row_lengths[j]);
        }
        slice_lengths[i] = (max_len + padding_factor - 1) / padding_factor * padding_factor; // Calculate padded length
    }
    //size of the colind and values
    long data_size = 0; // Changed to long
     for (int i = 0; i < num_slices; ++i) {
        data_size += (long)(slice_boundaries[i+1] - slice_boundaries[i]) * slice_lengths[i]; // Cast to long
    }
    values.resize(data_size);
    colind.resize(data_size);

     // Populate the values and colind vectors
    long k = 0; // Changed to long
     for (int i = 0; i < num_slices; ++i) {
        for (int j = slice_boundaries[i]; j < slice_boundaries[i + 1]; ++j) {
            long row_offset_in_slice = j - slice_boundaries[i]; // Changed to long
            for (int l = 0; l < slice_lengths[i]; ++l) {
                if (l < row_lengths[j])
                {
                    long col = col_dist(gen); // Changed to long
                    double val = val_dist(gen);
                    colind[k] = col;
                    values[k] = val;
                }
                else
                {
                    colind[k] = 0;
                    values[k] = 0.0;
                }
                k++;
            }
        }
    }
}

int main() {
  // Host (CPU) data
  long num_rows = 100;
  long num_cols = 100;
  long num_nonzeros = 100;
  int slice_size = 256;  // Example slice size
  int padding_factor = 4; // Example padding factor

  std::vector<int> slice_boundaries;
  std::vector<int> slice_lengths;
  std::vector<double> values;
  std::vector<int> colind;

  // Generate the random SELL-P matrix
  generateRandomSELLPMatrix(num_rows, num_cols, num_nonzeros, slice_size,
                             padding_factor, slice_boundaries, slice_lengths,
                             values, colind);

  std::vector<double> x = generateRandomVector(num_cols);
  std::vector<double> y(num_rows, 0.0);

  // Print the matrix
  std::cout << "Sparse Matrix (SELL-P format):" << std::endl;
  std::cout << "Number of Rows: " << num_rows << std::endl;
    std::cout << "Number of Cols: " << num_cols << std::endl;
    std::cout << "Number of Nonzeros: " << num_nonzeros << std::endl;
  std::cout << "Slice Boundaries: ";
  for (int i = 0; i < slice_boundaries.size(); ++i) {
    //std::cout << slice_boundaries[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "Slice Lengths (Padded): ";
  for (int i = 0; i < slice_lengths.size(); ++i) {
    //std::cout << slice_lengths[i] << " ";
  }
  std::cout << std::endl;


  // Print the input vector
  std::cout << "Input Vector (x):" << std::endl;
  //for (long i = 0; i < num_cols; ++i) {
    //std::cout << std::fixed << std::setprecision(1) << x[i] << " ";
  //}
  std::cout << std::endl;

  // 1. Copy data to the GPU
  int *d_slice_boundaries, *d_slice_lengths, *d_colind;
  double *d_values, *d_x, *d_y;

  // Allocate memory on the GPU
  cudaMalloc((void **)&d_slice_boundaries,
             slice_boundaries.size() * sizeof(int));
  cudaMalloc((void **)&d_slice_lengths, slice_lengths.size() * sizeof(int));
  cudaMalloc((void **)&d_colind, colind.size() * sizeof(int));
  cudaMalloc((void **)&d_values, values.size() * sizeof(double));
  cudaMalloc((void **)&d_x, x.size() * sizeof(double));
  cudaMalloc((void **)&d_y, y.size() * sizeof(double));

  // Copy data from host to device
  cudaMemcpy(d_slice_boundaries, slice_boundaries.data(),
             slice_boundaries.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_slice_lengths, slice_lengths.data(),
             slice_lengths.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_colind, colind.data(), colind.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, values.data(), values.size() * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x.data(), x.size() * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y.data(), y.size() * sizeof(double),
             cudaMemcpyHostToDevice);

  // 2. Configure and launch the kernel
  int threads_per_block = 256;
  int num_blocks = (num_rows + threads_per_block - 1) / threads_per_block;

    auto start_kernel = std::chrono::high_resolution_clock::now();
  sellp_spmv<<<num_blocks, threads_per_block>>>(
      num_rows, slice_size, d_slice_boundaries, d_slice_lengths, d_colind,
      d_values, d_x, d_y, padding_factor);
    cudaDeviceSynchronize();
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
   std::cout << "CUDA kernel execution time: " << duration_kernel.count()
              << " seconds" << std::endl;

  // 4. Free GPU memory
  cudaFree(d_slice_boundaries);
  cudaFree(d_slice_lengths);
  cudaFree(d_colind);
  cudaFree(d_values);
  cudaFree(d_x);
  cudaFree(d_y);

  return 0;
}
