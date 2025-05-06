#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <random>
#include <algorithm>
#include <iomanip>
#include <chrono>

// CUDA Kernel (from the paper, with corrections)
__global__ void sellp_spmv(
    long n, int b, int t, double alpha,
    int *rowptr, int *colind, double *values,
    double *x, double beta, double *y) {

    int idx = threadIdx.x;
    int idy = threadIdx.y;
    int ldx = idy * t + idx;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x;
    long row = bdx * b + idy;

    extern __shared__ double shared[];

    if (row < n) {
        double dot = 0.0;
        int offset = rowptr[bdx];
        int block = b * t;
        int row_end = rowptr[bdx + 1];
        int num_blocks_in_row = (row_end - offset + block - 1) / block;

        for (int kk = 0; kk < num_blocks_in_row; ++kk) {
            int col_index = offset + kk * t + idx;
            if (col_index < row_end) {
                int col = colind[col_index];
                dot += values[col_index] * x[col];
            }
        }

        shared[ldx] = dot;
        __syncthreads();

        // Reduction (simplified)
        for (int s = b / 2; s > 0; s /= 2) {
            if (idx < s) {
                shared[ldx] += shared[ldx + s * t];
            }
            __syncthreads();
        }

        if (idx == 0 && idy == 0) {
            y[row] = shared[0] * alpha + beta * y[row];
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

// Function to generate a random sparse matrix in SELL-P format
void generateRandomSELLPMatrix(long num_rows, long num_cols, int b, int t,
    std::vector<int>& rowptr, std::vector<int>& colind, std::vector<double>& values, long num_nonzeros) {
    // Use Mersenne Twister for random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> col_dist(0, num_cols - 1);
    std::uniform_real_distribution<> val_dist(-1.0, 1.0);

    rowptr.resize(num_rows + 1);
    rowptr[0] = 0;
    long current_nnz = 0;

    for (long i = 0; i < num_rows; ++i) {
        long nonzeros_in_row = 0;
        while (nonzeros_in_row < (num_nonzeros / num_rows) && current_nnz < num_nonzeros) {
            int col = col_dist(gen);
            double val = val_dist(gen);
            colind.push_back(col);
            values.push_back(val);
            nonzeros_in_row++;
            current_nnz++;
        }
        rowptr[i + 1] = current_nnz;
    }
}

int main() {
    // Host-side data
    long n = 100;
    long num_cols = 100;
    int b = 32;
    int t = 8;
    double alpha = 1.0;
    double beta = 0.0;
    long num_nonzeros = 100;

    // Generate random matrix and vector
    std::vector<int> h_rowptr;
    std::vector<int> h_colind;
    std::vector<double> h_values;
    generateRandomSELLPMatrix(n, num_cols, b, t, h_rowptr, h_colind, h_values, num_nonzeros);
    std::vector<double> h_x = generateRandomVector(num_cols);
    std::vector<double> h_y(n, 0.0);

    std::cout << "Number of Rows: " << n << std::endl;
    std::cout << "Number of Cols: " << num_cols << std::endl;
    std::cout << "Block Size (b): " << b << std::endl;
    std::cout << "Threads per block dimension (t): " << t << std::endl;
    std::cout << "Number of Nonzeros: " << num_nonzeros << std::endl;

    // Device memory pointers
    int *d_rowptr, *d_colind;
    double *d_values, *d_x, *d_y;

    // Allocate memory on the device
    cudaMalloc((void **)&d_rowptr, h_rowptr.size() * sizeof(int));
    cudaMalloc((void **)&d_colind, h_colind.size() * sizeof(int));
    cudaMalloc((void **)&d_values, h_values.size() * sizeof(double));
    cudaMalloc((void **)&d_x, h_x.size() * sizeof(double));
    cudaMalloc((void **)&d_y, h_y.size() * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(d_rowptr, h_rowptr.data(), h_rowptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colind, h_colind.data(), h_colind.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values.data(), h_values.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), h_y.size() * sizeof(double), cudaMemcpyHostToDevice);

    // Configure the grid and block dimensions
    dim3 blockDim(t, b);
    dim3 gridDim((n + b - 1) / b);

    // Launch the kernel
    auto start_kernel = std::chrono::high_resolution_clock::now();
    sellp_spmv<<<gridDim, blockDim, b * t * sizeof(double)>>>(
        n, b, t, alpha,
        d_rowptr, d_colind, d_values,
        d_x, beta, d_y);
    cudaDeviceSynchronize();
    auto end_kernel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_kernel = end_kernel - start_kernel;

    // Copy the result back to the host
    cudaMemcpy(h_y.data(), d_y, n * sizeof(double), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Result vector y: " << std::endl;
    //for (double val : h_y) {
        //std::cout << std::fixed << std::setprecision(2) << val << " ";
    //}
    std::cout << std::endl;
    std::cout << "CUDA kernel execution time: " << duration_kernel.count()
              << " seconds" << std::endl;

    // Free device memory
    cudaFree(d_rowptr);
    cudaFree(d_colind);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
