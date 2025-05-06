#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <random>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <set>

// Helper function to generate a random sparse CSR matrix on the host
void generateRandomCSRMatrix(long num_rows, long num_cols, std::vector<int>& row_offsets, std::vector<int>& col_indices, std::vector<double>& values, long num_nonzeros) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> col_dist(0, num_cols - 1);
    std::uniform_real_distribution<> val_dist(-1.0, 1.0);

    row_offsets.resize(num_rows + 1);
    row_offsets[0] = 0;
    long current_nnz = 0;
    for (long i = 0; i < num_rows; ++i) {
        long nonzeros_in_row = 0;
        // Limit the number of nonzeros in a row to keep it sparse
        while (nonzeros_in_row < (num_nonzeros / num_rows) && current_nnz < num_nonzeros) {
            int col = col_dist(gen);
            double val = val_dist(gen);
            col_indices.push_back(col);
            values.push_back(val);
            nonzeros_in_row++;
            current_nnz++;
        }
        row_offsets[i + 1] = current_nnz;
    }
}

// Helper function to generate a random sparse vector on the host
void generateRandomSparseVector(long size, std::vector<int>& indices, std::vector<double>& values, int num_nonzeros) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> index_dist(0, size - 1);
    std::uniform_real_distribution<> val_dist(-1.0, 1.0);
    std::set<int> unique_indices; // Use a set to ensure unique indices

    while (unique_indices.size() < num_nonzeros) {
        unique_indices.insert(index_dist(gen));
    }

    indices.clear();
    values.clear();
    for (int index : unique_indices) {
        indices.push_back(index);
        values.push_back(val_dist(gen));
    }
    // Sort indices for CSR-vector multiplication (optional, but often good practice)
    std::sort(indices.begin(), indices.end());
}

int main() {
    // Problem size
    long num_rows = 100;
    long num_cols = 100;
    long num_matrix_nonzeros = 100; // Number of non-zero elements in the matrix
    int num_vector_nonzeros = 100;    // Number of non-zero elements in the sparse vector

    // Host-side CSR matrix data
    std::vector<int> h_csr_row_offsets;
    std::vector<int> h_csr_col_indices;
    std::vector<double> h_csr_values;
    generateRandomCSRMatrix(num_rows, num_cols, h_csr_row_offsets, h_csr_col_indices, h_csr_values, num_matrix_nonzeros);

    // Host-side sparse vector data
    std::vector<int> h_vector_indices;
    std::vector<double> h_vector_values;
    generateRandomSparseVector(num_cols, h_vector_indices, h_vector_values, num_vector_nonzeros);

    // Host-side dense output vector
    std::vector<double> h_output_vector(num_rows, 0.0);

    // Device-side pointers
    int* d_csr_row_offsets = nullptr;
    int* d_csr_col_indices = nullptr;
    double* d_csr_values = nullptr;
    int* d_vector_indices = nullptr;
    double* d_vector_values = nullptr;
    double* d_output_vector = nullptr;

    // cuSPARSE variables
    cusparseHandle_t cusparse_handle = nullptr;
    cusparseMatDescr_t mat_descr = nullptr;
    cusparseSpVecDescr_t vec_descr = nullptr;
    cusparseDnVecDescr_t out_vec_descr = nullptr;
    cusparseSpMatDescr_t matA = nullptr; // Sparse matrix descriptor for CSR
    cudaError_t cuda_err;
    cusparseStatus_t cusparse_err;
    double alpha = 1.0;
    double beta = 0.0;
    size_t buffer_size = 0;
    void* d_buffer = nullptr;

    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    float elapsed_time = 0.0f;

    // Allocate device memory
    cuda_err = cudaMalloc((void**)&d_csr_row_offsets, (num_rows + 1) * sizeof(int));
    if (cuda_err != cudaSuccess) goto cleanup;
    cuda_err = cudaMalloc((void**)&d_csr_col_indices, h_csr_col_indices.size() * sizeof(int));
    if (cuda_err != cudaSuccess) goto cleanup;
    cuda_err = cudaMalloc((void**)&d_csr_values, h_csr_values.size() * sizeof(double));
    if (cuda_err != cudaSuccess) goto cleanup;
    cuda_err = cudaMalloc((void**)&d_vector_indices, h_vector_indices.size() * sizeof(int));
    if (cuda_err != cudaSuccess) goto cleanup;
    cuda_err = cudaMalloc((void**)&d_vector_values, h_vector_values.size() * sizeof(double));
    if (cuda_err != cudaSuccess) goto cleanup;
    cuda_err = cudaMalloc((void**)&d_output_vector, num_rows * sizeof(double));
    if (cuda_err != cudaSuccess) goto cleanup;

    // Copy host data to device
    cuda_err = cudaMemcpy(d_csr_row_offsets, h_csr_row_offsets.data(), (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) goto cleanup;
    cuda_err = cudaMemcpy(d_csr_col_indices, h_csr_col_indices.data(), h_csr_col_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) goto cleanup;
    cuda_err = cudaMemcpy(d_csr_values, h_csr_values.data(), h_csr_values.size() * sizeof(double), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) goto cleanup;
    cuda_err = cudaMemcpy(d_vector_indices, h_vector_indices.data(), h_vector_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) goto cleanup;
    cuda_err = cudaMemcpy(d_vector_values, h_vector_values.data(), h_vector_values.size() * sizeof(double), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) goto cleanup;
    cuda_err = cudaMemcpy(d_output_vector, h_output_vector.data(), num_rows * sizeof(double), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) goto cleanup;

    // Initialize cuSPARSE
    cusparse_err = cusparseCreate(&cusparse_handle);
    if (cusparse_err != CUSPARSE_STATUS_SUCCESS) goto cleanup;

    // Create matrix descriptor
    cusparse_err = cusparseCreateMatDescr(&mat_descr);
    if (cusparse_err != CUSPARSE_STATUS_SUCCESS) goto cleanup;
    cusparse_err = cusparseSetMatType(mat_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    if (cusparse_err != CUSPARSE_STATUS_SUCCESS) goto cleanup;
    cusparse_err = cusparseSetMatIndexBase(mat_descr, CUSPARSE_INDEX_BASE_ZERO);
    if (cusparse_err != CUSPARSE_STATUS_SUCCESS) goto cleanup;

     // Create sparse matrix descriptor for CSR
    cusparse_err = cusparseCreateCsr(&matA,
                                      num_rows,
                                      num_cols,
                                      h_csr_values.size(), // Use the size of the non-zero values
                                      d_csr_row_offsets,
                                      d_csr_col_indices,
                                      d_csr_values,
                                      CUSPARSE_INDEX_32I, // Type of row offsets
                                      CUSPARSE_INDEX_32I, // Type of column indices
                                      CUSPARSE_INDEX_BASE_ZERO, // Indexing base
                                      CUDA_R_64F); // Type of values
    if (cusparse_err != CUSPARSE_STATUS_SUCCESS) goto cleanup;

    // Create sparse vector descriptor
    cusparse_err = cusparseCreateSpVec(&vec_descr,
                                      num_cols,
                                      num_vector_nonzeros,
                                      d_vector_indices,
                                      d_vector_values,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO,
                                      CUDA_R_64F);
    if (cusparse_err != CUSPARSE_STATUS_SUCCESS) goto cleanup;

    // Create dense output vector descriptor
    cusparse_err = cusparseCreateDnVec(&out_vec_descr, num_rows, d_output_vector, CUDA_R_64F);
    if (cusparse_err != CUSPARSE_STATUS_SUCCESS) goto cleanup;


    // Allocate the buffer
    cuda_err = cudaMalloc(&d_buffer, 1000000000000);
    if (cuda_err != cudaSuccess) goto cleanup;

     // Create CUDA events for timing
    cuda_err = cudaEventCreate(&start_event);
    if (cuda_err != cudaSuccess) goto cleanup;
    cuda_err = cudaEventCreate(&stop_event);
    if (cuda_err != cudaSuccess) goto cleanup;

    // Record the start event
    cuda_err = cudaEventRecord(start_event, 0);
    if (cuda_err != cudaSuccess) goto cleanup;
    // Perform the sparse matrix-sparse vector multiplication (csrmv)
    cusparse_err = cusparseSpMV(cusparse_handle,
                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                              &alpha,
                              matA, // Use matA here
                              (cusparseConstDnVecDescr_t)vec_descr,
                              &beta,
                              out_vec_descr,
                              CUDA_R_64F,
                              CUSPARSE_SPMV_ALG_DEFAULT,
                              d_buffer);
    if (cusparse_err != CUSPARSE_STATUS_SUCCESS) goto cleanup;

     // Record the stop event
    cuda_err = cudaEventRecord(stop_event, 0);
    if (cuda_err != cudaSuccess) goto cleanup;

    // Wait for the stop event to complete
    cuda_err = cudaEventSynchronize(stop_event);
    if (cuda_err != cudaSuccess) goto cleanup;

    // Calculate the elapsed time
    cuda_err = cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
    if (cuda_err != cudaSuccess) goto cleanup;
    std::cout << "Time taken by cusparseSpMV: " << elapsed_time << " ms" << std::endl;

    // Copy the result back to the host
    cuda_err = cudaMemcpy(h_output_vector.data(), d_output_vector, num_rows * sizeof(double), cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess) goto cleanup;

    // Print the first 10 elements of the result
    std::cout << "Result (first 10 values): ";
    for (int i = 0; i < std::min((long)10, num_rows); ++i) {
        std::cout << std::fixed << std::setprecision(6) << h_output_vector[i] << " ";
    }
    std::cout << std::endl;

cleanup:
    // Clean up
    if (cusparse_handle != nullptr) cusparseDestroy(cusparse_handle);
    if (mat_descr != nullptr) cusparseDestroyMatDescr(mat_descr);
     if (matA != nullptr)  cusparseDestroySpMat(matA);
    if (vec_descr != nullptr) cusparseDestroySpVec(vec_descr);
    if (out_vec_descr != nullptr) cusparseDestroyDnVec(out_vec_descr);
    if (d_csr_row_offsets != nullptr) cudaFree(d_csr_row_offsets);
    if (d_csr_col_indices != nullptr) cudaFree(d_csr_col_indices);
    if (d_csr_values != nullptr) cudaFree(d_csr_values);
    if (d_vector_indices != nullptr) cudaFree(d_vector_indices);
    if (d_vector_values != nullptr) cudaFree(d_vector_values);
    if (d_output_vector != nullptr) cudaFree(d_output_vector);
    if (d_buffer != nullptr) cudaFree(d_buffer);
     if (start_event != nullptr) cudaEventDestroy(start_event);
    if (stop_event != nullptr) cudaEventDestroy(stop_event);

    // Check for any CUDA errors that occurred during the cleanup process
    if (cuda_err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cuda_err) << std::endl;
        return 1;
    }
    if (cusparse_err != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "cuSPARSE error: " << cusparseGetErrorString(cusparse_err) << std::endl;
        return 1;
    }

    return 0;
}
