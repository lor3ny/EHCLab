#pragma once

#include "knn.h"


// CUDA kernel function: Runs on the GPU
__global__ void addKernel(int* c, const int* a, const int* b) {
    int i = threadIdx.x; // Get the thread index
    c[i] = a[i] + b[i];  // Perform addition
}

int main() {
    // Define the size of arrays
    const int arraySize = 5;
    const int a[arraySize] = {1, 2, 3, 4, 5};
    const int b[arraySize] = {10, 20, 30, 40, 50};
    int c[arraySize] = {0}; // Result array

    // Device memory pointers
    int *d_a, *d_b, *d_c;

    // Allocate GPU memory
    cudaMalloc((void**)&d_a, arraySize * sizeof(int));
    cudaMalloc((void**)&d_b, arraySize * sizeof(int));
    cudaMalloc((void**)&d_c, arraySize * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel with one block of `arraySize` threads
    addKernel<<<1, arraySize>>>(d_c, d_a, d_b);

    // Copy result back to host
    cudaMemcpy(c, d_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the results
    std::cout << "Result: ";
    for (int i = 0; i < arraySize; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}