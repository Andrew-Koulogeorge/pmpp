/**
 * Hello World for CUDA 
 * Program for element wise addition 
 */

#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

// device method that computes element wise addition between vectors
 __global__ 
 void vecAddKernel(float *A, float *B, float *C, int n){
    // compute idx of this thread based on block.size, block.idx and thread.idx
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        C[i] = A[i] + B[i];
    }
 }

 // host method vecAdd to handle device allocation and copying
 void vecAdd(float *A_h, float *B_h, float *C_h, int n){
    
    // allocate memory on the GPU for A_d,B_d,C_d
    int size = n*sizeof(float);
    float *A_d, *B_d, *C_d;
    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);

    // copy memory from host to device
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // launch kernel with correct number of blocks
    int block_size = 32;
    int num_blocks = ceil(n / (double)block_size);
    printf("num_blocks: %d | block_suze: %d", num_blocks, block_size)
    vecAddKernel<<<num_blocks, block_size>>>(A_d, B_d, C_d, n);

    // copy memory of output C_d -> C_h
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    // free all memory on device
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
 }

  // main method to allocate arrays on host to be added together
 int main(int argc, char* argv[]){
    if (argc != 2){
        return -1;
    }
    
    // read in parameter n from command line (length of arrays)
    int n = atoi(argv[1]);

    // allocate [i for i in range(n)] and [2*i for i in range(n)]
    int size = n*sizeof(float);
    float *A_h = (float*) malloc(size);
    float *B_h = (float*) malloc(size);    
    for(int i = 0; i < n; i++){
        A_h[i] = i;
        B_h[i] = 2*i;
    }

    // allocate empty result array
    float *C_h = (float*) malloc(size);

    // call vecAdd stub
    vecAdd(A_h,B_h,C_h,n);

    // print out the values of A,B and C
    printf("Printing last 5 values: \n");
    for(int i = n-5; i < n; i++){
        printf("%.0f + %.0f = %.0f\n", A_h[i], B_h[i], C_h[i]);
    }
    return 0;
 }