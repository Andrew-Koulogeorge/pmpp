// test.cu
#include <stdio.h>

__global__ void hello() {
    printf("Hello from GPU thread %d in block %d!\n", threadIdx.x, blockIdx.x);
}

int main() {
    hello<<<2, 4>>>();  // 2 blocks, 4 threads each
    cudaDeviceSynchronize();
    printf("Done!\n");
    return 0;
}