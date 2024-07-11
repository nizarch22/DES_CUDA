
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DES_Matrices_NIST.h"

#include <stdio.h>
#include <iostream>

void printCharMatrix(unsigned char** matrix, int y, int x)
{
    //bool bit;
    //bool mask = 1;
    for (int i = 0; i < y; i++)
    {
        for (int j = 0; j < x; j++)
        {

            //bit = matrix & mask;
            std::cout << matrix[i][j] << ",";
            //matrix >>= 1;
        }
        std::cout << "\n";
    }
    std::cout << "Matrix printed.\n";
}

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void lengthTest(unsigned char* matrices, unsigned char* sboxes, unsigned char* results)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    results[tid] = sboxes[tid];
}

__global__ void cudaTest(unsigned char* a, unsigned char* b)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    b[tid] = tid;
}

// major helper functions
//__device__ void permuteMatrixCuda()
//{
//
//}
//__device__ void substituteCuda()
//{
//
//}
//__device__ void leftCircularShiftCuda()
//{
//
//}
//__device__ void 
__global__ void EncryptDESCuda(uint64_t* messages, uint64_t* keys, uint64_t* matrices, uint64_t* results);

void testPrint()
{
    unsigned char* matrices[7] = { IP,PC1,PC2, E, PMatrix,IPInverse, LCS };
    int matricesSizes[7] = { 64,56,48,48,32,64,16 };
    // cuda call memory test
    unsigned char* arr; unsigned char* result;
    arr = (unsigned char*)malloc(328);
    result = (unsigned char*)malloc(328);

    // setup memory
    arr[0] = (char)244;
    arr[1] = (char)211;

    unsigned char* d_arr; unsigned char* d_result;
    cudaMalloc(&d_arr, 328);
    cudaMalloc(&d_result, 328);

    // copy arr memory
    int offset = 0;
    for (int i = 0; i < 7; i++)
    {
        cudaMemcpy(d_arr + offset, &matrices[i][0], matricesSizes[i], cudaMemcpyHostToDevice);
        offset += matricesSizes[i];
    }
    // run cuda
    cudaTest << <1, 328 >> > (d_arr, d_result);
    cudaMemcpy(result, d_result, 328, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    offset = 0;
    std::cout << "Result:\n";
    for (int i = 0; i < 7; i++)
    {
        for (int j = 0; j < matricesSizes[i]; j++)
            std::cout << (int)result[offset + j] << ",";
        std::cout << "\n\n";
        offset += matricesSizes[i];
    }
}

void testPointerPrint()
{
    // doing testing
    unsigned char* ptrTest = &(SBoxes[1][0]);
    std::cout << SBoxes << "\n";
    std::cout << &SBoxes << "\n";
    std::cout << &(SBoxes[1]) << "\n";
    std::cout << ptrTest << "\n";
    std::cout << &(SBoxes[1][0]) << "\n";
    std::cout << (int) *ptrTest << "\n";
    std::cout << (int) *(ptrTest + 1) << "\n";
    std::cout << (int)SBoxes[1][0] << "\n";
    std::cout << (int)SBoxes[1][1] << "\n";

    // copy test
    unsigned char* test = (unsigned char*)malloc(64*8);
    //memcpy(test, SBoxes, 64*8);

    // copying each row of 64
    unsigned char* temp1, * temp2;
    for (int i = 0; i < 8; i++)
    {
        temp1 = test + i * 64;
        temp2 = &SBoxes[i][0];
        memcpy(temp1, temp2, 64);
    }

    // printout
    for (int i = 0; i < 8; i++)
    {
        std::cout << (int)test[i*64] << ",";
    }
    std::cout << "\n";
}
int main()
{
    const int numThreads = 1;
    const int arraySize = 1;
    const int numBlocks = (arraySize + numThreads - 1) / numThreads;
    const int bytes = sizeof(int) * arraySize;
    //load messages and keys
    //messages
    //keys

    //load matrices - 7. SBox
    // IP, PC1,E, PC2, SBox, PMatrix, IPInverse
    //unsigned char* d_IP, * d_PC1, * d_PC2, * d_E, * d_PMatrix, * d_IPInverse, * d_LCS;
    unsigned char* d_SBoxes;
    unsigned char* d_matrices;
    unsigned char* matrices[7] = {IP,PC1,PC2, E, PMatrix,IPInverse, LCS};
    int matricesSizes[7] = { 64,56,48,48,32,64,16 };

    // cuda allocate memory
    cudaMalloc(&d_matrices, 328);
    cudaMalloc(&d_SBoxes, 8*64);

    // copy memory
    cudaMemcpy(d_SBoxes, &SBoxes[0][0], 64*8, cudaMemcpyHostToDevice);

    int offset = 0;
    for (int i = 0; i < 7; i++)
    {
        cudaMemcpy(d_matrices + offset, &matrices[i][0], matricesSizes[i], cudaMemcpyHostToDevice);
        offset += matricesSizes[i];
    }

    // call cuda
    int threadNUM = 64;
    int blockNUM = 2;
    int totalThreadNUM = blockNUM * threadNUM;
    unsigned char* results = (unsigned char*)malloc(totalThreadNUM);
    unsigned char* d_results;
    cudaMalloc(&d_results, totalThreadNUM);

    lengthTest <<<1, threadNUM >>> (d_matrices, d_SBoxes, d_results);

    cudaMemcpy(results, d_results, totalThreadNUM, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int i = 0; i < totalThreadNUM; i++)
    {
        std::cout << (int)results[i] << ",";
    }

    return 0;
    // break here.
    int* c = (int*)malloc(bytes+sizeof(int));
    int* d_c;
    cudaMalloc(&d_c, bytes+4);
    for (int i = 0; i < arraySize; i++)
    {
        c[i] = 1000;
    }
    cudaMemcpy(d_c, c, (bytes), cudaMemcpyHostToDevice);

    // Add vectors in parallel.
    //EncryptDESCuda <<<1, 1>>>(d_c);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return 1;
    //}
   cudaMemcpy(c, d_c, bytes+4, cudaMemcpyDeviceToHost);
   for (int i = 0; i < arraySize+1; i++)
   {
       std::cout << c[i] << ",";
   }
   std::cout << "\n";
   // printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        //c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceReset failed!");
    //    return 1;
    //}

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
