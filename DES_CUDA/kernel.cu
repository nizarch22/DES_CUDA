
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DES.h"
#include <stdio.h>
#include <iostream>

__global__ void EncryptDESCuda(uint64_t* messages, uint64_t* keys, unsigned char* matrices, unsigned char* sboxes, uint64_t* results);
__global__ void EncryptDESCudaDebug(uint64_t* messages, uint64_t* keys, unsigned char* matrices, unsigned char* sboxes, uint64_t* results, uint64_t* debug, int n);
void EncryptDESDebug(const uint64_t& plaintext, const uint64_t& key, uint64_t& encryption, uint64_t* debug);
void printCharMatrix(unsigned char* matrix, int y, int x);


#define CHECK_CUDA_ERROR(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) \
        { \
            fprintf(stderr, "CUDA Error: %s (error code %d) at %s:%d\n", \
                    cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } 

int main()
{
    // kernel parameters
    const int numThreads = 256;
    const int numMessages = 524288;// 4MB - 10x speedup//33554432; // 268MB - 223 speedup!
    const int numBlocks = (numMessages + numThreads - 1) / numThreads;

    // size parameters
    int bytesMessages = sizeof(uint64_t) * numMessages;
    int bytesKeys = sizeof(uint64_t) * numMessages;

    // kernel argument prep stage
    // 
    // prep matrices, sboxes
    unsigned char* d_SBoxes, * d_matrices;
    unsigned char* matrices[7] = {IP,PC1,PC2, E, PMatrix,IPInverse, LCS};
    int matricesSizes[7] = { 64,56,48,48,32,64,16 };
    // prep keys, messages, encryptions, decryptions
    uint64_t* d_messages, * d_keys;
    uint64_t* messages = (uint64_t*)malloc(bytesMessages);
    uint64_t* keys = (uint64_t*)malloc(bytesKeys);
    for (int i = 0; i < numMessages; i++)
    {
        messages[i] = (((uint64_t)rand()) << 32) | rand();
        keys[i] = (((uint64_t)rand()) << 32) | rand();
    }
    
    // prep results
    uint64_t* d_resultsEncryption, * d_resultsDecryption;
    uint64_t* resultsEncryption = (uint64_t*)malloc(bytesMessages);
    for (int i = 0; i < numMessages; i++)
    {
        resultsEncryption[i] = 100;
    }
    uint64_t* resultsDecryption = (uint64_t*)malloc(bytesMessages);
    // CPU-run DES Results
    uint64_t* encryptions = (uint64_t*)malloc(bytesMessages);
    uint64_t* decryptions = (uint64_t*)malloc(bytesMessages);

    int startTimeAlloc = clock();
    // cuda allocate memory - matrices, sboxes
    const int matricesSize = 328;
    const int sboxesSize= 512;
    cudaMalloc(&d_matrices, matricesSize);
    cudaMalloc(&d_SBoxes, sboxesSize);
    // cuda allocate memory - messages, keys
    cudaMalloc(&d_messages, bytesMessages);
    cudaMalloc(&d_keys, bytesKeys);
    // cuda allocate memory - results
    cudaMalloc(&d_resultsEncryption, bytesMessages);
    cudaMalloc(&d_resultsDecryption, bytesMessages);

    int startTime = clock();
    // cuda copy memory - matrices, sboxes
    cudaMemcpy(d_SBoxes, &SBoxes[0][0], 64*8, cudaMemcpyHostToDevice);
    int offset = 0;
    for (int i = 0; i < 7; i++)
    {
        cudaMemcpy(d_matrices + offset, &matrices[i][0], matricesSizes[i], cudaMemcpyHostToDevice);
        offset += matricesSizes[i];
    }
    // cuda copy memory - messages, keys
    cudaMemcpy(d_messages, messages, bytesMessages, cudaMemcpyHostToDevice);
    cudaMemcpy(d_keys, keys, bytesKeys, cudaMemcpyHostToDevice);

    std::cout << resultsEncryption[10] << "\n";
    // Encryption cuda stage
    //
    //
    //EncryptDESCuda<<<numBlocks,numThreads>>>(d_messages, d_keys, d_matrices, d_SBoxes, d_resultsEncryption);

    //// results retrieval stage
    ////
    ////
    //cudaMemcpy(resultsEncryption, d_resultsEncryption, bytesMessages, cudaMemcpyDeviceToHost);
    //cudaDeviceSynchronize(); // remove?
    //for (int i = 0; i < numMessages; i++)
    //{
    //    //printMatrix(resultsEncryption[i], 8, 8);
    //}
    //
    //// CPU validate encryption results stage
    ////
    ////
    //int bSame = 1;
    //uint64_t message, key, encryption;
    //for (int i = 0; i < numMessages; i++)
    //{
    //    message = messages[i]; key = keys[i];
    //    EncryptDES(messages[i], keys[i], encryption);
    //    bSame &= encryption == resultsEncryption[i];
    //    if (!bSame)
    //    {
    //        //std::cout << "Operation failed!\n";
    //        //printMatrix(encryption, 8, 8);
    //    }
    //}
    // Debugging stage
    // 
    // 
    //const int numDebugs = 12;
    //const int numTotalDebugs = numDebugs * numMessages;
    //const int sizeDebug = (numTotalDebugs) * sizeof(uint64_t);
    //uint64_t* arrDebug = (uint64_t*)malloc(sizeDebug);
    //uint64_t* cudaArrDebug = (uint64_t*)malloc(sizeDebug);
    //// malloc
    //uint64_t* d_arrDebug;
    //cudaMalloc(&d_arrDebug, sizeDebug);
    //EncryptDESCudaDebug << <numBlocks, numThreads >> > (d_messages, d_keys, d_matrices, d_SBoxes, d_resultsEncryption, d_arrDebug, numDebugs);
    //cudaMemcpy(cudaArrDebug, d_arrDebug, sizeDebug, cudaMemcpyDeviceToHost);
    EncryptDESCuda<<<numBlocks, numThreads>>> (d_messages, d_keys, d_matrices, d_SBoxes, d_resultsEncryption);
    CHECK_CUDA_ERROR(cudaGetLastError());
    // copy result from cuda
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // Use the macro to check for errors
    CHECK_CUDA_ERROR(cudaMemcpy(resultsEncryption, d_resultsEncryption, bytesMessages, cudaMemcpyDeviceToHost));
    int endTime = clock();
    int CUDATime = endTime - startTimeAlloc;
    int CUDATimeCopy = endTime - startTime;
    startTime = clock();
    for (int i = 0; i < numMessages; i++)
    {
        EncryptDES(messages[i], keys[i], encryptions[i]);
        //EncryptDESDebug(messages[i], keys[i], encryptions[i], arrDebug);
    }
    endTime = clock();
    int CPUTime = endTime - startTime;

    std::cout << "CUDA Debug results:\n";

    std::cout << "Total time to allocate memory + copy memory back and forth:\n";
    std::cout << "GPU: " << CUDATime << "ms\n";
    std::cout << "CPU: " << CPUTime << "ms\n";
    std::cout << "GPU - only since copying: " << CUDATimeCopy << "ms\n";
    double speedup = (float)CPUTime / CUDATime;
    double speedupCopy = (float)CPUTime / CUDATimeCopy;
    std::cout << "Total speedup: " << speedup << "\n";
    std::cout << "speedup without counting allocation: " << speedupCopy << "\n";

    // confirming that indeed we have the correction results
    bool bEqual = 1;
    for (int i = 0; i < numMessages; i++)
    {
        bEqual &= (encryptions[i] == resultsEncryption[i]);
        if (!bEqual)
        {
            std::cout << "Debug GPU-CPU comparison failed at " << i << "!\n";
            std::cout << encryptions[i] << "\n";
            std::cout << resultsEncryption[i] << "\n";

            return 0;
        }
    }


    // confirm that CPU ones are good
    bEqual = 1;
    uint64_t* testing = (uint64_t*)malloc(bytesMessages);
    for (int i = 0; i < numMessages; i++)
    {
        EncryptDES(messages[i], keys[i], testing[i]);
        bEqual &= (encryptions[i] == testing[i]);
    }
    if (!bEqual)
    {
        std::cout << "CPU not equal!\n";
    }

    for (int i = 0; i < numMessages; i++)
    {
        bEqual &= (testing[i] == resultsEncryption[i]);
    }
    if (!bEqual)
    {
        std::cout << "Debug GPU-CPU comparison failed!\n";
        return 0;
    }
     //Decryption cuda stage
    
    
    return 0;
}



void printCharMatrix(unsigned char* matrix, int y, int x)
{
    //bool bit;
    //bool mask = 1;
    for (int i = 0; i < y; i++)
    {
        for (int j = 0; j < x; j++)
        {

            //bit = matrix & mask;
            std::cout << matrix[i*y+j] << ",";
            //matrix >>= 1;
        }
        std::cout << "\n";
    }
    std::cout << "Matrix printed.\n";
}

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void addKernel(int* c, const int* a, const int* b)
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

