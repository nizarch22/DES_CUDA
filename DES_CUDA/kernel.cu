﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DES.h"
#include <stdio.h>
#include <iostream>

__global__ void EncryptDESCuda(uint64_t* messages, uint64_t* keys, unsigned char* matrices, unsigned char* sboxes, uint64_t* results);
__global__ void EncryptDESCudaDebug(uint64_t* messages, uint64_t* keys, unsigned char* matrices, unsigned char* sboxes, uint64_t* results, uint64_t* debug, int n);
void EncryptDESDebug(const uint64_t& plaintext, const uint64_t& key, uint64_t& encryption, uint64_t* debug);
void printCharMatrix(unsigned char* matrix, int y, int x);
int main()
{
    // kernel parameters
    const int numThreads = 512;
    const int numMessages = 4194304;
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
    uint64_t* resultsDecryption = (uint64_t*)malloc(bytesMessages);
    // CPU-run DES Results
    uint64_t* encryptions = (uint64_t*)malloc(bytesMessages);
    uint64_t* decryptions = (uint64_t*)malloc(bytesMessages);

    int startTimeAlloc = clock();
    // cuda allocate memory - matrices, sboxes
    cudaMalloc(&d_matrices, 328);
    cudaMalloc(&d_SBoxes, 8*64);
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
    const int numDebugs = 12;
    const int numTotalDebugs = numDebugs * numMessages;
    const int sizeDebug = (numTotalDebugs) * sizeof(uint64_t);
    uint64_t* arrDebug = (uint64_t*)malloc(sizeDebug);
    uint64_t* cudaArrDebug = (uint64_t*)malloc(sizeDebug);
    // malloc
    uint64_t* d_arrDebug;
    cudaMalloc(&d_arrDebug, sizeDebug);
    EncryptDESCudaDebug << <numBlocks, numThreads >> > (d_messages, d_keys, d_matrices, d_SBoxes, d_resultsEncryption, d_arrDebug, numDebugs);
    // copy result from cuda
    cudaMemcpy(cudaArrDebug, d_arrDebug, sizeDebug, cudaMemcpyDeviceToHost);
    cudaMemcpy(resultsEncryption, d_resultsEncryption, bytesMessages, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    int endTime = clock();
    int CUDATime = endTime - startTimeAlloc;
    int CUDATimeCopy = endTime - startTime;
    startTime = clock();
    for (int i = 0; i < numMessages; i++)
    {
        EncryptDESDebug(messages[i], keys[i], encryptions[i], arrDebug);
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

    //bool bEqual = 1;
    //uint64_t encryption;
    //for (int j = 0; j < numMessages; j++)
    //{
    //    //EncryptDESDebug(messages[j], keys[j], encryption, arrDebug);
    //    for (int i = 0; i < numDebugs-1; i++)
    //    {
    //        if (cudaArrDebug[i+j*numDebugs] != arrDebug[i])
    //        {
    //            std::cout << "Fail happened at " << i << " on message number " << j << "\n";
    //            printMatrix(cudaArrDebug[i + j * numDebugs], 8, 8);
    //            printMatrix(arrDebug[i], 8, 8);
    //            // extra
    //            std::cout << "CUDA: " << cudaArrDebug[i + j * numDebugs] << "\nCPU: " << arrDebug[i] << "\n";
    //            std::cout << "Extras:\n";
    //            uint64_t extraVariable = messages[0];
    //            std::cout << "CUDA: " << cudaArrDebug[11 + j * numDebugs] << "\nCPU: " << extraVariable << "\n";
    //            //std::cout << cudaArrDebug[11+j*numDebugs] << "\n";
    //            bEqual = 0;
    //            break;
    //        }
    //        bEqual &= (resultsEncryption[j] == encryption);
    //    }
    //}
    //if (bEqual)
    //{
    //    std::cout << "Success!\n";
    //}
    // Decryption cuda stage
    //
    //
    return 0;

   // return 0;
   // // break here.
   // int* c = (int*)malloc(bytes+sizeof(int));
   // int* d_c;
   // cudaMalloc(&d_c, bytes+4);
   // for (int i = 0; i < arraySize; i++)
   // {
   //     c[i] = 1000;
   // }
   // cudaMemcpy(d_c, c, (bytes), cudaMemcpyHostToDevice);

   // // Add vectors in parallel.
   // //EncryptDESCuda <<<1, 1>>>(d_c);
   // //if (cudaStatus != cudaSuccess) {
   // //    fprintf(stderr, "addWithCuda failed!");
   // //    return 1;
   // //}
   //cudaMemcpy(c, d_c, bytes+4, cudaMemcpyDeviceToHost);
   //for (int i = 0; i < arraySize+1; i++)
   //{
   //    std::cout << c[i] << ",";
   //}
   //std::cout << "\n";
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

