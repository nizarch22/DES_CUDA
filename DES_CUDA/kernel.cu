﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DES.h"
#include <stdio.h>
#include <iostream>
#include "DES_CUDA.cuh"
// Checks cuda errors. Exits if detected. 
// This may be helpful in release mode, where the kernel may not run if we demand too many resources.
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
#define DEBUG_ITERATION 100
#define NUM_TESTS 9
#define NUM_TESTS_QUICK 4
int main()
{
    // kernel parameters
    const int numThreads = 64;
    const int numMessages[NUM_TESTS] = { 131072,262144,524288,1048576,2097152,4194304,8388608,16777216,33554432 };// 524288 -  4MB - 10x speedup. 33554432 - 256MB - 70x speedup!
    // size parameters
    const int bytesMessages[NUM_TESTS] = { 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864 , 134217728, 268435456 };
    const int bytesKeys[NUM_TESTS] = { 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864 , 134217728, 268435456 }; // change for keys

    const int bytesLargest = 268435456;
    //// Kernel arguments prep stage ////
    // prep keys, messages, encryptions, decryptions
    uint64_t* d_messages, * d_keys;
    uint64_t* messages = (uint64_t*)malloc(bytesLargest);
    uint64_t* keys = (uint64_t*)malloc(bytesLargest);

    // prep results
    uint64_t* d_resultsEncryption, * d_resultsDecryption;
    uint64_t* resultsEncryption = (uint64_t*)malloc(bytesLargest);
    uint64_t* resultsDecryption = (uint64_t*)malloc(bytesLargest);
    // CPU-run DES Results
    uint64_t* encryptions = (uint64_t*)malloc(bytesLargest);
    uint64_t* decryptions = (uint64_t*)malloc(bytesLargest);

    int startTimeAlloc = clock(); // Used to measure the time GPU finishes execution since allocation started.
    // cuda allocate memory - messages, keys
    cudaMalloc(&d_messages, bytesLargest);
    cudaMalloc(&d_keys, bytesLargest);
    // cuda allocate memory - results
    cudaMalloc(&d_resultsEncryption, bytesLargest);
    cudaMalloc(&d_resultsDecryption, bytesLargest);

    int endTimeAlloc = clock();

    // verification parameters
    bool bEqualDecrypt, bEqualEncrypt;
    bEqualDecrypt = 1; bEqualEncrypt = 1;

    // timing parameters
    int startTimeInputCopy[NUM_TESTS]; int endTimeInputCopy[NUM_TESTS];
    int startTimeExecute[NUM_TESTS]; int endTimeExecute[NUM_TESTS];
    int endTimeRetrieveResults[NUM_TESTS];
    //int startTimeCPU[NUM_TESTS]; int endTimeCPU[NUM_TESTS];

    for (int testCount = 0; testCount < NUM_TESTS_QUICK; testCount++)
    {
        // generating random messages and keys
        for (int i = 0; i < numMessages[testCount]; i++)
        {
            messages[i] = (((uint64_t)rand()) << 32) | rand();
            keys[i] = (((uint64_t)rand()) << 32) | rand();
        }
        // cuda copy memory - messages, keys
        startTimeInputCopy[testCount] = clock();
        cudaMemcpy(d_messages, messages, bytesMessages[testCount], cudaMemcpyHostToDevice);
        cudaMemcpy(d_keys, keys, bytesKeys[testCount], cudaMemcpyHostToDevice);
        endTimeInputCopy[testCount] = clock();

        //// Run Encryption & Decryption in CUDA stage ////
        // Encrypt the messages using EncryptDESCuda. 
        startTimeExecute[testCount] = clock();
        EncryptDESCuda << < numMessages[testCount], numThreads >> > (d_messages, d_keys, d_resultsEncryption);
        cudaDeviceSynchronize(); // wait for encrypt to finish

        // Decrypt all the encryption made by EncryptDesCuda above using DecryptDESCuda.
        DecryptDESCuda << <numMessages[testCount], numThreads >> > (d_resultsEncryption, d_keys, d_resultsDecryption);
        cudaDeviceSynchronize();
        endTimeExecute[testCount] = clock();

        // cuda copy results 
        cudaMemcpy(resultsEncryption, d_resultsEncryption, bytesMessages[testCount], cudaMemcpyDeviceToHost);
        cudaMemcpy(resultsDecryption, d_resultsDecryption, bytesMessages[testCount], cudaMemcpyDeviceToHost);
        endTimeRetrieveResults[testCount] = clock();

        // cuda check for errors in CUDA execution
        CHECK_CUDA_ERROR(cudaGetLastError());

        for (int i = 0; i < numMessages[testCount]; i++)
        {
            bEqualDecrypt &= (resultsDecryption[i] == messages[i]);
        }
    }




    if (!bEqualDecrypt)
    {
        std::cout << "GPU Decryption comparison failed!\n";
        return 0;
    }
    //if (!bEqualDecrypt)
    //{
    //    std::cout << "Decryption-message comparison failed!\n";
    //    return 0;
    //}

    //if (bEqualDecrypt && bEqualEncrypt)
    //{
    //    std::cout << "Success!\n";
    //}

    //// Runtime measurement and calculation stage ////
    // Calculate timings for CUDA, CPU execution. 
    // CUDA has 2 timing calculations: one with allocation time and one without. The reason is that the allocation time is very big, and impactful for small input data (where CPU performs better than the GPU).
    //int CPUTime;
    int CUDATime, CUDATimeCopy, CUDATimeExecute;
    int CUDATimeAlloc = endTimeAlloc - startTimeAlloc;
    
    double throughput, speedup, speedupExecute;

    std::cout << "CUDA Debug results:\n";
    for (int i = 0; i < NUM_TESTS_QUICK; i++)
    {
        CUDATimeExecute = endTimeExecute[i] - startTimeExecute[i];
        CUDATimeCopy = CUDATimeExecute + endTimeInputCopy[i] - startTimeInputCopy[i];
        CUDATime = CUDATimeCopy+CUDATimeAlloc;
        //CPUTime = endTimeCPU[i] - startTimeCPU[i];
        // printout of timing results
        std::cout << "Total messages size: " << (bytesMessages[i] >> 20) << " MegaBytes\n";
        std::cout << "Total timing measurements:\n";
        std::cout << "GPU - only kernel execution: " << CUDATimeExecute << "ms\n";
        std::cout << "GPU - with copying: " << CUDATimeCopy << "ms\n";
        std::cout << "GPU - with allocation and copying: " << CUDATime << "ms\n";
        //std::cout << "CPU: " << CPUTime << "ms\n";

        throughput = 1000.0f * (float)(bytesMessages[i] >> 17) / CUDATimeExecute;
        //speedup = (float)CPUTime / CUDATime;
        //speedupExecute = (float)CPUTime / CUDATimeExecute;
        std::cout << "Speed measurements:\n";
        std::cout << "GPU - Execution speed: " << throughput << " MegaBits per second\n";

        //std::cout << "Speedup measurements:\n";
        //std::cout << "Speedup - Execution: " << speedupExecute << "\n";
        //std::cout << "Speedup - with allocation and copy: " << speedup << "\n";
        std::cout << "#" << i << " Done." << "\n\n";
    }
    //// Memory release stage ////

    // CPU
    free(messages);
    free(keys);
    free(resultsEncryption);
    free(resultsDecryption);
    free(encryptions);
    free(decryptions);
    // GPU/CUDA
    //CHECK_CUDA_ERROR(cudaFree(d_matricesConst));
    //CHECK_CUDA_ERROR(cudaFree(d_SBoxesConst));
    CHECK_CUDA_ERROR(cudaFree(d_messages));
    CHECK_CUDA_ERROR(cudaFree(d_keys));
    CHECK_CUDA_ERROR(cudaFree(d_resultsEncryption));
    CHECK_CUDA_ERROR(cudaFree(d_resultsDecryption));

    return 0;
}