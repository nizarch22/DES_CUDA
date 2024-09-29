
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

int main()
{
    // kernel parameters
    const int numThreads = 256;
    const int numMessages = 33554432;// 524288 -  4MB - 10x speedup. 33554432 - 256MB - 70x speedup!
    const int numBlocks = (numMessages + numThreads - 1) / numThreads;

    // size parameters
    const int bytesMessages = sizeof(uint64_t) * numMessages;
    const int bytesKeys = sizeof(uint64_t) * numMessages;

    //// Kernel arguments prep stage ////
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

    int startTimeAlloc = clock(); // Used to measure the time GPU finishes execution since allocation started.
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

    int startTimeCopy = clock(); // Used to measure the time GPU finishes execution since copying started.
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

    //// Run Encryption & Decryption in CUDA stage ////
    // We encrypt the messages using EncryptDESCuda. Then, we use all those encrypted messages to run DecryptDESCuda.
    int startTimeExecute = clock();
    EncryptDESCuda << <numBlocks, numThreads >> > (d_messages, d_keys, d_matrices, d_SBoxes, d_resultsEncryption);
    cudaDeviceSynchronize(); // wait for encrypt to finish
    DecryptDESCuda << <numBlocks, numThreads >> > (d_resultsEncryption, d_keys, d_matrices, d_SBoxes, d_resultsDecryption);
    cudaDeviceSynchronize();
    int endTimeExecute = clock();
    // cuda copy results 
    cudaMemcpy(resultsEncryption, d_resultsEncryption, bytesMessages, cudaMemcpyDeviceToHost);
    cudaMemcpy(resultsDecryption, d_resultsDecryption, bytesMessages, cudaMemcpyDeviceToHost);
    int endTimeGPU = clock();

    // cuda check for errors in CUDA execution
    CHECK_CUDA_ERROR(cudaGetLastError());


    //// Runtime measurement and calculation stage ////
    // Calculate timings for CUDA, CPU execution. 
    // CUDA has 2 timing calculations: one with allocation time and one without. The reason is that the allocation time is very big, and impactful for small input data (where CPU performs better than the GPU).

    int startTimeCPU = clock();
    for (int i = 0; i < numMessages; i++)
    {
        EncryptDES(messages[i], keys[i], encryptions[i]);
        DecryptDES(encryptions[i], keys[i], decryptions[i]);
    }
    int endTimeCPU = clock();
    int CPUTime = endTimeCPU - startTimeCPU;
    int CUDATime = endTimeGPU - startTimeAlloc;
    int CUDATimeCopy = endTimeGPU - startTimeCopy;
    int CUDATimeExecute = endTimeExecute - startTimeExecute;
    // printout of timing results
    std::cout << "CUDA Debug results:\n";
    std::cout << "Total messages size: " << (bytesMessages >> 20) << "MB\n";
    std::cout << "Total timing measurements:\n";
    std::cout << "GPU - with allocation: " << CUDATime << "ms\n";
    std::cout << "GPU - with copying: " << CUDATimeCopy << "ms\n";
    std::cout << "GPU - only kernel execution: " << CUDATimeExecute << "ms\n";
    std::cout << "CPU: " << CPUTime << "ms\n";
    double throughput = 8.0f*1000.0f*(float)(bytesMessages >> 20) / CUDATimeExecute;
    double speedup = (float)CPUTime / CUDATime;
    double speedupCopy = (float)CPUTime / CUDATimeCopy;
    std::cout << "Speed measurements:\n";
    std::cout << "GPU - Execution speed: " << throughput << "Mbps\n";
    std::cout << "Total speedup: " << speedup << "\n";
    std::cout << "speedup without counting allocation: " << speedupCopy << "\n";

    
    //// GPU-CPU encryption-decryption validation stage ////
    bool bEqualDecrypt = 1; bool bEqualEncrypt = 1;
    for (int i = 0; i < numMessages; i++)
    {
        bEqualDecrypt &= (resultsDecryption[i] == messages[i]);
        bEqualEncrypt &= (resultsEncryption[i] == encryptions[i]);
    }
    if (!bEqualDecrypt)
    {
        std::cout << "Decryption-message comparison failed!\n";
        return 0;
    }
    if (!bEqualEncrypt)
    {
        std::cout << "CPU-GPU Encryption comparison failed!\n";
        return 0;
    }

    if (bEqualDecrypt && bEqualEncrypt)
    {
        std::cout << "Success!\n";
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
    CHECK_CUDA_ERROR(cudaFree(d_matrices));
    CHECK_CUDA_ERROR(cudaFree(d_SBoxes));
    CHECK_CUDA_ERROR(cudaFree(d_messages));
    CHECK_CUDA_ERROR(cudaFree(d_keys));
    CHECK_CUDA_ERROR(cudaFree(d_resultsEncryption));
    CHECK_CUDA_ERROR(cudaFree(d_resultsDecryption));

    return 0;
}