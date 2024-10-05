
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

#define NUM_TESTS 9
#define NUM_TESTS_QUICK 9
int main()
{
    // kernel parameters
    const int numThreads = 256;
    const int numMessages[NUM_TESTS] = { 131072,262144,524288,1048576,2097152,4194304,8388608,16777216,33554432 };// 524288 -  4MB - 10x speedup. 33554432 - 256MB - 70x speedup!
    const int numBlocks[NUM_TESTS] = { 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072 };

    // size parameters
    const int bytesMessages[NUM_TESTS] = { 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864 , 134217728, 268435456 };
    const int bytesKeys[NUM_TESTS] = { 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864 , 134217728, 268435456 }; // change for keys

    const int bytesLargest = 268435456;
    //// Kernel arguments prep stage ////
    // prep matrices, sboxes
    const unsigned char* matrices[7] = {IP,PC1,PC2, E, PMatrix,IPInverse, LCS};
    const int matricesSizes[7] = { 64,56,48,48,32,64,16 };
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

    int startTimeCopyMatrices = clock(); // Used to measure the time GPU finishes execution since copying started.
    // cuda copy memory - matrices, sboxes
    cudaMemcpyToSymbol(d_SBoxesConst, &SBoxes[0][0], 512, 0, cudaMemcpyHostToDevice);;
    int offset = 0;
    for (int i = 0; i < 7; i++)
    {
        cudaMemcpyToSymbol(d_matricesConst, &matrices[i][0], matricesSizes[i], offset, cudaMemcpyHostToDevice);;
        //cudaMemcpy(d_matrices + offset, &matrices[i][0], matricesSizes[i], cudaMemcpyHostToDevice);
        offset += matricesSizes[i];
    }
    int endTimeCopyMatrices = clock(); // Used to measure the time GPU finishes execution since copying started.


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
        // We encrypt the messages using EncryptDESCuda. Then, we use all those encrypted messages to run DecryptDESCuda.
        startTimeExecute[testCount] = clock();
        EncryptDESCuda << < numBlocks[testCount], numThreads>> > (d_messages, d_keys, d_resultsEncryption);
        cudaDeviceSynchronize(); // wait for encrypt to finish
        DecryptDESCuda << <numBlocks[testCount], numThreads >> > (d_resultsEncryption, d_keys, d_resultsDecryption);
        cudaDeviceSynchronize();
        endTimeExecute[testCount] = clock();
        // cuda copy results 
        cudaMemcpy(resultsEncryption, d_resultsEncryption, bytesMessages[testCount], cudaMemcpyDeviceToHost);
        cudaMemcpy(resultsDecryption, d_resultsDecryption, bytesMessages[testCount], cudaMemcpyDeviceToHost);
        endTimeRetrieveResults[testCount] = clock();

        // cuda check for errors in CUDA execution
        CHECK_CUDA_ERROR(cudaGetLastError());

        //startTimeCPU[testCount] = clock();
        //for (int i = 0; i < numMessages[testCount]; i++)
        //{
        //    EncryptDES(messages[i], keys[i], encryptions[i]);
        //    DecryptDES(encryptions[i], keys[i], decryptions[i]);
        //}
        //endTimeCPU[testCount] = clock();

        ////// GPU-CPU encryption-decryption validation stage ////
        //for (int i = 0; i < numMessages[testCount]; i++)
        //{
        //    bEqualDecrypt &= (resultsDecryption[i] == messages[i]);
        //    bEqualEncrypt &= (resultsEncryption[i] == encryptions[i]);
        //}
    }

    //if (!bEqualEncrypt)
    //{
    //    std::cout << "CPU-GPU Encryption comparison failed!\n";
    //    return 0;
    //}
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
    int CUDATimeCopyMatrices = endTimeCopyMatrices - startTimeCopyMatrices;
    
    double throughput, speedup, speedupExecute;

    std::cout << "CUDA Debug results:\n";
    for (int i = 0; i < NUM_TESTS_QUICK; i++)
    {
        CUDATimeExecute = endTimeExecute[i] - startTimeExecute[i];
        CUDATimeCopy = CUDATimeExecute + endTimeInputCopy[i] - startTimeInputCopy[i] + CUDATimeCopyMatrices;
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