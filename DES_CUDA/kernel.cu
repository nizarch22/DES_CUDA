
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
#define NUM_TESTS_QUICK 6
int main()
{
    // kernel parameters
    const int numThreads = 64;
    const int numMessages[NUM_TESTS] = { 128,256, 512, 1024, 2048, 4096, 8192,16384,33554432 };// 524288 -  4MB - 10x speedup. 33554432 - 256MB - 70x speedup!
    // size parameters
    const int bytesMessages[NUM_TESTS] = { 1024, 2048, 4096, 8192, 16384, 32768, 65536 , 131072, 268435456 };
    const int bytesKeys[NUM_TESTS] = { 1024, 2048, 4096, 8192, 16384, 32768, 65536 , 131072, 268435456 }; // change for keys

    const int bytesLargest = 268435456;

    //// Kernel arguments prep stage ////
    // prep matrices, sboxes
    unsigned char* d_matricesConst;
    unsigned char* d_SBoxesConst;

    cudaMalloc(&d_matricesConst, bytesLargest);
    cudaMalloc(&d_SBoxesConst, bytesLargest);

    const unsigned char* matrices[7] = { IP,PC1,PC2,E,PMatrix,IPInverse,LCS };
    const int matricesSizes[7] = { 64,56,48,48,32,64,16 };

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
    int startTimeInputCopy[NUM_TESTS] = {0}; int endTimeInputCopy[NUM_TESTS] = { 0 };
    int startTimeExecute[NUM_TESTS] = { 0 }; int endTimeExecute[NUM_TESTS] = { 0 };
    int endTimeRetrieveResults[NUM_TESTS] = { 0 };
    //int startTimeCPU[NUM_TESTS]; int endTimeCPU[NUM_TESTS];
    
    // Generate messages - 256MB
    //for (int i = 0; i < 33554432; i++)
    //{
    //    messages[i] = (((uint64_t)rand()) << 32) | rand();
    //    keys[i] = (((uint64_t)rand()) << 32) | rand();
    //}

    // cuda copy memory - matrices, sboxes



    int startTimeCopyMatrices = clock(); // Used to measure the time GPU finishes execution since copying started.
    // cuda copy memory - matrices, sboxes
    int offset = 0;
    cudaMemcpy(d_SBoxesConst, SBoxes[0], 512, cudaMemcpyHostToDevice);
    for (int i = 0; i < 7; i++)
    {
        //cudaMemcpyToSymbol(d_matricesConst, &matrices[i][0], matricesSizes[i], offset, cudaMemcpyHostToDevice);
        cudaMemcpy(d_matricesConst + offset, &matrices[i][0], matricesSizes[i], cudaMemcpyHostToDevice);
        offset += matricesSizes[i];
    }
    int endTimeCopyMatrices = clock(); // Used to measure the time GPU finishes execution since copying started.
    CHECK_CUDA_ERROR(cudaGetLastError());

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
        EncryptDESCuda << < numMessages[testCount], numThreads >> > (d_messages, d_keys, d_resultsEncryption, d_matricesConst, d_SBoxesConst);
        cudaDeviceSynchronize(); // wait for encrypt to finish

        // Decrypt all the encryption made by EncryptDesCuda above using DecryptDESCuda.
        DecryptDESCuda << <numMessages[testCount], numThreads >> > (d_resultsEncryption, d_keys, d_resultsDecryption, d_matricesConst, d_SBoxesConst);
        cudaDeviceSynchronize();
        endTimeExecute[testCount] = clock();

        // cuda copy results 
        cudaMemcpy(resultsEncryption, d_resultsEncryption, bytesMessages[testCount], cudaMemcpyDeviceToHost);
        cudaMemcpy(resultsDecryption, d_resultsDecryption, bytesMessages[testCount], cudaMemcpyDeviceToHost);
        endTimeRetrieveResults[testCount] = clock();

        CHECK_CUDA_ERROR(cudaGetLastError());
        // cuda check for errors in CUDA execution

        //for (int i = 0; i < numMessages[testCount]; i++)
        //{
        //    bEqualDecrypt &= (resultsDecryption[i] == messages[i]);
        //}
    }

    // Quick validation
    for (int i = 0; i < numMessages[0]; i++)
    {
        uint64_t temp;
        EncryptDES(messages[i], keys[i], temp);
        if (temp != resultsEncryption[i])
        {
            std::cout << "Occured at " << i << "\n";
            for (int i = 0; i < 64; i++)
            {
                std::cout << ((messages[i] >> i) & 1) << ",";
            }
            std::cout << "\n";
            for (int i = 0; i < 64; i++)
            {
                std::cout << ((resultsDecryption[i] >> i) & 1) << ",";
            }
            std::cout << "\n";
            return -1;
        }
        bEqualDecrypt &= (resultsDecryption[i] == messages[i]);
        if (!bEqualDecrypt)
        {
            for (int i = 0; i < 64; i++)
            {
                std::cout << ((messages[i] >> i) & 1) << ",";
            }
            std::cout << "\n";
            for (int i = 0; i < 64; i++)
            {
                std::cout << ((resultsDecryption[i] >> i) & 1) << ",";
            }
            std::cout << "\n";
            return -1;
        }
    }
    // Quick validation
    for (int i = 0; i < numMessages[0]; i++)
    {
        bEqualDecrypt &= (resultsDecryption[i] == messages[i]);
        if (!bEqualDecrypt)
        {
            for (int i = 0; i < 64; i++)
            {
                std::cout << ((messages[i] >> i) & 1) << ",";
            }
            std::cout << "\n";
            for (int i = 0; i < 64; i++)
            {
                std::cout << ((resultsDecryption[i] >> i) & 1) << ",";
            }
            std::cout << "\n";
            return -1;
        }
    }

    if (!bEqualDecrypt)
    {
        std::cout << "GPU Decryption comparison failed!\n";
        return -1;
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
EVALUATE_PERFORMANCE:
    int CUDATime, CUDATimeCopy, CUDATimeExecute;
    int CUDATimeAlloc = endTimeAlloc - startTimeAlloc;
    
    double throughput, speedup, speedupExecute;

    //CUDATimeExecute = endTimeExecute[0] - startTimeExecute[0];
    //CUDATimeCopy = CUDATimeExecute + endTimeInputCopy[0] - startTimeInputCopy[0];
    //CUDATime = CUDATimeCopy + CUDATimeAlloc;
    ////CPUTime = endTimeCPU[i] - startTimeCPU[i];
    //// printout of timing results
    //std::cout << "Total messages size: " << (bytesLargest >> 20) << " MegaBytes\n";
    //std::cout << "Total timing measurements:\n";
    //std::cout << "GPU - only kernel execution: " << CUDATimeExecute << "ms\n";
    //std::cout << "GPU - with copying: " << CUDATimeCopy << "ms\n";
    //std::cout << "GPU - with allocation and copying: " << CUDATime << "ms\n";
    ////std::cout << "CPU: " << CPUTime << "ms\n";

    //throughput = 1000.0f * (float)(bytesLargest >> 17) / CUDATimeExecute;
    ////speedup = (float)CPUTime / CUDATime;
    ////speedupExecute = (float)CPUTime / CUDATimeExecute;
    //std::cout << "Speed measurements:\n";
    //std::cout << "GPU - Execution speed: " << throughput << " MegaBits per second\n";

    //std::cout << "Speedup measurements:\n";
    //std::cout << "Speedup - Execution: " << speedupExecute << "\n";
    //std::cout << "Speedup - with allocation and copy: " << speedup << "\n";


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
    //int a;
    //cudaOccupancyMaxActiveBlocksPerMultiprocessor(&a, EncryptDESCuda, 32, 0);
    //std::cout << "Number of resident blocks: " << a << "\n";
    return 0;
}