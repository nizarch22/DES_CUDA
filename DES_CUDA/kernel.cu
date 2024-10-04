
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

    for (int testCount = 2; testCount < NUM_TESTS_QUICK; testCount++)
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

        const uint64_t numTesting = numMessages[testCount];
        //Debug - delete later
        const uint64_t byteTestingDebug = numTesting * 150 * 64;
        const uint64_t byteTestingDebugInt = numTesting * 150 * 8;
        unsigned char* debug;
        uint64_t* debugInt;
        debug = (unsigned char*)malloc(byteTestingDebug);
        debugInt = (uint64_t*)malloc(byteTestingDebugInt);
        for (int i = 0; i < numTesting; i++)
        {
            debugInt[149+150*i] = DEBUG_ITERATION;
        }
        unsigned char* d_debug;
        uint64_t* d_debugInt;
        cudaMalloc(&d_debug, byteTestingDebug);
        cudaMalloc(&d_debugInt, byteTestingDebugInt);
        cudaMemcpy(d_debugInt, debugInt, byteTestingDebugInt, cudaMemcpyHostToDevice);

        debugFoo << < numTesting, 64 >> > (d_messages, d_keys, d_resultsEncryption, d_debug, d_debugInt);
        cudaDeviceSynchronize(); // wait for encrypt to finish
        cudaMemcpy(&debug[0], d_debug, byteTestingDebug, cudaMemcpyDeviceToHost);
        cudaMemcpy(&debugInt[0], d_debugInt, byteTestingDebugInt, cudaMemcpyDeviceToHost);
        cudaMemcpy(resultsEncryption, d_resultsEncryption, bytesMessages[testCount], cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize(); // wait for memory to arrive

        //for (int i = 0; i < numTesting; i++)
        //{
        //    if (debugInt[i] != messages[i])
        //    {
        //        std::cout << "It did not work.\n";
        //    }
        //}
        //return 0;
        uint64_t temp = 0;
        uint64_t debugCPU[150] = { 0 };
        uint64_t debugGPU[150] = { 0 };
        //for (int k = 0; k < numTesting; k++)
        //{
        //    for (int i = 0; i <= 100; i++)
        //    {
        //        uint64_t word = 0;
        //        for (int j = 0; j < 64; j++)
        //        {
        //            word <<= 1;
        //            word += debug[i * 64 + 63 - j + k * 150*64];
        //        }
        //        debugGPU[i] = word;
        //    }
        //    debugCPU[149] = DEBUG_ITERATION;
        //    EncryptDESDebug(messages[k], keys[k], temp, debugCPU);
        //    int size = 64;
        //    // first 3 checks
        //    for (int i = 0; i < 2; i++)
        //    {
        //        if (debugCPU[i] != debugGPU[i])
        //        {
        //            std::cout << "Initial mismatch occured. Block: " << k << ". Round: " << i % 6 << "\n";
        //            std::cout << debugCPU[i] << " != " << debugGPU[i] << "\n";

        //            for (int j = 0; j < 64; j++)
        //            {
        //                std::cout << (int)(debug[i * 64 + j + k * 150]) << ",";
        //            }
        //            std::cout << "\n";
        //            uint64_t mismatchedWord = debugCPU[i];
        //            for (int j = 0; j < 64; j++)
        //            {
        //                std::cout << (mismatchedWord & 1) << ",";
        //                mismatchedWord >>= 1;
        //            }
        //            std::cout << "\n";

        //            std::cout << "Original message:\n";
        //            mismatchedWord = messages[k];
        //            for (int j = 0; j < 64; j++)
        //            {
        //                std::cout << (mismatchedWord & 1) << ",";
        //                mismatchedWord >>= 1;
        //            }
        //            std::cout << "\n";

        //            std::cout << "CPU original message:\n";
        //            mismatchedWord = debugCPU[140];
        //            for (int j = 0; j < 64; j++)
        //            {
        //                std::cout << (mismatchedWord & 1) << ",";
        //                mismatchedWord >>= 1;
        //            }
        //            std::cout << "\n";

        //            std::cout << "GPU original message:\n";
        //            mismatchedWord = debugInt[0];
        //            for (int j = 0; j < 64; j++)
        //            {
        //                std::cout << (mismatchedWord & 1) << ",";
        //                mismatchedWord >>= 1;
        //            }
        //            std::cout << "\n";

        //            std::cout << "Original key:\n";
        //            mismatchedWord = keys[k];
        //            for (int j = 0; j < 64; j++)
        //            {
        //                std::cout << (mismatchedWord & 1) << ",";
        //                mismatchedWord >>= 1;
        //            }
        //            std::cout << "\n";

        //            std::cout << "CPU original key:\n";
        //            mismatchedWord = debugCPU[141];
        //            for (int j = 0; j < 64; j++)
        //            {
        //                std::cout << (mismatchedWord & 1) << ",";
        //                mismatchedWord >>= 1;
        //            }
        //            std::cout << "\n";

        //            std::cout << "GPU original key:\n";
        //            mismatchedWord = debugInt[1];
        //            for (int j = 0; j < 64; j++)
        //            {
        //                std::cout << (mismatchedWord & 1) << ",";
        //                mismatchedWord >>= 1;
        //            }
        //            std::cout << "\n";
        //            return -1;
        //        }
        //    }
        //    for (int i = 3; i <= 98; i++)
        //    {
        //        if (debugCPU[i] != debugGPU[i])
        //        {
        //            std::cout << "Mismatch occured. Block: " << k << ". Iteration: " << (i - 3) / 6 << ", Index: " << 3 + (i - 3) % 6 << "\n";
        //            std::cout << debugCPU[i] << " != " << debugGPU[i] << "\n";
        //            for (int j = 0; j < 64; j++)
        //            {
        //                std::cout << (int)(debug[i * 64 + j + k * 150]) << ",";
        //            }
        //            std::cout << "\n";
        //            uint64_t mismatchedWord = debugCPU[i];
        //            for (int j = 0; j < 64; j++)
        //            {
        //                std::cout << (mismatchedWord & 1) << ",";
        //                mismatchedWord >>= 1;
        //            }
        //            std::cout << "\n";

        //            std::cout << "results from sboxout\n";
        //            for (int i = 0; i < 8; i++)
        //            {
        //                std::cout << debugInt[i + 150 * k] << ",";
        //            }
        //            std::cout << "\n";
        //            for (int i = 0; i < 8; i++)
        //            {
        //                std::cout << debugCPU[101 + i] << ",";
        //            }
        //            std::cout << "\n";

        //            std::cout << "subcuda - x\n";
        //            for (int i = 0; i < 8; i++)
        //            {
        //                std::cout << debugInt[i + 8 + 150 * k] << ",";
        //            }
        //            std::cout << "\n";
        //            for (int i = 0; i < 8; i++)
        //            {
        //                std::cout << debugCPU[101 + i + 8] << ",";
        //            }
        //            std::cout << "\n";

        //            std::cout << "subcuda - y\n";
        //            for (int i = 0; i < 8; i++)
        //            {
        //                std::cout << debugInt[i + 16 + 150 * k] << ",";
        //            }
        //            std::cout << "\n";
        //            for (int i = 0; i < 8; i++)
        //            {
        //                std::cout << debugCPU[101 + i + 16] << ",";
        //            }
        //            std::cout << "\n";
        //            return -1;
        //        }
        //    }
        //    for (int i = 99; i <= 100; i++)
        //    {
        //        if (debugCPU[i] != debugGPU[i])
        //        {
        //            std::cout << "Ending mismatch occured. Index: " << i << "\n";
        //            std::cout << debugCPU[i] << " != " << debugGPU[i] << "\n";

        //            for (int j = 0; j < 64; j++)
        //            {
        //                std::cout << (int)(debug[i * 64 + j + k * 150]) << ",";
        //            }
        //            std::cout << "\n";
        //            uint64_t mismatchedWord = debugCPU[i];
        //            for (int j = 0; j < 64; j++)
        //            {
        //                std::cout << (mismatchedWord & 1) << ",";
        //                mismatchedWord >>= 1;
        //            }
        //            return -1;
        //        }
        //    }
        //}

        uint64_t errorCounter = 0;
        for (int i = 0; i < numTesting; i++)
        {
            //EncryptDESDebug(messages[i], keys[i], temp, debugCPU);
            EncryptDES(messages[i], keys[i], temp);
            if (temp != resultsEncryption[i])
            {
                std::cout << "Ending mismatch occured. Index: " << i << "\n";
                std::cout << temp << " != " << resultsEncryption[i] << "\n";
                errorCounter++;
                //cudaMemcpy(d_messages, &messages[i], 64, cudaMemcpyHostToDevice);
                //cudaMemcpy(d_keys, &keys[i], 64, cudaMemcpyHostToDevice);

                //debugFoo << < 1, 64 >> > (d_messages, d_keys, d_resultsEncryption, d_debug, d_debugInt);
                //cudaDeviceSynchronize(); // wait for encrypt to finish
                //cudaMemcpy(resultsEncryption, d_resultsEncryption, 64, cudaMemcpyDeviceToHost);
                //cudaDeviceSynchronize(); // wait for encrypt to finish

                // Check 'debug' rather than 'results' from GPU
                uint64_t word = 0;
                for (int j = 0; j < 64; j++)
                {
                    word <<= 1;
                    word += debug[100 * 64 + 63 - j + i * 150];
                }
                std::cout << temp << " =?= " << word << "\n";
                std::cout << "Result: " << ((temp == word) ? "Success" : "Fail") << "\n";

                // Check using debug function version on the CPU
                EncryptDESDebug(messages[i], keys[i], word, debugCPU);
                if (word != temp)
                {
                    std::cout << "EncryptDES is corrupt!\n";
                    return -1;
                }
                //std::cout << "Singular attempt - matching check: " << i << "\n";
                //std::cout << temp << " =?= " << resultsEncryption[0] << "\n";
                //std::cout << "Result: " << ((temp == resultsEncryption[0]) ? "Success" : "Fail") << "\n";
                //return -1;
            }
        }
        if (errorCounter != 0)
        {
            std::cout << "Total errors encountered : " << errorCounter << "\n";
            return -1;
        }
        std::cout << "Success! All debugging parameters match!\n";

        return 0;
        //// Run Encryption & Decryption in CUDA stage ////
        // We encrypt the messages using EncryptDESCuda. Then, we use all those encrypted messages to run DecryptDESCuda.
        startTimeExecute[testCount] = clock();
        EncryptDESCuda << < numMessages[testCount], numThreads >> > (d_messages, d_keys, d_resultsEncryption);
        cudaDeviceSynchronize(); // wait for encrypt to finish
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