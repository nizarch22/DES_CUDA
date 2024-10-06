#pragma once
#include <cstdlib>
// External
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void EncryptDESCuda(uint64_t* messages, uint64_t* keys, uint64_t* results, unsigned char* d_matricesConst);
__global__ void DecryptDESCuda(uint64_t* encryptions, uint64_t* keys, uint64_t* results, unsigned char* d_matricesConst, unsigned char* d_SBoxesConst);

// Debug functions
//__global__ void EncryptDESCudaDebug(uint64_t* messages, uint64_t* keys, unsigned char* matrices, unsigned char* sboxes, uint64_t* results, uint64_t* debug, int n);

// Constant arrays

