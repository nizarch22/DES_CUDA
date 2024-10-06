#pragma once
#include <cstdlib>
// External
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void EncryptDESCuda(uint64_t* messages, uint64_t* keys, uint64_t* results);
__global__ void DecryptDESCuda(uint64_t* encryptions, uint64_t* keys, uint64_t* results);

// Debug functions
//__global__ void EncryptDESCudaDebug(uint64_t* messages, uint64_t* keys, unsigned char* matrices, unsigned char* sboxes, uint64_t* results, uint64_t* debug, int n);

// Constant arrays
__constant__ unsigned char d_SBoxesConst[512];
__constant__ unsigned char d_matricesConst[328];



// Debug - delete later
__global__ void debugFoo(uint64_t* messages, uint64_t* keys, uint64_t* results, unsigned char* debug, uint64_t* debugInt);
__global__ void debugFooDecrypt(uint64_t* messages, uint64_t* keys, uint64_t* results);
