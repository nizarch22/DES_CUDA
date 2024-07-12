#pragma once
#include <cstdlib>
// External
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__device__ void swapLR(uint64_t& input); // Swap left (32 bit) and right (32 bit) parts of the 64 bit input.
__device__ void substitute(uint64_t& input);
__device__ void leftCircularShift(uint32_t& input, uint8_t times);
__device__ void generateShiftedKey(const int& index, uint64_t& roundKey);
__device__ void permuteMatrix(uint64_t& input, const unsigned char* P, const unsigned int size);

__global__ void EncryptDESCuda(uint64_t* messages, uint64_t* keys, uint64_t* matrices, uint64_t* results);
//void DecryptDES(const uint64_t& encryption, const uint64_t& key, uint64_t& decryption);
//void InitKeyDES(uint64_t& key);
//
//// debug functions
//void printMatrix(uint64_t matrix, int y, int x);
//void foo();
//
////- including will not be necessary after debugging is done.
//// matrix helper functions 
//void permuteMatrix(uint64_t& input, const unsigned char* P, const unsigned int size);
