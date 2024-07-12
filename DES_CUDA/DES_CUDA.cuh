#pragma once
#include <cstdlib>
// External
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ void generateReverseShiftedKeyCuda(const int& index, uint64_t& roundKey, unsigned char* cLCS);
__device__ void rightCircularShiftCuda(uint32_t& input, uint8_t times);
__device__ void fullShiftLCSCuda(uint64_t& roundKey);
__device__ void swapLRCuda(uint64_t& input); // Swap left (32 bit) and right (32 bit) parts of the 64 bit input.
__device__ void substituteCuda(uint64_t& input, unsigned char* sboxes);
__device__ void leftCircularShiftCuda(uint32_t& input, uint8_t times);
__device__ void generateShiftedKeyCuda(const int& index, uint64_t& roundKey, unsigned char* cLCS);
__device__ void permuteMatrixCuda(uint64_t& input, const unsigned char* P, const unsigned int size);

__global__ void EncryptDESCuda(uint64_t* messages, uint64_t* keys, unsigned char* matrices, unsigned char* sboxes, uint64_t* results);
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
