#define __CUDACC__
#include <cstdlib>
// External
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DES_CUDA.cuh"

__device__ void generateReverseShiftedKeyCuda(const int& index, uint64_t& roundKey, const unsigned char* cLCS);
__device__ void rightCircularShiftCuda(uint32_t& input, uint8_t times);
__device__ void swapLRCuda(uint64_t& input); // Swap left (32 bit) and right (32 bit) parts of the 64 bit input.
__device__ void substituteCuda(uint64_t& input, const unsigned char* d_SBoxesConst);
__device__ void leftCircularShiftCuda(uint32_t& input, uint8_t times);
__device__ void generateShiftedKeyCuda(const int& index, uint64_t& roundKey, const unsigned char* cLCS);
__device__ void permuteMatrixCuda(uint64_t& input, const unsigned char* P, const unsigned int size);

__launch_bounds__(128,3)
__global__ void EncryptDESCuda(uint64_t* messages, uint64_t* keys, uint64_t* results, const unsigned char* d_matricesConst, const unsigned char* d_SBoxesConst)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint64_t result; // setting alias for encryption

	uint64_t input = messages[tid];
	uint64_t shiftedKey = keys[tid];
	uint64_t permutedRoundKey;
	uint64_t left; // last 32 bits of plaintext/input to algorithm are preserved in this variable 

	// loading matrices variable
	//int matricesIndices[7] = { 0,64,120,168,216,248,312 };


	// Initial operations 
	permuteMatrixCuda(input, &d_matricesConst[0], 64); //initialPermutation(input);
	permuteMatrixCuda(shiftedKey, &d_matricesConst[64], 56); // PC1 of key

	for (int i = 0; i < 16; i++)
	{
		// Preserving L,R.
		// preserve right side (Result[63:32] = Input[31:0])
		result = input;
		result <<= 32;
		// preserve left side
		left = input >> 32;

		// Round key
		generateShiftedKeyCuda(i, shiftedKey, &d_matricesConst[312]);
		permutedRoundKey = shiftedKey;
		permuteMatrixCuda(permutedRoundKey, &d_matricesConst[120], 48);//roundKeyPermutation(permutedRoundKey);

		// Expansion permutation
		permuteMatrixCuda(input, &d_matricesConst[168], 48);//expandPermutation(input); // 48 bits

		// XOR with permuted round key
		input ^= permutedRoundKey;

		// Substitution S-boxes
		substituteCuda(input, d_SBoxesConst); // 32 bits

		// "P-matrix" permutation i.e. mix/shuffle
		permuteMatrixCuda(input, &d_matricesConst[216], 32);// mixPermutation(input);

		// XOR with preserved left side
		result += left ^ input; // Result[31:0] = L XOR f[31:0];

		// End of loop
		input = result;
	}

	swapLRCuda(result);
	permuteMatrixCuda(result, &d_matricesConst[248], 64);//reverseInitialPermutation(result);
	results[tid] = result;
}
__global__ void DecryptDESCuda(uint64_t* encryptions, uint64_t* keys, uint64_t* results, const unsigned char* d_matricesConst, const unsigned char* d_SBoxesConst)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint64_t result; // setting alias for decryption

	uint64_t input = encryptions[tid];
	uint64_t shiftedKey = keys[tid];
	uint64_t permutedRoundKey;
	uint64_t left; // last 32 bits of plaintext/input to algorithm are preserved in this variable 

	// loading matrices variable
	int matricesIndices[7] = { 0,64,120,168,216,248,312 };

	// Initial operations 
	permuteMatrixCuda(input, &d_matricesConst[matricesIndices[0]], 64); //initialPermutation(input);
	permuteMatrixCuda(shiftedKey, &d_matricesConst[matricesIndices[1]], 56); // PC1 of key

	for (int i = 0; i < 16; i++)
	{
		// Preserving L,R.
		// preserve right side (Result[63:32] = Input[31:0])
		result = input;
		result <<= 32;
		// preserve left side
		left = input >> 32;

		// Round key
		permutedRoundKey = shiftedKey;
		permuteMatrixCuda(permutedRoundKey, &d_matricesConst[matricesIndices[2]], 48);//roundKeyPermutation(permutedRoundKey);
		generateReverseShiftedKeyCuda(i, shiftedKey, &d_matricesConst[matricesIndices[6]]);

		// Expansion permutation
		permuteMatrixCuda(input, &d_matricesConst[matricesIndices[3]], 48);//expandPermutation(input); // 48 bits

		// XOR with permuted round key
		input ^= permutedRoundKey;

		// Substitution S-boxes
		substituteCuda(input, d_SBoxesConst); // 32 bits

		// "P-matrix" permutation i.e. mix/shuffle
		permuteMatrixCuda(input, &d_matricesConst[matricesIndices[4]], 32);// mixPermutation(input);

		// XOR with preserved left side
		result += left ^ input; // Result[31:0] = L XOR f[31:0];

		// End of loop
		input = result;
	}

	swapLRCuda(result);
	permuteMatrixCuda(result, &d_matricesConst[matricesIndices[5]], 64);//reverseInitialPermutation(result);
	results[tid] = result;
}


__device__ void permuteMatrixCuda(uint64_t& input, const unsigned char* P, const unsigned int size)
{
	uint64_t output = 0;
	uint64_t bit;

	for (int i = 0; i < size; i++)
	{
		bit = (input >> (P[i] - 1)) & 1;
		output += bit << i;
	}
	input = output;
}
__device__ void generateShiftedKeyCuda(const int& index, uint64_t& roundKey, const unsigned char* cLCS)
{
	uint32_t left, right;
	uint64_t mask28Bits = 268435455; // covers first 28 bits

	// getting left and right sides
	right = roundKey & mask28Bits;
	mask28Bits <<= 28;
	mask28Bits = roundKey & mask28Bits;
	left = mask28Bits >> 28;

	// circular shifts
	leftCircularShiftCuda(left, cLCS[index]);
	leftCircularShiftCuda(right, cLCS[index]);

	// copying left and right shifted keys to roundKey.
	roundKey = left;
	roundKey <<= 28;
	roundKey += right;
}
__device__ void generateReverseShiftedKeyCuda(const int& index, uint64_t& roundKey, const unsigned char* cLCS)
{
	uint32_t left, right;
	uint64_t mask28Bits = 268435455; // covers first 28 bits

	// getting left and right sides
	right = roundKey & mask28Bits;
	mask28Bits <<= 28;
	mask28Bits = roundKey & mask28Bits;
	left = mask28Bits >> 28;

	// circular shifts
	rightCircularShiftCuda(left, cLCS[15 - index]);
	rightCircularShiftCuda(right, cLCS[15 - index]);

	// copying left and right shifted keys to roundKey.
	roundKey = left;
	roundKey <<= 28;
	roundKey += right;
}
__device__ void leftCircularShiftCuda(uint32_t& input, uint8_t times)
{
	uint32_t mask28thBit = 1 << 27; // 28th bit
	uint32_t mask28Bits = 268435455; // covers first 28 bits

	uint8_t bit;
	for (int i = 0; i < times; i++)
	{
		bit = (input & mask28thBit) >> 27;
		input <<= 1;
		input += bit;
	}
	input = input & mask28Bits;
}

__device__ void rightCircularShiftCuda(uint32_t& input, uint8_t times)
{
	uint32_t bit28th = 1 << 27; // 28th bit
	uint32_t mask1stBit = 1; // 28th bit
	uint32_t mask28Bits = 268435455; // covers first 28 bits

	uint32_t bit;
	for (int i = 0; i < times; i++)
	{
		bit = (input & mask1stBit);
		input >>= 1;
		input += bit * bit28th;
	}
	input = input & mask28Bits;
}

__device__ void substituteCuda(uint64_t& input, const unsigned char* d_SBoxesConst)
{
	uint64_t result = 0; uint64_t temp;
	uint16_t y, x;
	uint16_t in;

	uint64_t mask = 63;
	uint8_t maskY1, maskY2, maskX;
	maskY1 = 1;
	maskY2 = 32;
	maskX = 30;
	for (int i = 0; i < 8; i++)
	{
		// getting x,y coordinates for Sbox
		in = input & mask;
		x = (in & maskX) >> 1;
		y = (in & maskY2) >> 4;
		y += in & maskY1;

		// Substitution 
		temp = d_SBoxesConst[i*64 + (y * 16) + x];
		result += temp << (4 * i);

		// next bits
		input >>= 6;
	}
	input = result;
}
__device__ void swapLRCuda(uint64_t& input) // Swap left (32 bit) and right (32 bit) parts of the 64 bit input.
{
	uint64_t temp = input;
	// containing left side 
	temp >>= 32;

	// right side moved to left
	input <<= 32;

	// left side moved to right
	input += temp;
}

