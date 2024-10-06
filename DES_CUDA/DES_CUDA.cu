#define __CUDACC__
#include <cstdlib>
// External
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DES_CUDA.cuh"

__device__ void generateReverseShiftedKeyCuda(const int& index, uint64_t& roundKey, unsigned char* cLCS);
__device__ void rightCircularShiftCuda(uint32_t& input, uint8_t times);
__device__ void swapLRCuda(uint64_t& input); // Swap left (32 bit) and right (32 bit) parts of the 64 bit input.
__device__ void substituteCuda(uint64_t& input);
__device__ void leftCircularShiftCuda(uint32_t& input, uint8_t times);
__device__ void generateShiftedKeyCuda(const int& index, uint64_t& roundKey, unsigned char* cLCS);
__device__ void permuteMatrixCuda(uint64_t& input, const unsigned char* P, const unsigned int size);


__constant__ unsigned char d_SBoxesConst[8][64] =
{
	{
		14,4,13,1,2,15,11,8,3,10,6,12,5,9,0,7,
		0,15,7,4,14,2,13,1,10,6,12,11,9,5,3,8,
		4,1,14,8,13,6,2,11,15,12,9,7,3,10,5,0,
		15,12,8,2,4,9,1,7,5,11,3,14,10,0,6,13
	},
	{
		15,1,8,14,6,11,3,4,9,7,2,13,12,0,5,10,
		3,13,4,7,15,2,8,14,12,0,1,10,6,9,11,5,
		0,14,7,11,10,4,13,1,5,8,12,6,9,3,2,15,
		13,8,10,1,3,15,4,2,11,6,7,12,0,5,14,9
	},
	{
		10,0,9,14,6,3,15,5,1,13,12,7,11,4,2,8,
		13,7,0,9,3,4,6,10,2,8,5,14,12,11,15,1,
		13,6,4,9,8,15,3,0,11,1,2,12,5,10,14,7,
		1,10,13,0,6,9,8,7,4,15,14,3,11,5,2,12
	},
	{
		7,13,14,3,0,6,9,10,1,2,8,5,11,12,4,15,
		13,8,11,5,6,15,0,3,4,7,2,12,1,10,14,9,
		10,6,9,0,12,11,7,13,15,1,3,14,5,2,8,4,
		3,15,0,6,10,1,13,8,9,4,5,11,12,7,2,14
	},
	{
		2,12,4,1,7,10,11,6,8,5,3,15,13,0,14,9,
		14,11,2,12,4,7,13,1,5,0,15,10,3,9,8,6,
		4,2,1,11,10,13,7,8,15,9,12,5,6,3,0,14,
		11,8,12,7,1,14,2,13,6,15,0,9,10,4,5,3
	},
	{
		12,1,10,15,9,2,6,8,0,13,3,4,14,7,5,11,
		10,15,4,2,7,12,9,5,6,1,13,14,0,11,3,8,
		9,14,15,5,2,8,12,3,7,0,4,10,1,13,11,6,
		4,3,2,12,9,5,15,10,11,14,1,7,6,0,8,13
	},
	{
		4,11,2,14,15,0,8,13,3,12,9,7,5,10,6,1,
		13,0,11,7,4,9,1,10,14,3,5,12,2,15,8,6,
		1,4,11,13,12,3,7,14,10,15,6,8,0,5,9,2,
		6,11,13,8,1,4,10,7,9,5,0,15,14,2,3,12
	},
	{
		13,2,8,4,6,15,11,1,10,9,3,14,5,0,12,7,
		1,15,13,8,10,3,7,4,12,5,6,11,0,14,9,2,
		7,11,4,1,9,12,14,2,0,6,10,13,15,3,5,8,
		2,1,14,7,4,10,8,13,15,12,9,0,3,5,6,11
	}
};
__constant__ unsigned char d_matricesConst[328] =
{
	// IP
	58,50,42,34,26,18,10,2,
	60,52,44,36,28,20,12,4,
	62,54,46,38,30,22,14,6,
	64,56,48,40,32,24,16,8,
	57,49,41,33,25,17, 9,1,
	59,51,43,35,27,19,11,3,
	61,53,45,37,29,21,13,5,
	63,55,47,39,31,23,15,7

	,

	// PC1
	57,49,41,33,25,17,9,
	1,58,50,42,34,26,18,
	10,2,59,51,43,35,27,
	19,11,3,60,52,44,36,
	63,55,47,39,31,23,15,
	7,62,54,46,38,30,22,
	14,6,61,53,45,37,29,
	21,13,5,28,20,12,4

	,

	// PC2
	14,17,11,24,1,5,
	3,28,15,6,21,10,
	23,19,12,4,26,8,
	16,7,27,20,13,2,
	41,52,31,37,47,55,
	30,40,51,45,33,48,
	44,49,39,56,34,53,
	46,42,50,36,29,32

	,

	// Expansion matrix
	32,1, 2, 3, 4, 5,
	4 ,5 ,6 ,7 ,8 ,9,
	8 ,9 ,10,11,12,13,
	12,13,14,15,16,17,
	16,17,18,19,20,21,
	20,21,22,23,24,25,
	24,25,26,27,28,29,
	28,29,30,31,32,1

	,

	// P Matrix
	16,7,20,21,
	29,12,28,17,
	1,15,23,26,
	5,18,31,10,
	2,8,24,14,
	32,27,3,9,
	19,13,30,6,
	22,11,4,25

	,

	// Inverse IP 
	40,8,48,16,56,24,64,32,
	39,7,47,15,55,23,63,31,
	38,6,46,14,54,22,62,30,
	37,5,45,13,53,21,61,29,
	36,4,44,12,52,20,60,28,
	35,3,43,11,51,19,59,27,
	34,2,42,10,50,18,58,26,
	33,1,41,9,49,17,57,25

	,

	// LCS Shifts
	1,
	1,
	2,
	2,
	2,
	2,
	2,
	2,
	1,
	2,
	2,
	2,
	2,
	2,
	2,
	1
};

__global__ void EncryptDESCuda(uint64_t* messages, uint64_t* keys, uint64_t* results)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint64_t result; // setting alias for encryption

	uint64_t input = messages[tid];
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
		generateShiftedKeyCuda(i, shiftedKey, &d_matricesConst[matricesIndices[6]]);
		permutedRoundKey = shiftedKey;
		permuteMatrixCuda(permutedRoundKey, &d_matricesConst[matricesIndices[2]], 48);//roundKeyPermutation(permutedRoundKey);

		// Expansion permutation
		permuteMatrixCuda(input, &d_matricesConst[matricesIndices[3]], 48);//expandPermutation(input); // 48 bits

		// XOR with permuted round key
		input ^= permutedRoundKey;

		// Substitution S-boxes
		substituteCuda(input); // 32 bits

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
__global__ void DecryptDESCuda(uint64_t* encryptions, uint64_t* keys, uint64_t* results)
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
		substituteCuda(input); // 32 bits

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
	__syncthreads();
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
__device__ void generateShiftedKeyCuda(const int& index, uint64_t& roundKey, unsigned char* cLCS)
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
__device__ void generateReverseShiftedKeyCuda(const int& index, uint64_t& roundKey, unsigned char* cLCS)
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

__device__ void substituteCuda(uint64_t& input)
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
		temp = d_SBoxesConst[i][(y * 16) + x];
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

