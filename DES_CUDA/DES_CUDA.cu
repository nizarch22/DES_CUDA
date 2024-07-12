#include <cstdlib>
// External
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DES_CUDA.cuh"


// Definitions
#define NUM_THREADS 128
#define NUM_TESTS 300

__global__ void EncryptDESCuda(uint64_t* messages, uint64_t* keys, unsigned char* matrices, unsigned char* sboxes, uint64_t* results)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint64_t result; // setting alias for encryption

	uint64_t input = messages[tid];
	uint64_t shiftedKey = keys[tid];
	uint64_t permutedRoundKey;
	uint64_t left; // last 32 bits of plaintext/input to algorithm are preserved in this variable 

	// load matrices
	// essential variables
	unsigned char* cIP, * cPC1, * cPC2, * cE, * cPMatrix, * cIPInverse, * cLCS;
	int matricesSizes[7] = { 64,56,48,48,32,64,16 };

	// loading matrices process
	unsigned char* temp = matrices;
	cIP = temp; temp += matricesSizes[0];
	cPC1 = temp; temp += matricesSizes[1];
	cPC2 = temp; temp += matricesSizes[2];
	cE = temp; temp += matricesSizes[3];
	cPMatrix = temp; temp += matricesSizes[4];
	cIPInverse = temp; temp += matricesSizes[5];
	cLCS = temp;

	// Initial operations 
	permuteMatrixCuda(input, cIP, 64); //initialPermutation(input);
	permuteMatrixCuda(shiftedKey, cPC1, 56); // PC1 of key
	for (int i = 0; i < 16; i++)
	{
		// Preserving L,R.
		// preserve right side (Result[63:32] = Input[31:0])
		result = input;
		result <<= 32;
		// preserve left side
		left = input >> 32;

		// Round key
		generateShiftedKeyCuda(i, shiftedKey, cLCS);
		permutedRoundKey = shiftedKey;
		permuteMatrixCuda(permutedRoundKey, cPC2, 48);//roundKeyPermutation(permutedRoundKey);

		// Expansion permutation
		permuteMatrixCuda(input, cE, 48);//expandPermutation(input); // 48 bits

		// XOR with permuted round key
		input ^= permutedRoundKey;

		// Substitution S-boxes
		substituteCuda(input, sboxes); // 32 bits

		// "P-matrix" permutation i.e. mix/shuffle
		permuteMatrixCuda(input, cPMatrix, 32);// mixPermutation(input);

		// XOR with preserved left side
		result += left ^ input; // Result[31:0] = L XOR f[31:0];

		// End of loop
		input = result;
	}

	swapLRCuda(result);
	permuteMatrixCuda(result, cIPInverse, 64);//reverseInitialPermutation(result);
	results[tid] = result;
	// debug final point
}
__global__ void DecryptDESCuda(uint64_t* messages, uint64_t* keys, unsigned char* matrices, unsigned char* sboxes, uint64_t* results)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint64_t result; // setting alias for encryption

	uint64_t input = messages[tid];
	uint64_t shiftedKey = keys[tid];
	uint64_t permutedRoundKey;
	uint64_t left; // last 32 bits of plaintext/input to algorithm are preserved in this variable 

	// load matrices
	// essential variables
	unsigned char* cIP, * cPC1, * cPC2, * cE, * cPMatrix, * cIPInverse, * cLCS;
	int matricesSizes[7] = { 64,56,48,48,32,64,16 };

	// loading matrices process
	int offset = 0;
	unsigned char* temp = matrices;
	cIP = temp; temp += matricesSizes[0];
	cPC1 = temp; temp += matricesSizes[1];
	cPC2 = temp; temp += matricesSizes[2];
	cE = temp; temp += matricesSizes[3];
	cPMatrix = temp; temp += matricesSizes[4];
	cIPInverse = temp; temp += matricesSizes[5];
	cLCS = temp;


	// Initial operations 
	permuteMatrixCuda(input, cIP, 64); //initialPermutation(input);
	permuteMatrixCuda(shiftedKey, cPC1, 56); // PC1 of key
	fullShiftLCSCuda(shiftedKey);


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
		permuteMatrixCuda(permutedRoundKey, cPC2, 48);//roundKeyPermutation(permutedRoundKey);
		generateShiftedKeyCuda(i, shiftedKey, cLCS);

		// Expansion permutation
		permuteMatrixCuda(input, cE, 48);//expandPermutation(input); // 48 bits

		// XOR with permuted round key
		input ^= permutedRoundKey;

		// Substitution S-boxes
		substituteCuda(input, sboxes); // 32 bits

		// "P-matrix" permutation i.e. mix/shuffle
		permuteMatrixCuda(input, cPMatrix, 32);// mixPermutation(input);

		// XOR with preserved left side
		result += left ^ input; // Result[31:0] = L XOR f[31:0];

		// End of loop
		input = result;
	}

	swapLRCuda(result);
	permuteMatrixCuda(result, cIPInverse, 64);//reverseInitialPermutation(result);
	results[tid] = result;
}

__global__ void EncryptDESCudaDebug(uint64_t* messages, uint64_t* keys, unsigned char* matrices, unsigned char* sboxes, uint64_t* results, uint64_t* debug, int n)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint64_t result; // setting alias for encryption

	uint64_t input = messages[tid];
	uint64_t shiftedKey = keys[tid];
	uint64_t permutedRoundKey;
	uint64_t left; // last 32 bits of plaintext/input to algorithm are preserved in this variable 
	
	// load matrices
	// essential variables
	unsigned char* cIP, * cPC1, * cPC2, * cE, * cPMatrix, * cIPInverse, * cLCS;
	int matricesSizes[7] = { 64,56,48,48,32,64,16 };

	// loading matrices process
	unsigned char* temp = matrices;
	cIP = temp; temp += matricesSizes[0];
	cPC1 = temp; temp += matricesSizes[1];
	cPC2 = temp; temp += matricesSizes[2];
	cE = temp; temp += matricesSizes[3];
	cPMatrix = temp; temp += matricesSizes[4];
	cIPInverse = temp; temp += matricesSizes[5];
	cLCS = temp;

	// Initial operations 
	permuteMatrixCuda(input, cIP, 64); //initialPermutation(input);
	debug[0 + tid * n] = input;
	debug[1 + tid * n] = shiftedKey;
	permuteMatrixCuda(shiftedKey, cPC1, 56); // PC1 of key
	debug[2 + tid * n] = shiftedKey;
	for (int i = 0; i < 16; i++)
	{
		// Preserving L,R.
		// preserve right side (Result[63:32] = Input[31:0])
		result = input;
		result <<= 32;
		// preserve left side
		left = input >> 32;

		// Round key
		generateShiftedKeyCuda(i, shiftedKey, cLCS);
		debug[3 + tid * n] = shiftedKey;
		permutedRoundKey = shiftedKey;
		permuteMatrixCuda(permutedRoundKey, cPC2, 48);//roundKeyPermutation(permutedRoundKey);
		debug[4 + tid * n] = permutedRoundKey;

		// Expansion permutation
		permuteMatrixCuda(input, cE, 48);//expandPermutation(input); // 48 bits
		debug[5 + tid * n] = input;

		// XOR with permuted round key
		input ^= permutedRoundKey;
		debug[6 + tid * n] = input;
		// Substitution S-boxes
		substituteCuda(input, sboxes); // 32 bits
		debug[7 + tid * n] = input;

		// "P-matrix" permutation i.e. mix/shuffle
		permuteMatrixCuda(input, cPMatrix, 32);// mixPermutation(input);
		debug[8 + tid * n] = input;

		// XOR with preserved left side
		result += left ^ input; // Result[31:0] = L XOR f[31:0];

		// End of loop
		input = result;
	}

	swapLRCuda(result);
	debug[9 + tid * n] = result;
	permuteMatrixCuda(result, cIPInverse, 64);//reverseInitialPermutation(result);
	debug[10 + tid * n] = result;
	results[tid] = result;
	// debug final point
	debug[11 + tid * n] = messages[tid];
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

__device__ void substituteCuda(uint64_t& input, unsigned char* sboxes)
{
	uint64_t result = 0; uint64_t temp;
	uint8_t y, x;
	uint8_t in;

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
		temp = sboxes[i*64+y*16+x];
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

// Preemptively shifting all keys using LCS matrix.
__device__ void fullShiftLCSCuda(uint64_t& roundKey)
{
	uint32_t left, right;
	uint64_t mask28Bits = 268435455; // covers first 28 bits

	// getting left and right sides
	right = roundKey & mask28Bits;
	mask28Bits <<= 28;
	mask28Bits = roundKey & mask28Bits;
	left = mask28Bits >> 28;
	
	// circular shifts
	leftCircularShiftCuda(left, 28);
	leftCircularShiftCuda(right, 28);

	// copying left and right shifted keys to roundKey.
	roundKey = left;
	roundKey <<= 28;
	roundKey += right;
}