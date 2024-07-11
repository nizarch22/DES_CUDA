#include <cstdlib>
// External
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DES_CUDA.cuh"


// Definitions
#define NUM_THREADS 128
#define NUM_TESTS 300

__global__ void EncryptDESCuda(uint64_t* messages, uint64_t* keys, unsigned char** matrices1D, uint64_t* results)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	// load matrices
	// IP, PC1,PC2, SBox, PMatrix, IPInverse

	
	uint64_t result; // setting alias for decryption

	uint64_t input = messages[0];
	uint64_t shiftedKey = keys[0];
	uint64_t permutedRoundKey;
	uint64_t left; // last 32 bits of plaintext/input to algorithm are preserved in this variable 
	unsigned char* IP, * PC1, * PC2, * E, * PMatrix, * IPInverse, * LCS;
	unsigned char a[3] = { 1,2,3 };
	unsigned char* matrices[7];
	int matricesSizes[7] = { 64,56,48,48,32,64,16 };
	int count = 0;
	for (int i = 0; i < 7; i++)
	{
		matrices[i] = matrices1D;
		count += matricesSizes[i];
	}
	IP = matrices[0];
	PC1 = matrices[1];
	PC2 = matrices[2];
	E = matrices[3];
	PMatrix = matrices[4];
	IPInverse = matrices[5];
	LCS = matrices[6];
	// Initial operations 
	permuteMatrix(input, IP, 64); //initialPermutation(input);
	permuteMatrix(shiftedKey, PC1, 56); // PC1 of key

	for (int i = 0; i < 16; i++)
	{
		// Preserving L,R.
		// preserve right side (Result[63:32] = Input[31:0])
		result = input;
		result <<= 32;
		// preserve left side
		left = input >> 32;

		// Round key
		generateShiftedKey(i, shiftedKey);
		permutedRoundKey = shiftedKey;
		permuteMatrix(permutedRoundKey, PC2, 48);//roundKeyPermutation(permutedRoundKey);

		// Expansion permutation
		permuteMatrix(input, E, 48);//expandPermutation(input); // 48 bits

		// XOR with permuted round key
		input ^= permutedRoundKey;

		// Substitution S-boxes
		substitute(input); // 32 bits

		// "P-matrix" permutation i.e. mix/shuffle
		permuteMatrix(input, PMatrix, 32);// mixPermutation(input);

		// XOR with preserved left side
		result += left ^ input; // Result[31:0] = L XOR f[31:0];

		// End of loop
		input = result;
	}

	swapLR(result);
	permuteMatrix(result, IPInverse, 64);//reverseInitialPermutation(result);
	results[0] = result;
}

__device__ void permuteMatrix(uint64_t& input, const unsigned char* P, const unsigned int size)
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
__device__ void generateShiftedKey(const int& index, uint64_t& roundKey)
{
	uint32_t left, right;
	uint64_t mask28Bits = 268435455; // covers first 28 bits

	// getting left and right sides
	right = roundKey & mask28Bits;
	mask28Bits <<= 28;
	mask28Bits = roundKey & mask28Bits;
	left = mask28Bits >> 28;

	// circular shifts
	//leftCircularShift(left, LCS[index]);
	//leftCircularShift(right, LCS[index]);

	// copying left and right shifted keys to roundKey.
	roundKey = left;
	roundKey <<= 28;
}
__device__ void leftCircularShift(uint32_t& input, uint8_t times)
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
__device__ void substitute(uint64_t& input)
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
		//temp = SBoxes[i][y * 16 + x];
		temp = 0;
		result += temp << (4 * i);

		// next bits
		input >>= 6;
	}
	input = result;
}
__device__ void swapLR(uint64_t& input) // Swap left (32 bit) and right (32 bit) parts of the 64 bit input.
{
	uint64_t temp = input;
	// containing left side 
	temp >>= 32;

	// right side moved to left
	input <<= 32;

	// left side moved to right
	input += temp;
}