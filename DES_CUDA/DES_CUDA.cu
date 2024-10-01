#include <cstdlib>
// External
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DES_CUDA.cuh"

__device__ void generateReverseShiftedKeyCuda(const int& index, uint64_t& roundKey, unsigned char* cLCS);
__device__ void rightCircularShiftCuda(uint32_t& input, uint8_t times);
__device__ void fullShiftLCSCuda(uint64_t& roundKey);
__device__ void swapLRCuda(uint64_t& input); // Swap left (32 bit) and right (32 bit) parts of the 64 bit input.
__device__ void substituteCuda(uint64_t& input);
__device__ void leftCircularShiftCuda(uint32_t& input, uint8_t times);
__device__ void generateShiftedKeyCuda(const int& index, uint64_t& roundKey, unsigned char* cLCS);
__device__ void permuteMatrixCuda(unsigned char* input, const unsigned char* P, const unsigned int size);

__global__ void EncryptDESCuda(uint64_t* messages, uint64_t* keys, uint64_t* results)
{
	// figure out whether to use 64 or 64,56,48, etc.
	__shared__ unsigned char sharedInput[64];
	__shared__ unsigned char sharedLeft[64];
	__shared__ unsigned char sharedResult[64];
	__shared__ unsigned char sharedKey[64];
	__shared__ unsigned char sharedRoundkey[64];

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint64_t result; // setting alias for encryption

	__shared__ uint64_t input = messages[blockIdx.x];
	__shared__ uint64_t shiftedKey = keys[blockIdx.x];
	__shared__ uint64_t permutedRoundKey;
	__shared__ uint64_t left; // last 32 bits of plaintext/input to algorithm are preserved in this variable 

	// load matrices
	// essential variables
	unsigned char* cIP, * cPC1, * cPC2, * cE, * cPMatrix, * cIPInverse, * cLCS;
	int matricesSizes[7] = { 64,56,48,48,32,64,16 };

	// loading matrices process
	unsigned char* temp = d_matricesConst;
	cIP = temp; temp += matricesSizes[0];
	cPC1 = temp; temp += matricesSizes[1];
	cPC2 = temp; temp += matricesSizes[2];
	cE = temp; temp += matricesSizes[3];
	cPMatrix = temp; temp += matricesSizes[4];
	cIPInverse = temp; temp += matricesSizes[5];
	cLCS = temp;


	// Initial operations 
	// The 64 bits of message,key (uint64_t) are converted into 64 bytes (unsigned char) so that they are easily parallelizable. 
	sharedInput[threadIdx.x] = (input >> threadIdx.x) & 1;
	sharedKey[threadIdx.x] = (shiftedKey >> threadIdx.x) & 1;
	__syncthreads();

	// Initial permutation parallelized
	permuteMatrixCuda(sharedInput, cIP, 64); //initialPermutation(input);
	permuteMatrixCuda(sharedKey, cPC1, 56); // PC1 of key

	for (int i = 0; i < 16; i++)
	{
		// Preserving L,R.
		// preserve right side, R. (Result[63:32] = Input[31:0])
		sharedResult[threadIdx.x] = sharedInput[threadIdx.x];

		// preserve left side, L.
		sharedLeft[threadIdx.x] = sharedInput[threadIdx.x];
		__syncthreads();
		//left = input >> 32;
		
		// Round key
		generateShiftedKeyCuda(i, sharedKey, cLCS);
		
		// preserve the current shifted key (sharedKey) for the next iteration.
		sharedRoundkey[threadIdx.x] = sharedKey[threadIdx.x];
		__syncthreads();

		// Permutation PC2 of the roundKey
		permuteMatrixCuda(sharedRoundkey, cPC2, 48);//roundKeyPermutation(permutedRoundKey);

		// Expansion permutation of input's right side (R).
		permuteMatrixCuda(sharedInput, cE, 48);//expandPermutation(input); // 48 bits

		// XOR with permuted round key
		sharedInput[threadIdx.x] = sharedInput[threadIdx.x] ^ sharedRoundkey[threadIdx.x];
		__syncthreads();

		// Substitution S-boxes
		substituteCuda(input); // 32 bits

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
__global__ void DecryptDESCuda(uint64_t* encryptions, uint64_t* keys, uint64_t* results)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint64_t result; // setting alias for decryption

	uint64_t input = encryptions[tid];
	uint64_t shiftedKey = keys[tid];
	uint64_t permutedRoundKey;
	uint64_t left; // last 32 bits of plaintext/input to algorithm are preserved in this variable 

	// load matrices
	// essential variables
	unsigned char* cIP, * cPC1, * cPC2, * cE, * cPMatrix, * cIPInverse, * cLCS;
	int matricesSizes[7] = { 64,56,48,48,32,64,16 };

	// loading matrices process
	unsigned char* temp = d_matricesConst;
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
		generateReverseShiftedKeyCuda(i, shiftedKey, cLCS);

		// Expansion permutation
		permuteMatrixCuda(input, cE, 48);//expandPermutation(input); // 48 bits

		// XOR with permuted round key
		input ^= permutedRoundKey;

		// Substitution S-boxes
		substituteCuda(input); // 32 bits

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

//__global__ void DecryptDESCudaDebug(uint64_t* encryptions, uint64_t* keys, unsigned char* matrices, unsigned char* sboxes, uint64_t* results)
//{
//	int tid = threadIdx.x + blockIdx.x * blockDim.x;
//	uint64_t result; // setting alias for decryption
//
//	uint64_t input = encryptions[tid];
//	uint64_t shiftedKey = keys[tid];
//	uint64_t permutedRoundKey;
//	uint64_t left; // last 32 bits of plaintext/input to algorithm are preserved in this variable 
//
//	// load matrices
//	// essential variables
//	unsigned char* cIP, * cPC1, * cPC2, * cE, * cPMatrix, * cIPInverse, * cLCS;
//	int matricesSizes[7] = { 64,56,48,48,32,64,16 };
//
//	// loading matrices process
//	unsigned char* temp = matrices;
//	cIP = temp; temp += matricesSizes[0];
//	cPC1 = temp; temp += matricesSizes[1];
//	cPC2 = temp; temp += matricesSizes[2];
//	cE = temp; temp += matricesSizes[3];
//	cPMatrix = temp; temp += matricesSizes[4];
//	cIPInverse = temp; temp += matricesSizes[5];
//	cLCS = temp;
//
//
//	// Initial operations 
//	permuteMatrixCuda(input, cIP, 64); //initialPermutation(input);
//	permuteMatrixCuda(shiftedKey, cPC1, 56); // PC1 of key
//	fullShiftLCSCuda(shiftedKey);
//
//
//	for (int i = 0; i < 16; i++)
//	{
//		// Preserving L,R.
//		// preserve right side (Result[63:32] = Input[31:0])
//		result = input;
//		result <<= 32;
//		// preserve left side
//		left = input >> 32;
//
//		// Round key
//		permutedRoundKey = shiftedKey;
//		permuteMatrixCuda(permutedRoundKey, cPC2, 48);//roundKeyPermutation(permutedRoundKey);
//		generateReverseShiftedKeyCuda(i, shiftedKey, cLCS);
//
//		// Expansion permutation
//		permuteMatrixCuda(input, cE, 48);//expandPermutation(input); // 48 bits
//
//		// XOR with permuted round key
//		input ^= permutedRoundKey;
//
//		// Substitution S-boxes
//		substituteCuda(input); // 32 bits
//
//		// "P-matrix" permutation i.e. mix/shuffle
//		permuteMatrixCuda(input, cPMatrix, 32);// mixPermutation(input);
//
//		// XOR with preserved left side
//		result += left ^ input; // Result[31:0] = L XOR f[31:0];
//
//		// End of loop
//		input = result;
//	}
//
//	swapLRCuda(result);
//	permuteMatrixCuda(result, cIPInverse, 64);//reverseInitialPermutation(result);
//	results[tid] = result;
//}


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
	__shared__ unsigned char* cIP, * cPC1, * cPC2, * cE, * cPMatrix, * cIPInverse, * cLCS;
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
		substituteCuda(input); // 32 bits
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

__device__ void permuteMatrixCuda(unsigned char* input, const unsigned char* P, unsigned int size)
{
	// Figure out how to not make warps here.
	unsigned char bit;
	if (threadIdx.x >= size)
		goto PERMUTE_FUNC_END;
	bit = input[P[threadIdx.x] - 1] & 1;
	__syncthreads();
	input[threadIdx.x] = bit;
PERMUTE_FUNC_END:
	__syncthreads();
}
__device__ void generateShiftedKeyCuda(const int& index, unsigned char* roundKey, unsigned char* cLCS)
{
	// circular shifts
	leftCircularShiftCuda(roundKey, cLCS[index]);
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
__device__ void leftCircularShiftCuda(unsigned char* input, uint8_t times)
{
	__shared__ unsigned char sharedKeyCopy[64];

	// copying the key
	sharedKeyCopy[threadIdx.x] = input[threadIdx.x];
	__syncthreads();

	// set offset to determine left and right (L,R) sides of key.
	int offset = 28 * (threadIdx.x / 28);

	int index = offset + (threadIdx.x + times) % 28;

	// accounting for edge case with 64 bits.
	// Note shifting is not necessary here, as we do not care about the last 8 bits. 
	index = (index >= 56) ? (offset+index%8): index;

	// Finally applying the shift
	input[index] = sharedKeyCopy[threadIdx.x];
	__syncthreads();
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
	// Try to not warp
	if (threadIdx.x >= 32)
		goto SUB_FUNC_END;


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
		temp = d_SBoxesConst[i*64+y*16+x];
		result += temp << (4 * i);

		// next bits
		input >>= 6;
	}
	input = result;

SUB_FUNC_END:
	__syncthreads();
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