#define __CUDACC__
#include <cstdlib>
// External
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DES_CUDA.cuh"
// Primary header is compatible with pre-C++11, collective algorithm headers require C++11
#include <cooperative_groups.h>
using namespace cooperative_groups;
__device__ void swapLRCuda(unsigned char* input, unsigned char* sharedCopy); // Swap left (32 bit) and right (32 bit) parts of the 64 bit input.
__device__ void substituteCuda(unsigned char* input, uint16_t* sharedX, uint16_t* sharedY, unsigned char* sharedOutput);
__device__ void substituteCudaDebug(unsigned char* input, uint16_t* sharedX, uint16_t* sharedY, unsigned char* sharedOutput, unsigned char* debug, uint64_t* debugInt);
__device__ void leftCircularShiftCuda(unsigned char* input, unsigned char* sharedCopy, uint8_t times);
__device__ void rightCircularShiftCuda(unsigned char* input, unsigned char* sharedCopy, uint8_t times);
__device__ void permuteMatrixCuda(unsigned char* input, unsigned char* sharedCopy, const unsigned char* P, const unsigned int size);


__device__ void copy(unsigned char* debug, unsigned char* target, int num)
{
	debug[threadIdx.x + num*64 + blockIdx.x * 150*64] = target[threadIdx.x];
	__syncthreads();
}

__global__ void EncryptDESCuda(uint64_t* messages, uint64_t* keys, uint64_t* results, const unsigned char* d_matricesConst, const unsigned char* d_SBoxesConst)
{
	// Kernel iterations shared memory
	__shared__ unsigned char sharedInput[64];
	__shared__ unsigned char sharedLeft[64];
	__shared__ unsigned char sharedResult[64];
	__shared__ unsigned char sharedKey[64];
	__shared__ unsigned char sharedRoundkey[64];
	__shared__ uint64_t result; // setting alias for encryption

	// General shared array. Typically for copying input. Used in the following functions: permuteMatrixCuda, swapLRCuda, leftCircularShiftCuda, rightCircularShiftCuda
	__shared__ unsigned char sharedCopy[64];
	// Special arrays for 'subsituteCuda' function:
	__shared__ uint16_t sharedX[8];
	__shared__ uint16_t sharedY[8];

	uint64_t input;
	uint64_t shiftedKey;

	// Initializations
	const int matricesIndices[7] = { 0,64,120,168,216,248,312 };
	input = messages[blockIdx.x];
	shiftedKey = keys[blockIdx.x];
	sharedInput[threadIdx.x] = 0;
	sharedLeft[threadIdx.x] = 0;
	sharedResult[threadIdx.x] = 0;
	sharedKey[threadIdx.x] = 0;
	sharedRoundkey[threadIdx.x] = 0;
	sharedCopy[threadIdx.x] = 0;
	if (threadIdx.x < 8)
	{
		sharedX[threadIdx.x] = 0;
		sharedY[threadIdx.x] = 0;
		result = 0;
	}
	__syncthreads();

	// Initial operations 
	// The 64 bits of message,key (uint64_t) are converted into 64 bytes (unsigned char) so that they are easily parallelizable. 
	sharedInput[threadIdx.x] = (input >> threadIdx.x) & 1;
	sharedKey[threadIdx.x] = (shiftedKey >> threadIdx.x) & 1;
	__syncthreads();

	// Initial permutation parallelized
	permuteMatrixCuda(sharedInput, sharedCopy, &d_matricesConst[matricesIndices[0]], 64); //initialPermutation(input);
	permuteMatrixCuda(sharedKey, sharedCopy, &d_matricesConst[matricesIndices[1]], 56); // PC1 of key

	for (int i = 0; i < 16; i++)
	{
		// Preserving L,R.
		// preserve right side, R. (Result[63:32] = Input[31:0])
		sharedResult[threadIdx.x] = (threadIdx.x >= 32) ? sharedInput[threadIdx.x - 32] : 0;

		// preserve left side, L. (Left[31:0] = Input[63:32])
		sharedLeft[threadIdx.x] = (threadIdx.x < 32) ? sharedInput[threadIdx.x + 32] : 0;
		__syncthreads();

		// Round key
		// Shift key to the right for the next iteration
		leftCircularShiftCuda(sharedKey, sharedCopy, d_matricesConst[matricesIndices[6]+15-i]);

		// Preserve the current shifted key (sharedKey) for the next iteration.
		sharedRoundkey[threadIdx.x] = sharedKey[threadIdx.x];
		__syncthreads();

		// Permutation PC2 of the roundKey
		permuteMatrixCuda(sharedRoundkey, sharedCopy, &d_matricesConst[matricesIndices[2]], 48);//roundKeyPermutation(permutedRoundKey);

		// Expansion permutation of input's right side (R).
		permuteMatrixCuda(sharedInput, sharedCopy, &d_matricesConst[matricesIndices[3]], 48);//expandPermutation(input); // 48 bits

		// XOR with permuted round key
		sharedInput[threadIdx.x] = sharedInput[threadIdx.x] ^ sharedRoundkey[threadIdx.x];
		__syncthreads();

		// Substitution S-boxes
		substituteCuda(sharedInput, sharedX, sharedY, d_SBoxesConst);

		// "P-matrix" permutation i.e. mix/shuffle
		permuteMatrixCuda(sharedInput, sharedCopy, &d_matricesConst[matricesIndices[4]], 32);// mixPermutation(input);

		// XOR with preserved left side
		if (threadIdx.x < 32)
		{
			sharedResult[threadIdx.x] = sharedLeft[threadIdx.x] ^ sharedInput[threadIdx.x];
		}
		__syncthreads();

		// End of loop
		sharedInput[threadIdx.x] = sharedResult[threadIdx.x];

		__syncthreads();
	}

	swapLRCuda(sharedResult, sharedCopy);

	permuteMatrixCuda(sharedResult, sharedCopy, &d_matricesConst[matricesIndices[5]], 64);//reverseInitialPermutation(result);

	if (threadIdx.x == 0)
	{
		result = 0;
		for (int i = 0; i < 64; i++)
		{
			result <<= 1;
			result += sharedResult[63 - i] & 1;
		}
		results[blockIdx.x] = result;
	}
	__syncthreads();
}
//
__global__ void DecryptDESCuda(uint64_t* encryptions, uint64_t* keys, uint64_t* results, const unsigned char* d_matricesConst, const unsigned char* d_SBoxesConst)
{
	// Kernel iterations shared memory
	__shared__ unsigned char sharedInput[64];
	__shared__ unsigned char sharedLeft[64];
	__shared__ unsigned char sharedResult[64];
	__shared__ unsigned char sharedKey[64];
	__shared__ unsigned char sharedRoundkey[64];
	__shared__ uint64_t result; // setting alias for encryption

	// General shared array. Typically for copying input. Used in the following functions: permuteMatrixCuda, swapLRCuda, leftCircularShiftCuda, rightCircularShiftCuda
	__shared__ unsigned char sharedCopy[64];
	// Special arrays for 'subsituteCuda' function:
	__shared__ uint16_t sharedX[8];
	__shared__ uint16_t sharedY[8];

	uint64_t input;
	uint64_t shiftedKey;

	// Initializations
	const int matricesIndices[7] = { 0,64,120,168,216,248,312 };
	input = encryptions[blockIdx.x];
	shiftedKey = keys[blockIdx.x];
	sharedInput[threadIdx.x] = 0;
	sharedLeft[threadIdx.x] = 0;
	sharedResult[threadIdx.x] = 0;
	sharedKey[threadIdx.x] = 0;
	sharedRoundkey[threadIdx.x] = 0;
	sharedCopy[threadIdx.x] = 0;
	if (threadIdx.x < 8)
	{
		sharedX[threadIdx.x] = 0;
		sharedY[threadIdx.x] = 0;
	}
	if (threadIdx.x == 0)
	{
		result = 0;
	}
	__syncthreads();

	// Initial operations 
	// The 64 bits of message,key (uint64_t) are converted into 64 bytes (unsigned char) so that they are easily parallelizable. 
	sharedInput[threadIdx.x] = (input >> threadIdx.x) & 1;
	sharedKey[threadIdx.x] = (shiftedKey >> threadIdx.x) & 1;
	__syncthreads();

	// Initial permutation parallelized
	permuteMatrixCuda(sharedInput, sharedCopy, &d_matricesConst[matricesIndices[0]], 64); //initialPermutation(input);
	permuteMatrixCuda(sharedKey, sharedCopy, &d_matricesConst[matricesIndices[1]], 56); // PC1 of key

	__syncthreads();
	for (int i = 0; i < 16; i++)
	{
		// Preserving L,R.
		// preserve right side, R. (Result[63:32] = Input[31:0])
		sharedResult[threadIdx.x] = (threadIdx.x >= 32) ? sharedInput[threadIdx.x - 32] : 0;

		// preserve left side, L. (Left[31:0] = Input[63:32])
		sharedLeft[threadIdx.x] = (threadIdx.x < 32) ? sharedInput[threadIdx.x + 32] : 0;
		__syncthreads();

		// Round key
		// Preserve the current shifted key (sharedKey) for the next iteration.
		sharedRoundkey[threadIdx.x] = sharedKey[threadIdx.x];
		__syncthreads();

		// Permutation PC2 of the roundKey
		permuteMatrixCuda(sharedRoundkey, sharedCopy, &d_matricesConst[matricesIndices[2]], 48);//roundKeyPermutation(permutedRoundKey);

		// Shift key to the right for the next iteration
		rightCircularShiftCuda(sharedKey, sharedCopy, d_matricesConst[matricesIndices[6] + 15 - i]);

		// Expansion permutation of input's right side (R).
		permuteMatrixCuda(sharedInput, sharedCopy, &d_matricesConst[matricesIndices[3]], 48);//expandPermutation(input); // 48 bits

		// XOR with permuted round key
		sharedInput[threadIdx.x] = sharedInput[threadIdx.x] ^ sharedRoundkey[threadIdx.x];
		__syncthreads();

		// Substitution S-boxes
		substituteCuda(sharedInput, sharedX, sharedY, d_SBoxesConst);

		// "P-matrix" permutation i.e. mix/shuffle
		permuteMatrixCuda(sharedInput, sharedCopy, &d_matricesConst[matricesIndices[4]], 32);// mixPermutation(input);

		// XOR with preserved left side
		if (threadIdx.x < 32)
		{
			sharedResult[threadIdx.x] = sharedLeft[threadIdx.x] ^ sharedInput[threadIdx.x];
		}
		__syncthreads();

		// End of loop
		sharedInput[threadIdx.x] = sharedResult[threadIdx.x];

		__syncthreads();
	}

	swapLRCuda(sharedResult, sharedCopy);
	permuteMatrixCuda(sharedResult, sharedCopy, &d_matricesConst[matricesIndices[5]], 64);//reverseInitialPermutation(result);

	if (threadIdx.x == 0)
	{
		result = 0;
		for (int i = 0; i < 64; i++)
		{
			result <<= 1;
			result += sharedResult[63 - i] & 1;
		}
		results[blockIdx.x] = result;
	}
	__syncthreads();
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


//__global__ void EncryptDESCudaDebug(uint64_t* messages, uint64_t* keys, unsigned char* matrices, unsigned char* sboxes, uint64_t* results, uint64_t* debug, int n)
//{
//	int tid = threadIdx.x + blockIdx.x * blockDim.x;
//	uint64_t result; // setting alias for encryption
//
//	uint64_t input = messages[tid];
//	uint64_t shiftedKey = keys[tid];
//	uint64_t permutedRoundKey;
//	uint64_t left; // last 32 bits of plaintext/input to algorithm are preserved in this variable 
//	
//	// load matrices
//	// essential variables
//	__shared__ unsigned char* cIP, * cPC1, * cPC2, * cE, * cPMatrix, * cIPInverse, * cLCS;
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
//	// Initial operations 
//	permuteMatrixCuda(input, cIP, 64); //initialPermutation(input);
//	debug[0 + tid * n] = input;
//	debug[1 + tid * n] = shiftedKey;
//	permuteMatrixCuda(shiftedKey, cPC1, 56); // PC1 of key
//	debug[2 + tid * n] = shiftedKey;
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
//		generateShiftedKeyCuda(i, shiftedKey, cLCS);
//		debug[3 + tid * n] = shiftedKey;
//		permutedRoundKey = shiftedKey;
//		permuteMatrixCuda(permutedRoundKey, cPC2, 48);//roundKeyPermutation(permutedRoundKey);
//		debug[4 + tid * n] = permutedRoundKey;
//
//		// Expansion permutation
//		permuteMatrixCuda(input, cE, 48);//expandPermutation(input); // 48 bits
//		debug[5 + tid * n] = input;
//
//		// XOR with permuted round key
//		input ^= permutedRoundKey;
//		debug[6 + tid * n] = input;
//		// Substitution S-boxes
//		substituteCuda(input); // 32 bits
//		debug[7 + tid * n] = input;
//
//		// "P-matrix" permutation i.e. mix/shuffle
//		permuteMatrixCuda(input, cPMatrix, 32);// mixPermutation(input);
//		debug[8 + tid * n] = input;
//
//		// XOR with preserved left side
//		result += left ^ input; // Result[31:0] = L XOR f[31:0];
//
//		// End of loop
//		input = result;
//	}
//
//	swapLRCuda(result);
//	debug[9 + tid * n] = result;
//	permuteMatrixCuda(result, cIPInverse, 64);//reverseInitialPermutation(result);
//	debug[10 + tid * n] = result;
//	results[tid] = result;
//	// debug final point
//	debug[11 + tid * n] = messages[tid];
//}

__device__ void permuteMatrixCuda(unsigned char* input, unsigned char* sharedCopy, const unsigned char* P, unsigned int size)
{
	sharedCopy[threadIdx.x] = input[threadIdx.x];
	__syncthreads();

	// if thread is bigger than the alloted permutation size, make slot equal to 0.
	// Note (threadIdx.x%size) is used in case of a memory violation. But, this precaution is rendered unnecessary by the ternary operator.
	unsigned char bit;
	bit = (threadIdx.x >= size) ? 0 : (sharedCopy[P[threadIdx.x%size] - 1] & 1);
	input[threadIdx.x] = bit;
	__syncthreads();
}

__device__ void leftCircularShiftCuda(unsigned char* input, unsigned char* sharedCopy, uint8_t times)
{
	// copying the key
	sharedCopy[threadIdx.x] = input[threadIdx.x];
	__syncthreads();

	// set offset to determine left and right (L,R) sides of key.
	int offset = 28 * (threadIdx.x / 28);

	int index = offset + (threadIdx.x + times) % 28;

	// accounting for edge case with 64 bits.
	// Note shifting is not necessary here, as we do not care about the last 8 bits. 
	index = (index >= 56) ? (offset+index%8) : index;

	// Finally applying the shift
	input[index] = sharedCopy[threadIdx.x];
	__syncthreads();
}

// Note: maximum of 28 shifts at call of function
__device__ void rightCircularShiftCuda(unsigned char* input, unsigned char* sharedCopy, uint8_t times)
{
	// copying the key
	sharedCopy[threadIdx.x] = input[threadIdx.x];
	__syncthreads();

	// set offset to determine left and right (L,R) sides of key.
	int offset = 28 * (threadIdx.x / 28);

	int index = offset + (threadIdx.x + 28 - times) % 28;

	// accounting for edge case with 64 bits.
	// Note shifting is not necessary here, as we do not care about the last 8 bits. 
	index = (index >= 56) ? (offset + index % 8) : index;

	// Finally applying the shift
	input[index] = sharedCopy[threadIdx.x];
	__syncthreads();
}



__device__ void substituteCuda(unsigned char* input, uint16_t* sharedX, uint16_t* sharedY, const unsigned char* d_SBoxesConst)
{
	// 16 inputs (8 x,y pairs) and 8 outputs - 8 extractions from SBox. 
	// 64 threads will allow for 8 simulataneous extractions.

	// Thus, 16 threads will suffice to calculate x,y pairs.
	// 8 threads for each of x and y.

	int tid = threadIdx.x;
	int setIndex, index, threadPos;
	uint8_t x = 0;
	uint8_t y = 0;
	unsigned char byte, bit;

	// Y calculation
	// Threads 0 -> 7 work here - First warp
	if (tid < 8)
	{
		// y = b5,b0 then b11,b6, b17,b12 ... b47,b42 i.e. tid*6 + 5, tid*6
		y = (input[tid * 6 + 5]) << 1;
		y |= input[tid * 6];
		sharedY[tid] = y;
	}

	// X calculation 
	// Threads 32 -> 39 work here - Second warp
	if (tid >= 32 && tid < 40)
	{
		// x = b4,b3,b2,b1 then b10,...,b7 i.e. tid * 6 + 4, ..., tid * 6 + 1
		// note we reduced tid by 8, as we work with threads 8->15.
		// i.e. (x's index) * 6 + 4, ..., (x's index) * 6 + 1
		index = tid - 32;
		for (int i = 0; i < 4; i++)
		{
			x |= input[index * 6 + (i + 1)] << i;
		}
		sharedX[index] = x;
	}

	// Warps are joined here.
	__syncthreads(); 

	// Extract Sbox output and place it into 'input'
	if (tid < 32)
	{
		setIndex = tid / 4;
		threadPos = tid % 4;
		byte = d_SBoxesConst[(setIndex << 6) + (sharedY[setIndex] << 4) + (sharedX[setIndex])];
		byte >>= threadPos;
		bit = byte & 1;

		input[tid] = byte & 1;
	}

	// Wipe the last 32 bits of input.
	if (tid >= 32)
	{
		input[tid] = 0;
	}
	__syncthreads();

}
//__device__ void substituteCudaDebug(unsigned char* input, uint16_t* sharedX, uint16_t* sharedY, unsigned char* sharedOutput, unsigned char* debug, uint64_t* debugInt)
//{
//	// 16 inputs (8 x,y pairs) and 8 outputs - 8 extractions from SBox. 
//	// 64 threads will allow for 8 simulataneous extractions.
//
//	// Thus, 16 threads will suffice to calculate x,y pairs.
//	// 8 threads for each of x and y.
//
//	int tid = threadIdx.x;
//	int setIndex, index, threadPos;
//	uint8_t x = 0;
//	uint8_t y = 0;
//	unsigned char byte, bit;
//
//	// Y calculation
//	// Threads 0 -> 7 work here - First warp
//	if (tid < 8)
//	{
//		// y = b5,b0 then b11,b6, b17,b12 ... b47,b42 i.e. tid*6 + 5, tid*6
//		y = (input[tid * 6 + 5]) << 1;
//		y |= input[tid * 6];
//		sharedY[tid] = y;
//	}
//
//	// X calculation 
//	// Threads 32 -> 39 work here - Second warp
//	if (tid >= 32 && tid < 40)
//	{
//		// x = b4,b3,b2,b1 then b10,...,b7 i.e. tid * 6 + 4, ..., tid * 6 + 1
//		// note we reduced tid by 8, as we work with threads 8->15.
//		// i.e. (x's index) * 6 + 4, ..., (x's index) * 6 + 1
//		index = tid - 32;
//		for (int i = 0; i < 4; i++)
//		{
//			x |= input[index * 6 + (i + 1)] << i;
//		}
//		sharedX[index] = x;
//	}
//
//	// Y calculation
//	// Threads 0 -> 7 work here - First warp
//	//if (tid < 16)
//	//{
//	//	// y = b5,b0 then b11,b6, b17,b12 ... b47,b42 i.e. tid*6 + 5, tid*6
//	//	index = tid / 2;
//	//	y = (input[index * 6 + 5]) << 1;
//	//	y += input[tid * 6];
//	//	sharedY[tid] = y;
//	//}
//		
//
//	// X calculation 
//	// Threads 32 -> 39 work here - Second warp
//	//if (tid >= 32)
//	//{
//	//	// x = b4,b3,b2,b1 then b10,...,b7 i.e. tid * 6 + 4, ..., tid * 6 + 1
//	//	// note we reduced tid by 8, as we work with threads 8->15.
//	//	// i.e. (x's index) * 6 + 4, ..., (x's index) * 6 + 1
//
//	//	// 0,1,2,3 (meaning threads 32,...,35) together make a single x. Then, 4,...,7 etc.
//	//	// These are our 'thread sets' of 8. Each set will produce a value x, 8 in total. i.e. x's index will be setIndex.
//	//	//index = tid - 32;
//	//	//setIndex = index / 4;
//	//	//threadPos = index % 4;
//
//	//	//sharedOutput[tid] = input[6 * setIndex + (threadPos + 1)] << threadPos;
//
//	//	//if (threadPos == 0)
//	//	//{
//	//	//	sharedTemp[setIndex]
//	//	//}
//	//	//sharedTemp[setIndex] += x;
//	//	//for (int i = 1; i < 5; i++)
//	//	//{
//	//	//	x += input[(tid - 32) * 6 + i] << (i - 1);
//	//	//}
//
//	//	//if (threadPos==0)
//	//	//{
//	//	//	// add all the x values
//	//	//	x = sharedOutput[tid];
//	//	//	for (int i = 1; i < 4; i++)
//	//	//	{
//	//	//		x |= sharedOutput[tid+i];
//	//	//	}
//
//	//	//	//  Finally set the x values
//	//	//	sharedX[setIndex] = x;
//	//	//}
//	//}
//	
//
//
//	__syncthreads(); // Warps are joined here.
//
//	// Wipe input.
//	input[tid] = 0;
//	__syncthreads();
//
//	// Extract Sbox output and place it into 'input'
//	if (tid < 32)
//	{
//		setIndex = tid / 4;
//		threadPos = tid % 4;
//		byte = d_SBoxesConst[setIndex][(sharedY[setIndex] << 4) + sharedX[setIndex]];
//		byte >>= threadPos;
//		bit = byte & 1;
//
//		input[tid] = byte & 1;
//	}
//
//	// Debug
//	if (tid < 8)
//	{
//		debugInt[tid + blockIdx.x * 150] = d_SBoxesConst[tid][(sharedY[tid] << 4) + sharedX[tid]];
//		debugInt[tid + 8 + blockIdx.x * 150] = sharedX[tid];
//		debugInt[tid + 16 + blockIdx.x * 150] = sharedY[tid];
//	}
//
//
//	//// Threads 0 -> 7 work here - First warp
//	//if (tid < 8)
//	//{
//	//	//return;
//	//	sharedOutput[tid] = d_SBoxesConst[tid][(sharedY[tid] << 4) + sharedX[tid]];
//	//	debugInt[tid + blockIdx.x * 150] = sharedOutput[tid];
//	//	debugInt[tid + 8 + blockIdx.x * 150] = sharedX[tid];
//	//	debugInt[tid + 16 + blockIdx.x * 150] = sharedY[tid];
//	//	
//	//	//debugInt[2 + tid] = sharedOutput[tid];
//	//}
//
//	//// Extract 4 bits
//	//// Threads 32 -> 63 work here - Second warp
//	//__syncthreads();
//	//int index; int outputIndex; int bitIndex;
//	//unsigned char bit;
//	//unsigned char byte;
//	//if (tid >= 32 && tid < 64)
//	//{
//	//	index = tid - 32;
//	//	outputIndex = index / 4;
//	//	bitIndex = (index % 4);
//	//	byte = sharedOutput[outputIndex];
//	//	byte >>= bitIndex;
//	//	bit = byte&1;
//	//	//bit = (sharedOutput[outputIndex] >> bitIndex) & 1;
//
//	//	// Place bits into input
//	//	input[index] = bit;
//	//}
//
//	__syncthreads();
//}
//__device__ void substituteCuda(unsigned char* input)
//{
//	// 16 inputs (8 x,y pairs) and 8 outputs - 8 extractions from SBox. 
//	// 64 threads will allow for 8 simulataneous extractions.
//	
//
//	// Thus, 16 threads will suffice to calculate x,y pairs.
//	// 8 threads for each of x and y.
//	__shared__ uint8_t sharedX[8];
//	__shared__ uint8_t sharedY[8];
//	__shared__ unsigned char sharedOutput[8];
//
//	int tid = threadIdx.x;
//	// grouping
//	//thread_block tb = this_thread_block();
//	//thread_block_tile<8> tileX = tiled_partition<
//
//	// Y calculation
//	// Threads 0 -> 7 work here - First warp
//	if (tid < 8)
//	{
//		// y = b5,b0 then b11,b6, b17,b12 ... b47,b42 i.e. (tid+1)*5, tid*6
//		uint8_t y = input[tid * 6 + 5] << 1;
//		y += input[tid * 6];
//		sharedY[tid] = y;
//	}
//
//	// X calculation 
//	// Threads 32 -> 39 work here - Second warp
//	if (tid >= 32 && tid < 40)
//	{
//		// x = b4,b3,b2,b1 then b10,...,b7 i.e. tid * 6 + 4, ..., tid * 6 + 1
//		// note we reduced tid by 8, as we work with threads 8->15.
//		// i.e. (tid-8) * 6 + 4, ..., (tid-8) * 6 + 1
//		uint8_t x = 0;
//		for (int i = 1; i < 5; i++)
//		{
//			x += input[(tid - 32) * 6 + i] << i;
//		}
//		sharedX[tid - 32] = x;
//
//	}
//
//	// Extract Sbox output and place it into 'input'
//	__syncthreads(); // Warps are joined here.
//	// clean slate for input 
//	input[tid] = 0;
//	// Threads 0 -> 7 work here - First warp
//	if (tid < 8)
//	{
//		sharedOutput[tid] = d_SBoxesConst[sharedY[tid]][sharedX[tid]];
//	}
//
//	// Extract 4 bits
//	// Threads 32 -> 63 work here - Second warp
//	__syncthreads();
//	int index; int outputIndex; int bitIndex;
//	unsigned char bit;
//	if (tid >= 32 && tid < 64)
//	{
//		index = tid - 32;
//		outputIndex = index / 4;
//		bitIndex = (index % 4);
//
//		bit = (sharedOutput[outputIndex] >> bitIndex) & 1;
//
//		// Place bits into input
//		input[index] = bit;
//	}
//
//	__syncthreads();
//}
__device__ void swapLRCuda(unsigned char* input, unsigned char* sharedCopy) // Swap left (32 bit) and right (32 bit) parts of the 64 bit input.
{
	sharedCopy[threadIdx.x] = input[threadIdx.x];
	__syncthreads();

	input[threadIdx.x] = sharedCopy[(threadIdx.x + 32)%64];
	__syncthreads();
}