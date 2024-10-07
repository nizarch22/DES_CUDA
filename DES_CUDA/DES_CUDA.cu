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
__device__ void substituteCuda(unsigned char* input, uint16_t* sharedX, uint16_t* sharedY, const unsigned char* d_SBoxesConst);
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
	__shared__ unsigned char sharedInput[128];
	__shared__ unsigned char sharedLeft[128];
	__shared__ unsigned char sharedResult[128];
	__shared__ unsigned char sharedKey[128];
	__shared__ unsigned char sharedRoundkey[128];
	__shared__ uint64_t result; // setting alias for encryption

	// General shared array. Typically for copying input. Used in the following functions: permuteMatrixCuda, swapLRCuda, leftCircularShiftCuda, rightCircularShiftCuda
	__shared__ unsigned char sharedCopy[128];
	// Special arrays for 'subsituteCuda' function:
	__shared__ uint16_t sharedX[16];
	__shared__ uint16_t sharedY[16];

	uint64_t input;
	uint64_t shiftedKey;
	const int matricesIndices[7] = { 0, 64,120,168,216,248,312 };

	int threadSetIndex = threadIdx.x >> 6; // divide by 64=2^6
	int threadIndex = (threadIdx.x >= 64) ? (threadIdx.x - 64) : threadIdx.x;

	// Initializations
	input = messages[(blockIdx.x << 1) + threadSetIndex];
	shiftedKey = keys[(blockIdx.x << 1) + threadSetIndex];
	sharedInput[threadIdx.x] = 0;
	sharedLeft[threadIdx.x] = 0;
	sharedResult[threadIdx.x] = 0;
	sharedKey[threadIdx.x] = 0;
	sharedRoundkey[threadIdx.x] = 0;
	sharedCopy[threadIdx.x] = 0;
	if (threadIndex < 8)
	{
		sharedX[threadIdx.x] = 0;
		sharedY[threadIdx.x] = 0;
	}
	if (threadIndex == 0)
	{
		result = 0;
	}
	__syncthreads();

	// Initial operations 
	// The 64 bits of message,key (uint64_t) are converted into 64 bytes (unsigned char) so that they are easily parallelizable. 
	sharedInput[threadIdx.x] = (input >> threadIndex) & 1;
	sharedKey[threadIdx.x] = (shiftedKey >> threadIndex) & 1;
	__syncthreads();

	// Initial permutation parallelized
	permuteMatrixCuda(sharedInput, sharedCopy, &d_matricesConst[matricesIndices[0]], 64); //initialPermutation(input);
	permuteMatrixCuda(sharedKey, sharedCopy, &d_matricesConst[matricesIndices[1]], 56); // PC1 of key

	for (int i = 0; i < 16; i++)
	{
		// Preserving L,R.
		// preserve right side, R. (Result[63:32] = Input[31:0])
		sharedResult[threadIdx.x] = (threadIndex >= 32) ? sharedInput[threadIdx.x - 32] : 0;

		// preserve left side, L. (Left[31:0] = Input[63:32])
		sharedLeft[threadIdx.x] = (threadIndex < 32) ? sharedInput[threadIdx.x + 32] : 0;
		// Round key
		// Shift key to the right for the next iteration
		leftCircularShiftCuda(sharedKey, sharedCopy, d_matricesConst[matricesIndices[6]+i]);

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
		if (threadIndex < 32)
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

	if (threadIndex == 0)
	{
		result = 0;
		for (int i = 0; i < 64; i++)
		{
			result <<= 1;
			result += sharedResult[63 - i + (threadSetIndex << 6)] & 1;
		}
		results[(blockIdx.x << 1) + threadSetIndex] = result;
	}
	__syncthreads();
}
//
__global__ void DecryptDESCuda(uint64_t* encryptions, uint64_t* keys, uint64_t* results, const unsigned char* d_matricesConst, const unsigned char* d_SBoxesConst)
{
	// Kernel iterations shared memory
	__shared__ unsigned char sharedInput[128];
	__shared__ unsigned char sharedLeft[128];
	__shared__ unsigned char sharedResult[128];
	__shared__ unsigned char sharedKey[128];
	__shared__ unsigned char sharedRoundkey[128];
	__shared__ uint64_t result; // setting alias for encryption

	// General shared array. Typically for copying input. Used in the following functions: permuteMatrixCuda, swapLRCuda, leftCircularShiftCuda, rightCircularShiftCuda
	__shared__ unsigned char sharedCopy[128];
	// Special arrays for 'subsituteCuda' function:
	__shared__ uint16_t sharedX[16];
	__shared__ uint16_t sharedY[16];

	uint64_t input;
	uint64_t shiftedKey;
	const int matricesIndices[7] = { 0, 64,120,168,216,248,312 };

	int threadSetIndex = threadIdx.x >> 6; // divide by 64=2^6
	int threadIndex = (threadIdx.x >= 64) ? (threadIdx.x - 64) : threadIdx.x;
	// Initializations
	input = encryptions[(blockIdx.x << 1) + threadSetIndex];
	shiftedKey = keys[(blockIdx.x << 1) + threadSetIndex];
	sharedInput[threadIdx.x] = 0;
	sharedLeft[threadIdx.x] = 0;
	sharedResult[threadIdx.x] = 0;
	sharedKey[threadIdx.x] = 0;
	sharedRoundkey[threadIdx.x] = 0;
	sharedCopy[threadIdx.x] = 0;
	if (threadIndex < 8)
	{
		sharedX[threadIdx.x] = 0;
		sharedY[threadIdx.x] = 0;
	}
	if (threadIndex == 0)
	{
		result = 0;
	}
	__syncthreads();

	// Initial operations 
	// The 64 bits of message,key (uint64_t) are converted into 64 bytes (unsigned char) so that they are easily parallelizable. 
	sharedInput[threadIdx.x] = (input >> threadIndex) & 1;
	sharedKey[threadIdx.x] = (shiftedKey >> threadIndex) & 1;
	__syncthreads();

	// Initial permutation parallelized
	permuteMatrixCuda(sharedInput, sharedCopy, &d_matricesConst[matricesIndices[0]], 64); //initialPermutation(input);
	permuteMatrixCuda(sharedKey, sharedCopy, &d_matricesConst[matricesIndices[1]], 56); // PC1 of key

	__syncthreads();
	for (int i = 0; i < 16; i++)
	{
		// Preserving L,R.
		// preserve right side, R. (Result[63:32] = Input[31:0])
		sharedResult[threadIdx.x] = (threadIndex >= 32) ? sharedInput[threadIdx.x - 32] : 0;

		// preserve left side, L. (Left[31:0] = Input[63:32])
		sharedLeft[threadIdx.x] = (threadIndex < 32) ? sharedInput[threadIdx.x + 32] : 0;
		__syncthreads();

		// Round key
		// Preserve the current shifted key (sharedKey) for the next iteration.
		sharedRoundkey[threadIdx.x] = sharedKey[threadIdx.x];
		__syncthreads();

		// Permutation PC2 of the roundKey
		permuteMatrixCuda(sharedRoundkey, sharedCopy, &d_matricesConst[matricesIndices[2]], 48);//roundKeyPermutation(permutedRoundKey);

		// Shift key to the right for the next iteration
		rightCircularShiftCuda(sharedKey, sharedCopy, d_matricesConst[matricesIndices[6] + 15-i]);

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
		if (threadIndex < 32)
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

	if (threadIndex == 0)
	{
		result = 0;
		for (int i = 0; i < 64; i++)
		{
			result <<= 1;
			result += sharedResult[63 - i + (threadSetIndex << 6)] & 1;
		}
		results[(blockIdx.x << 1) + threadSetIndex] = result;
	}
	__syncthreads();
}

__device__ void permuteMatrixCuda(unsigned char* input, unsigned char* sharedCopy, const unsigned char* P, unsigned int size)
{
	int threadIndex = (threadIdx.x >= 64) ? (threadIdx.x - 64) : threadIdx.x; // divide by 64=2^6
	sharedCopy[threadIdx.x] = input[threadIdx.x];
	__syncthreads();

	// if thread is bigger than the alloted permutation size, make slot equal to 0.
	// Note (threadIdx.x%size) is used in case of a memory violation. But, this precaution is rendered unnecessary by the ternary operator.
	unsigned char bit;
	bit = (threadIndex >= size) ? 0 : (sharedCopy[P[threadIndex%size] - 1] & 1);
	input[threadIdx.x] = bit;
	__syncthreads();
}

__device__ void leftCircularShiftCuda(unsigned char* input, unsigned char* sharedCopy, uint8_t times)
{
	int threadSetIndex = threadIdx.x >> 6;
	int threadIndex = (threadIdx.x >= 64) ? (threadIdx.x - 64) : threadIdx.x;
	
	// copying the key
	sharedCopy[threadIdx.x] = input[threadIdx.x];
	__syncthreads();

	// set offset to determine left and right (L,R) sides of key.
	int offset = 28 * (threadIndex / 28);

	int index = offset + (threadIndex + times) % 28;

	// accounting for edge case with 64 bits.
	// Note shifting is not necessary here, as we do not care about the last 8 bits. 
	index = (index >= 56) ? (offset+index%8) : index;

	// Finally applying the shift
	input[index + (threadSetIndex << 6)] = sharedCopy[threadIdx.x];
	__syncthreads();
}

// Note: maximum of 28 shifts at call of function
__device__ void rightCircularShiftCuda(unsigned char* input, unsigned char* sharedCopy, uint8_t times)
{
	int threadSetIndex = threadIdx.x >> 6;
	int threadIndex = (threadIdx.x >= 64) ? (threadIdx.x - 64) : threadIdx.x;
	// copying the key
	sharedCopy[threadIdx.x] = input[threadIdx.x];
	__syncthreads();

	// set offset to determine left and right (L,R) sides of key.
	int offset = 28 * (threadIndex / 28);

	int index = offset + (threadIndex + 28 - times) % 28;

	// accounting for edge case with 64 bits.
	// Note shifting is not necessary here, as we do not care about the last 8 bits. 
	index = (index >= 56) ? (offset + index % 8) : index;

	// Finally applying the shift
	input[index + (threadSetIndex << 6)] = sharedCopy[threadIdx.x];
	__syncthreads();
}

__device__ void substituteCuda(unsigned char* input, uint16_t* sharedX, uint16_t* sharedY, const unsigned char* d_SBoxesConst)
{
	// 16 inputs (8 x,y pairs) and 8 outputs - 8 extractions from SBox. 
	// 64 threads will allow for 8 simulataneous extractions.

	// Thus, 16 threads will suffice to calculate x,y pairs.
	// 8 threads for each of x and y.

	int threadSetIndex = threadIdx.x >> 6;
	int threadIndex = (threadIdx.x >= 64) ? (threadIdx.x - 64) : threadIdx.x;
	int tid = threadIdx.x;
	int setIndex, index, threadPos;
	uint8_t x = 0;
	uint8_t y = 0;
	unsigned char byte, bit;

	// Y calculation
	// Threads 0 -> 7 work here - First warp
	if (threadIndex < 8)
	{
		// y = b5,b0 then b11,b6, b17,b12 ... b47,b42 i.e. tid*6 + 5, tid*6
		y = (input[(threadSetIndex << 6) + threadIndex * 6 + 5]) << 1;
		y |= input[(threadSetIndex << 6) + threadIndex * 6];
		sharedY[threadIndex + (threadSetIndex << 3)] = y;
	}

	// X calculation 
	// Threads 32 -> 39 work here - Second warp
	if (threadIndex >= 32 && threadIndex < 40)
	{
		// x = b4,b3,b2,b1 then b10,...,b7 i.e. tid * 6 + 4, ..., tid * 6 + 1
		// note we reduced tid by 8, as we work with threads 8->15.
		// i.e. (x's index) * 6 + 4, ..., (x's index) * 6 + 1
		index = threadIndex - 32;
		for (int i = 0; i < 4; i++)
		{
			x |= input[index * 6 + (i + 1) + (threadSetIndex << 6)] << i;
		}
		sharedX[index + (threadSetIndex << 3)] = x;
	}

	// Warps are joined here.
	__syncthreads(); 

	// Extract Sbox output and place it into 'input'
	if (threadIndex < 32)
	{
		setIndex = threadIndex / 4;
		threadPos = threadIndex % 4;
		byte = d_SBoxesConst[(setIndex << 6) + (sharedY[setIndex + (threadSetIndex << 3)] << 4) + (sharedX[setIndex + (threadSetIndex << 3)]) ];
		byte >>= threadPos;
		bit = byte & 1;

		input[tid] = byte & 1;
	}

	// Wipe the last 32 bits of input.
	if (threadIndex >= 32)
	{
		input[tid] = 0;
	}
	__syncthreads();

}

__device__ void swapLRCuda(unsigned char* input, unsigned char* sharedCopy) // Swap left (32 bit) and right (32 bit) parts of the 64 bit input.
{
	sharedCopy[threadIdx.x] = input[threadIdx.x];
	__syncthreads();

	input[threadIdx.x] = sharedCopy[(threadIdx.x + 32)%64 + ((threadIdx.x>=64) << 6)];
	__syncthreads();
}