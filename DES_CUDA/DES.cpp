#include <iostream>
#include <chrono>
#include <random>
#include <time.h>
#include "DES.h"
/////////////////////////////////////////////////////////////////////////////////////
// permutation - substitution functions
/////////////////////////////////////////////////////////////////////////////////////
void permuteMatrix(uint64_t& input, const unsigned char* P, const unsigned int size)
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
void initialPermutation(uint64_t& input)
{
	permuteMatrix(input, IP, 64);
}
void substitute(uint64_t& input)
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
		temp = SBoxes[i][y * 16 + x];
		result += temp << (4 * i);

		// next bits
		input >>= 6;
	}
	input = result;
}
void mixPermutation(uint64_t& input)
{
	permuteMatrix(input, PMatrix, 32);
}
void reverseInitialPermutation(uint64_t& input)
{
	permuteMatrix(input, IPInverse, 64);
}
void swapLR(uint64_t& input) // Swap left (32 bit) and right (32 bit) parts of the 64 bit input.
{
	uint64_t temp = input;
	// containing left side 
	temp >>= 32;

	// right side moved to left
	input <<= 32;

	// left side moved to right
	input += temp;
}
/////////////////////////////////////////////////////////////////////////////////////
// key generation functions
/////////////////////////////////////////////////////////////////////////////////////
void generateKey(uint64_t& key)
{
	// 64 bits
	key = ((uint64_t)rand()) << 32 | rand();
}
void leftCircularShift(uint32_t& input, uint8_t times)
{
	uint32_t mask28thBit = 1 << 27; // 28th bit
	uint32_t mask28Bits = 268435455; // covers first 28 bits

	uint8_t bit;
	for (int i = 0; i < times; i++)
	{
		bit = (input & mask28thBit)>>27;
		input <<= 1;
		input += bit;
	}
	input = input & mask28Bits;

}
void rightCircularShift(uint32_t& input, uint8_t times)
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
void generateShiftedKey(const int& index, uint64_t& roundKey)
{
	uint32_t left, right;
	uint64_t mask28Bits = 268435455; // covers first 28 bits

	// getting left and right sides
	right = roundKey & mask28Bits;
	mask28Bits <<= 28;
	mask28Bits = roundKey & mask28Bits;
	left = mask28Bits >> 28;

	// circular shifts
	leftCircularShift(left, LCS[index]);
	leftCircularShift(right, LCS[index]);

	// copying left and right shifted keys to roundKey.
	roundKey = left;
	roundKey <<= 28;
	roundKey += right;
}
void generateReverseShiftedKey(const int& index, uint64_t& roundKey)
{
	uint32_t left, right;
	uint64_t mask28Bits = 268435455; // covers first 28 bits

	// getting left and right sides
	right = roundKey & mask28Bits;
	mask28Bits <<= 28;
	mask28Bits = roundKey & mask28Bits;
	left = mask28Bits >> 28;

	// circular shifts
	rightCircularShift(left, LCS[15-index]);
	rightCircularShift(right, LCS[15-index]);

	// copying left and right shifted keys to roundKey.
	roundKey = left;
	roundKey <<= 28;
	roundKey += right;
}
// Preemptively shifting all keys using LCS matrix.
void fullShiftLCS(uint64_t& roundKey)
{
	uint32_t left, right;
	uint64_t mask28Bits = 268435455; // covers first 28 bits

	// getting left and right sides
	right = roundKey & mask28Bits;
	mask28Bits <<= 28;
	mask28Bits = roundKey & mask28Bits;
	left = mask28Bits >> 28;

	uint32_t numShifts = 0;
	for (int i = 0; i < 16; i++)
	{
		numShifts += LCS[i];
	}
	// circular shifts
	leftCircularShift(left,numShifts);
	leftCircularShift(right, numShifts);

	// copying left and right shifted keys to roundKey.
	roundKey = left;
	roundKey <<= 28;
	roundKey += right;
}
void roundKeyPermutation(uint64_t& roundKey)
{
	permuteMatrix(roundKey, PC2, 48);
}
void expandPermutation(uint64_t& input)
{
	permuteMatrix(input, E, 48);
}

/////////////////////////////////////////////////////////////////////////////////////
// Debug functions
/////////////////////////////////////////////////////////////////////////////////////
void printMatrix(uint64_t matrix, int y, int x)
{
	bool bit;
	bool mask = 1;
	for (int i = 0; i < y; i++)
	{
		for (int j = 0; j < x; j++)
		{

			bit = matrix & mask;
			std::cout << bit << ",";
			matrix >>= 1;
		}
		std::cout << "\n";
	}
	std::cout << "Matrix printed.\n";
}

/////////////////////////////////////////////////////////////////////////////////////
// Essential functions
/////////////////////////////////////////////////////////////////////////////////////
void InitKeyDES(uint64_t& key)
{
	generateKey(key);
	permuteMatrix(key, PC1, 56);
}
void EncryptDES(const uint64_t& plaintext, const uint64_t& key, uint64_t& encryption)
{
	uint64_t& result = encryption; // setting alias for decryption

	uint64_t input = plaintext;
	uint64_t shiftedKey = key;
	uint64_t permutedRoundKey;
	uint64_t left; // last 32 bits of plaintext/input to algorithm are preserved in this variable 

	// Initial operations 
	initialPermutation(input);
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
		roundKeyPermutation(permutedRoundKey);

		// Expansion permutation
		expandPermutation(input); // 48 bits

		// XOR with permuted round key
		input ^= permutedRoundKey;

		// Substitution S-boxes
		substitute(input); // 32 bits
		
		// "P-matrix" permutation i.e. mix/shuffle
		mixPermutation(input); 

		// XOR with preserved left side
		result += left^input; // Result[31:0] = L XOR f[31:0];

		// End of loop
		input = result;
	}

	swapLR(result);
	reverseInitialPermutation(result);
}

void EncryptDESDebug(const uint64_t& plaintext, const uint64_t& key, uint64_t& encryption, uint64_t* debug)
{
	uint64_t& result = encryption; // setting alias for decryption

	uint64_t input = plaintext;
	uint64_t shiftedKey = key;
	uint64_t permutedRoundKey;
	uint64_t left; // last 32 bits of plaintext/input to algorithm are preserved in this variable 

	// Initial operations 
	initialPermutation(input);
	debug[0] = input;
	debug[1] = shiftedKey;
	permuteMatrix(shiftedKey, PC1, 56); // PC1 of key
	debug[2] = shiftedKey;
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
		debug[3] = shiftedKey;
		permutedRoundKey = shiftedKey;
		roundKeyPermutation(permutedRoundKey);
		debug[4] = permutedRoundKey;

		// Expansion permutation
		expandPermutation(input); // 48 bits
		debug[5] = input;

		// XOR with permuted round key
		input ^= permutedRoundKey;
		debug[6] = input;

		// Substitution S-boxes
		substitute(input); // 32 bits
		debug[7] = input;

		// "P-matrix" permutation i.e. mix/shuffle
		mixPermutation(input);
		debug[8] = input;

		// XOR with preserved left side
		result += left ^ input; // Result[31:0] = L XOR f[31:0];

		// End of loop
		input = result;
	}

	swapLR(result);
	debug[9] = result;

	reverseInitialPermutation(result);
	debug[10] = result;

}
void DecryptDES(const uint64_t& encryption, const uint64_t& key, uint64_t& decryption)
{
	uint64_t input = encryption;
	uint64_t shiftedKey = key;
	uint64_t permutedRoundKey;
	uint64_t& result = decryption;
	uint64_t left;

	// Initial operations 
	permuteMatrix(shiftedKey, PC1, 56); // PC1 of key
	fullShiftLCS(shiftedKey);
	initialPermutation(input);

	for (int i = 0; i < 16; i++)
	{
		// Result[63:32] = Input[31:0];
		result = input;
		result <<= 32;
		// preserve left side
		left = input >> 32;

		// round key
		permutedRoundKey = shiftedKey;
		roundKeyPermutation(permutedRoundKey);
		generateReverseShiftedKey(i, shiftedKey);

		// Expansion
		expandPermutation(input); // 48 bits
		// XOR with key
		input ^= permutedRoundKey;

		// Substitution S-boxes
		substitute(input); // 32 bits

		// "P-matrix" permutation i.e. mix/shuffle
		mixPermutation(input);

		// XOR with preserved left side
		result += left ^ input; // Result[31:0] = L XOR f[31:0];

		// End of loop
		input = result;
	}

	swapLR(result);
	reverseInitialPermutation(result);
}

/////////////////////////////////////////////////////////////////////////////////////
// Testing functions
/////////////////////////////////////////////////////////////////////////////////////
void foo()
{
}



