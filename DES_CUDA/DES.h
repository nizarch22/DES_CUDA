#pragma once
#include "DES_Matrices_NIST.h"
void EncryptDES(const uint64_t& plaintext, const uint64_t& key, uint64_t& encryption);
void DecryptDES(const uint64_t& encryption, const uint64_t& key, uint64_t& decryption);
void InitKeyDES(uint64_t& key);
// Delete this
void permuteMatrix(uint64_t& input, const unsigned char* P, const unsigned int size);
void substitute(uint64_t& input);
void generateShiftedKey(const int& index, uint64_t& roundKey);
void EncryptDESDebug(const uint64_t& plaintext, const uint64_t& key, uint64_t& encryption, uint64_t* debug);

// debug functions
//void printMatrix(uint64_t matrix, int y, int x);
//void foo();
//void EncryptDESDebug(const uint64_t& plaintext, const uint64_t& key, uint64_t& encryption, uint64_t* debug);

//- including will not be necessary after debugging is done.
// matrix helper functions 
//void permuteMatrix(uint64_t& input, const unsigned char* P, const unsigned int size);
