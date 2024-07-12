#pragma once
// defining permutation matrices
extern unsigned char IP[64];
// Permutation Choice (round key permutations)
extern unsigned char PC1[56];
extern unsigned char PC2[48];
extern unsigned char E[48];
// Substitution boxes
extern unsigned char SBoxes[8][64];
// P-matrix
extern unsigned char PMatrix[32];
// Key shifting (LCS)
extern unsigned char LCS[16];
// Inverse permutation at the end
extern unsigned char IPInverse[64];

