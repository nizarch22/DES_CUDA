## Description
Runs DES algorithm in CPU and GPU (CUDA) to make a comparison in runtime. The comparison is output as printout.
## How to use:
Go to kernel.cu.\
Change numMessages to whatever you want (messages and keys are randomly generated).\
Run in release mode for fastest results.

## An array of random messages (1MB-256MB)
Generates random messages generated from 1MB-256MB for plaintext alone, and outputs comparisons as printout.
#### How to use:
Choose a mode by checking out to its branch. Available modes listed below.\
Run in release mode.
#### Available modes:
1. **Only GPU, only encryption**: (*Branch* - ***EncryptOnlyChangingSizeGPUOnly**)* - to get accurate throughput results from the CPU. 
2. **GPU and CPU, only encryption**: *(Branch - **EncryptOnlyChangingSize**)* - to compare accurate throughput with the CPU.
3. **GPU and CPU, encryption and decryption**: *(Branch - **ChangingSize**)* - to verify that everything works as it should.
4. **Only GPU, encryption and decryption**: *(Branch - **EncryptOnlyChangingSizeGPUOnly**)* - to speed up result retrieval for the GPU, and verifies encryption-decryption by original message comparison.


#### To vary the size of the array:
Change NUM_TESTS_QUICK in kernel.cu from 1-9 to determine a range mapped to 1MB,2MB,4MB,...,256MB. Set to 256MB by default.

