## Description
Runs DES algorithm in CPU and GPU (CUDA) to make a comparison in runtime. The comparison is output as printout.
## How to use:
Go to kernel.cu.\
Change numMessages to whatever you want (messages and keys are randomly generated).\
Run in release mode for fastest results.
###### Important note
Must have an Nvidia GPU with CUDA framework installed on the OS from Nvidia's official website in order to run this code.
## An array of random messages (1MB-256MB)
Generates random messages generated from 1MB-256MB for plaintext alone, and outputs comparisons as printout.
### How to use:
Choose a mode by checking out to its branch. Available modes listed below.\
Run in release mode.
#### Available modes:
1. **Only GPU, only encryption**: (*Branch* - ***EncryptOnlyChangingSizeGPUOnly**)*\
To get accurate throughput results with the GPU. 
2. **GPU and CPU, only encryption**: *(Branch - **EncryptOnlyChangingSize**)*\
To compare accurate throughput with the GPU and CPU.
3. **GPU and CPU, encryption and decryption**: *(Branch - **ChangingSize**)*\
verifies encryption-decryption by original message comparison with the GPU and CPU.
4. **Only GPU, encryption and decryption**: *(Branch - **EncryptOnlyChangingSizeGPUOnly**)*\
To speed up result retrieval for the GPU, and verifies encryption-decryption by original message comparison with the GPU.


#### To vary the size of the array:
Change NUM_TESTS_QUICK in kernel.cu from 1-9 to determine a range mapped to 1MB,2MB,4MB,...,256MB. Set to 256MB by default.

