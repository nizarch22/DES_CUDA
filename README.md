## Description
Runs DES algorithm in CPU and GPU (CUDA) to make a comparison in runtime. The comparison is output as printout.


The program generates pseudorandom messages at a predetermined size, which are then encrypted, and then decrypted and compared to the original messages.\
Throughout this process several measurements of time are made: execution, memory allocation, memory copying; each the CPU and GPU are measured independently.


Take note: the process, total time runtime and the output of the measurements above are dependent on the branch selected in this respository i.e. "modes". Read more about this in the "Available modes" section of this document.
#### Important note
Must have an Nvidia GPU with the CUDA framework installed on the OS running this code from Nvidia's official website.

## How to use the program:
### Running the program
Choose a desired mode from the "Available modes" section below. Check out to the branch corresponding to the desired mode. Go to 'kernel.cu'.\
Finally, run in release mode for more reliable (and faster) results.

### Vary the size of messages (1MB-256MB)
Change "NUM_TESTS_QUICK" in kernel.cu from 1-9 to determine a range mapped to 1MB,2MB,4MB,...,256MB.\
By default, "NUM_TESTS_QUICK" is set to 9 in all modes (or the corresponding size of 256MB).
#### Explanation
The program generates an array of random messages each sized at 1MB-256MB for plaintext.


**Sizing of the random messages** \
Each size being 1MB, 2MB, 4MB, 8MB, 16MB, ..., 256MB. Each size corresponds to a number 1,2,3,...9.\
For example, 1 maps to 1MB, 3 maps to 4MB, 4 maps to 8MB, ..., 9 maps to 256MB.
This number is defined as "NUM_TESTS_QUICK" in 'kernel.cu'.



Note: A higher size will typically showcase a higher speedup as compared from CPU to GPU. This remains true as long as the GPU is not overburdened, but even then it is likely to outperform the CPU. 

### Available modes:
#### GPU only 
1. **Only GPU, only encryption**: (*Branch* - ***EncryptionOnlyChangingSizeGPUOnly**)*\
To get accurate throughput results with the GPU only. 
2. **Only GPU, encryption and decryption**: *(Branch - **ChangingSizeGPUOnly**)*\
verifies encryption-decryption by original message comparison with the GPU only. Confirming that the decryptions are identical to the original messages. 
#### GPU vs CPU comparison 
1. **GPU and CPU, only encryption**: *(Branch - **EncryptionOnlyChangingSize**)*\
Throughput comparison (speedup, execution time) for GPU vs CPU.
2. **GPU and CPU, encryption and decryption**: *(Branch - **ChangingSize**)*\
Throughput, memory allocation, and memory copy comparison (speedup, execution/copy/allocation time) for GPU vs CPU.\
verifies encryption-decryption by original message comparison between GPU and CPU. Confirming both decryptions are identical to the original messages.








