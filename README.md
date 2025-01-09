## Description
Runs DES algorithm in CPU and GPU (CUDA) to make a comparison in runtime. The comparison is output as printout.


The program generates pseudorandom messages at a predetermined size, which are then encrypted, and then decrypted and compared to the original messages.\
Throughout this process several measurements of time are made: execution, memory allocation, memory copying; each the CPU and GPU are measured independently.


Take note: the process, total time runtime and the output of the measurements above are dependent on the branch selected in this respository i.e. "modes". Read more about this in the "Available modes" section of this document.
#### Important note
Must have an Nvidia GPU with the CUDA framework installed on the OS running this code from Nvidia's official website.

## How to use the program:
### Running the program
1. Checkout to a branch using the desired mode from the "Available modes" section found below.
2. Go to 'kernel.cu'.
3. Run in release mode for more reliable (and faster) results.

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
1. **Only GPU 64-bit granularity**: (*Branch* - ***GPUOnly_64bitGranularity**)*\
Encrypt/Decrypt one message per thread.
2. **Only GPU 128-bit granularity**: *(Branch - **GPUOnly_128bitGranularity**)*\
Encrypt 2 messages per thread.
3. **GPU 1 bit granularity**: *(Branch - **GPUOnly_1bitGranularity**)*\
Encrypt/Decrypt one message using 64 threads.

#### CPU Only 
4. **CPU - sequential**: *(Branch - **CPUOnly**)*\
Get CPU's throughput.

#### Important note
Other modes are available. However, they need to be configured. At this current time, they may be accessed - but the user access them at their own discretion.



