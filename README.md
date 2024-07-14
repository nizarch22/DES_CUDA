## Description
Uses DES algorithm in CPU and GPU (CUDA) to make a comparison in runtime. The comparison is output as printout.
## How to use:
Go to kernel.cu.\
Change numMessages to whatever you want (messages and keys are randomly generated).\
Run in release mode for fastest results.

## An array of random messages (1MB-256MB)
Generates random messages generated from 1MB-256MB for plaintext alone, and outputs comparisons as prinout.\
Combinations worthy of looking at:\
1. GPU and CPU to verify that everything works as it should.\
2. Only GPU, to speed things up but still verify encryption-decryption is correct by comparing to original message.\
3. Only encryption to compare accurate performance with CPU.\
4. Only GPU, only encryption to get accurate performance results from the CPU.\

We get these modes by checking out to certain branches.
Once finished, run in release mode.
Note: Just change NUM_TESTS_QUICK from 1-9 to determine a range mapped to 1MB-256MB.

### 1. GPU and CPU
Checkout branch ChangingSize.\
Run in release mode.

### 2. Only GPU
Checkout branch ChangingSizeGPUOnly.\
Run in release mode.

### 3. Only encryption
Checkout branch EncryptOnlyChangingSize.\
Run in release mode.

### 4. Only GPU, only encryption
Checkout branch EncryptOnlyChangingSizeGPUOnly.\
Run in release mode.
