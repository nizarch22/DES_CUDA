## Description
Uses DES algorithm in CPU and GPU (CUDA) to make a comparison in runtime. The comparison is output as printout.
## How to use:
Go to kernel.cu.\
Change numMessages to whatever you want (messages and keys are randomly generated).\
Run in release mode for fastest results.

## An array of random messages (1MB-256MB)
Generates random messages generated from 1MB-256MB for plaintext alone, and outputs comparisons as prinout.\
#### Modes worthy of looking at:
1. **Only GPU, only encryption** - to get accurate performance results from the CPU.
2. **GPU and CPU, only encryption** - to compare accurate performance with CPU.
3. **GPU and CPU, encryption and decryption** - to verify that everything works as it should.
4. **Only GPU, encryption and decryption** - to speed things up but still verify encryption-decryption is correct by comparing to original message.

#### How to choose modes:
Checkout to the branch of the mode you are interested in. Branches are provided below.
##### Branches:
1. Only GPU, only encryption: **EncryptOnlyChangingSizeGPUOnly**
2. GPU and CPU, only encryption: **EncryptOnlyChangingSize**
3. GPU and CPU, encryption and decryption: **ChangingSize**
4. Only GPU, encryption and decryption: **ChangingSizeGPUOnly**

Finally, run in release mode.
#### To vary the size of the array:
Change NUM_TESTS_QUICK in kernel.cu from 1-9 to determine a range mapped to 1MB,2MB,4MB,...,256MB. Set to 256MB by default.\

