nvcc test.cu GPUMemoryManager.cu -o test -I. -I/usr/include/eigen3 -arch=sm_75 -O3
