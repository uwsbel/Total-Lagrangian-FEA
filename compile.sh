nvcc test.cu GPUMemoryManager.cu -o test \
  -I. -I/usr/include/eigen3 -O3 \
  -gencode arch=compute_61,code=sm_61 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_86,code=compute_86   # PTX fallback (forward-compat)
