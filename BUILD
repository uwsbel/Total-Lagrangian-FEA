# Compile test.cu
genrule(
    name = "test_obj",
    srcs = ["test.cu", "GPUMemoryManager.cuh"],   # <- added header
    outs = ["test.cu.o"],
    cmd = "/usr/local/cuda/bin/nvcc -c $(location test.cu) -o $@ "
          + "-I. -I/usr/include/eigen3 -O3 "
          + "-gencode arch=compute_61,code=sm_61 "
          + "-gencode arch=compute_75,code=sm_75 "
          + "-gencode arch=compute_86,code=sm_86 "
          + "-gencode arch=compute_86,code=compute_86 "
          + "-Xcompiler -fPIC",
)

# Compile GPUMemoryManager.cu
genrule(
    name = "gpumm_obj",
    srcs = ["GPUMemoryManager.cu", "GPUMemoryManager.cuh"],
    outs = ["GPUMemoryManager.cu.o"],
    cmd = "/usr/local/cuda/bin/nvcc -c $(location GPUMemoryManager.cu) -o $@ "
          + "-I. -I/usr/include/eigen3 -O3 "
          + "-gencode arch=compute_61,code=sm_61 "
          + "-gencode arch=compute_75,code=sm_75 "
          + "-gencode arch=compute_86,code=sm_86 "
          + "-gencode arch=compute_86,code=compute_86 "
          + "-Xcompiler -fPIC",
)

# Link objects
cc_binary(
    name = "gpumm_test",
    srcs = [
        ":test_obj",
        ":gpumm_obj",
    ],
    linkopts = [
        "-L/usr/local/cuda/lib64",   # <-- add this
        "-lcudart",
    ],
)
