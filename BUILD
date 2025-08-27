load("@rules_cuda//cuda:defs.bzl", "cuda_library", "cuda_binary")

# cuda library section
cuda_library(
    name = "gpu_memory_manager",
    srcs = ["lib_src/GPUMemoryManager.cu"],
    hdrs = ["lib_src/GPUMemoryManager.cuh"],
    copts = ["--std=c++17"],
    deps = ["@eigen//:eigen"],
    visibility = ["//visibility:public"],
)

# cuda binary section
cuda_binary(
    name = "test",
    srcs = ["lib_bin/test.cu"],
    copts = ["--std=c++17"],
    deps = [
        ":gpu_memory_manager",
        "@eigen//:eigen",
    ],
)