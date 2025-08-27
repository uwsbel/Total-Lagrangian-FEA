load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name  = "gpumm",
    srcs  = ["test.cu", "GPUMemoryManager.cu"],
    hdrs  = ["GPUMemoryManager.cuh"],
    copts = ["--std=c++17"],
    # Only Eigen here; do NOT reference @rules_cuda//cuda:cuda_runtime
    deps  = ["@eigen//:eigen"],
)

cc_binary(
    name = "gpumm_test",
    deps = [":gpumm"],
)
