load("@rules_cuda//cuda:defs.bzl", "cuda_library")
# ========================================

# ========================================
# cuda library section
cuda_library(
    name = "gpu_memory_manager",
    srcs = ["lib_src/GPUMemoryManager.cu"],
    hdrs = ["lib_src/GPUMemoryManager.cuh"],
    copts = ["--std=c++17"],
    deps = ["@eigen//:eigen"],
    visibility = ["//visibility:public"],
)
# ========================================

# ========================================
# utility library section
cc_library(
    name = "cpu_utils",
    srcs = ["lib_utils/cpu_utils.cc"],
    hdrs = ["lib_utils/cpu_utils.h"],
    copts = ["--std=c++17"],
    deps = [
        "@eigen//:eigen",
    ],
    visibility = ["//visibility:public"],
)
# ========================================

# ========================================
# cc binary section
cc_binary(
    name = "test",
    srcs = ["lib_bin/test.cc"],
    copts = ["--std=c++17"],
    deps = [
        ":gpu_memory_manager",
        ":cpu_utils",
        "@eigen//:eigen",
    ],
)
# ========================================

# ========================================
# unit test section
cc_test(
    name = "utest_3243",
    srcs = ["lib_utest/utest_3243.cc"],
    copts = ["--std=c++17"],
    deps = [
        ":gpu_memory_manager",
        "@eigen//:eigen",
        ":cpu_utils",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "utest_utils",
    srcs = ["lib_utest/utest_utils.cc"],
    copts = ["--std=c++17"],
    deps = [
        "@eigen//:eigen",
        ":cpu_utils",
        "@googletest//:gtest_main",
    ],
)
# ========================================