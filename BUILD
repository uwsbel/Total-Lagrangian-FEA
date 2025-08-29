load("@rules_cuda//cuda:defs.bzl", "cuda_library")
# ========================================

# ========================================
# utility library section (put this first since gpu_memory_manager depends on it)
cc_library(
    name = "cpu_utils",
    srcs = ["lib_utils/cpu_utils.cc"],
    hdrs = ["lib_utils/cpu_utils.h", "lib_utils/quadrature_utils.h"],
    copts = ["--std=c++17"],
    deps = [
        "@eigen//:eigen",
    ],
    visibility = ["//visibility:public"],
)

# csv utilities
cc_library(
    name = "csv_utils",
    srcs = ["lib_utils/csv_utils.cc"],
    hdrs = ["lib_utils/csv_utils.h"],
    copts = ["--std=c++17"],
    deps = ["@eigen//:eigen"],
    visibility = ["//visibility:public"],
)
# ========================================

# ========================================
# cuda library section
cuda_library(
    name = "gpu_memory_manager",
    srcs = ["lib_src/GPUMemoryManager.cu"],
    hdrs = ["lib_src/GPUMemoryManager.cuh"],
    copts = ["--std=c++17"],
    linkopts = ["-lcusolver","-lcublas"],
    deps = [
        ":cpu_utils",  # Add dependency on cpu_utils for quadrature.h
        "@eigen//:eigen"
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
    data = glob([
        "data/utest/**/*",
    ]),
    deps = [
        ":gpu_memory_manager",
        ":cpu_utils",
        "@eigen//:eigen",
        "@googletest//:gtest_main",
        ":csv_utils",
    ],
)

cc_test(
    name = "utest_utils",
    srcs = ["lib_utest/utest_utils.cc"],
    copts = ["--std=c++17"],
    deps = [
        ":cpu_utils",
        "@eigen//:eigen",
        "@googletest//:gtest_main",
    ],
)
# ========================================