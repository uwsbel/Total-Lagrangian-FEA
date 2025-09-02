load("@rules_cuda//cuda:defs.bzl", "cuda_library")
# ========================================

# ========================================
# utility library section (put this first since ANCF3243Data depends on it)
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
# element library section
cuda_library(
    name = "ANCF3243Data",
    srcs = ["lib_src/elements/ANCF3243Data.cu"],
    hdrs = ["lib_src/elements/ANCF3243Data.cuh",
            "lib_src/elements/ANCF3243DataKernels.cuh",],
    copts = ["--std=c++17", "-O3"],
    linkopts = ["-lcusolver","-lcublas"],
    deps = [
        ":cpu_utils",  # Add dependency on cpu_utils for quadrature.h
        "@eigen//:eigen"
    ],
    visibility = ["//visibility:public"],
)
# ========================================

# ========================================
# solver library section
cuda_library(
    name = "solvers",
    srcs = [
        "lib_src/solvers/SyncedNesterov.cu",
    ],
    hdrs = [
        "lib_src/solvers/SolverBase.h",
        "lib_src/solvers/SyncedNesterov.cuh",
    ],
    copts = ["--std=c++17", "-O3"],
    deps = [
        ":ANCF3243Data",
        ":cpu_utils",
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
        ":ANCF3243Data",
        ":cpu_utils",
        ":solvers",
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
        ":ANCF3243Data",
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