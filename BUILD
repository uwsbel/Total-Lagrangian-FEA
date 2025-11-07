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

# mesh utilities
cc_library(
    name = "mesh_utils",
    srcs = ["lib_utils/mesh_utils.cc"],
    hdrs = ["lib_utils/mesh_utils.h"],
    copts = ["--std=c++17"],
    deps = ["@eigen//:eigen"],
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

cc_library(
    name = "cuda_utils",
    hdrs = ["lib_utils/cuda_utils.h"],
    copts = ["--std=c++17"],
    visibility = ["//visibility:public"],
)
# ========================================

# ========================================
# element library section
cuda_library(
    name = "ANCF3243Data",
    srcs = ["lib_src/elements/ANCF3243Data.cu"],
    hdrs = ["lib_src/elements/ANCF3243Data.cuh",
            "lib_src/elements/ANCF3243DataFunc.cuh",
            "lib_src/elements/ElementBase.h"],
    copts = ["--std=c++17", "-O3", "--use_fast_math", "--extra-device-vectorization"],
    linkopts = ["-lcusolver","-lcublas", "-lcusparse"],
    deps = [
        ":cpu_utils",
        "@eigen//:eigen",
        ":cuda_utils"
    ],
    visibility = ["//visibility:public"],
)

cuda_library(
    name = "ANCF3443Data",
    srcs = ["lib_src/elements/ANCF3443Data.cu"],
    hdrs = ["lib_src/elements/ANCF3443Data.cuh",
            "lib_src/elements/ANCF3443DataFunc.cuh",
            "lib_src/elements/ElementBase.h"],
    copts = ["--std=c++17", "-O3", "--use_fast_math", "--extra-device-vectorization"],
    linkopts = ["-lcusolver","-lcublas","-lcusparse"],
    deps = [
        ":cpu_utils",
        "@eigen//:eigen",
        ":cuda_utils"
    ],
    visibility = ["//visibility:public"],
)

cuda_library(
    name = "FEAT10Data",
    srcs = ["lib_src/elements/FEAT10Data.cu"],
    hdrs = ["lib_src/elements/FEAT10Data.cuh",
            "lib_src/elements/FEAT10DataFunc.cuh",
            "lib_src/elements/ElementBase.h"],
    copts = ["--std=c++17", "-O3", "--use_fast_math", "--extra-device-vectorization"],
    linkopts = ["-lcusolver","-lcublas", "-lcusparse"],
    deps = [
        ":cpu_utils",
        "@eigen//:eigen",
        ":cuda_utils"
    ],
    visibility = ["//visibility:public"],
)
# ========================================

# ========================================
# solver library section
cuda_library(
    name = "solvers_syncednesterov",
    srcs = [
        "lib_src/solvers/SyncedNesterov.cu",
    ],
    hdrs = [
        "lib_src/solvers/SolverBase.h",
        "lib_src/solvers/SyncedNesterov.cuh"
    ],
    copts = ["--std=c++17", "-O3", "--use_fast_math", "--extra-device-vectorization"],
    deps = [
        ":ANCF3243Data",
        ":ANCF3443Data",
        ":FEAT10Data",
        ":cpu_utils",
        ":cuda_utils",
        "@eigen//:eigen",
    ],
    visibility = ["//visibility:public"],
)

cuda_library(
    name = "solvers_syncedadamw",
    srcs = [
        "lib_src/solvers/SyncedAdamW.cu",
    ],
    hdrs = [
        "lib_src/solvers/SolverBase.h",
        "lib_src/solvers/SyncedAdamW.cuh",
    ],
    copts = ["--std=c++17", "-O3", "--use_fast_math", "--extra-device-vectorization"],
    deps = [
        ":ANCF3243Data",
        ":ANCF3443Data",
        ":FEAT10Data",
        ":cuda_utils",
        ":cpu_utils",
        "@eigen//:eigen",
    ],
    visibility = ["//visibility:public"],
)

cuda_library(
    name = "solvers_syncednewton",
    srcs = [
        "lib_src/solvers/SyncedNewton.cu",
    ],
    hdrs = [
        "lib_src/solvers/SolverBase.h",
        "lib_src/solvers/SyncedNewton.cuh",
    ],
    copts = ["--std=c++17", "-O3", "--use_fast_math", "--extra-device-vectorization"],
    deps = [
        ":ANCF3243Data",
        ":ANCF3443Data",
        ":FEAT10Data",
        ":cuda_utils",
        ":cpu_utils",
        "@eigen//:eigen",
    ],
    visibility = ["//visibility:public"],
)

# ========================================

# ========================================
# cc binary section
cc_binary(
    name = "test_ancf3243_nesterov",
    srcs = ["lib_bin/test_ancf3243_nesterov.cc"],
    copts = ["--std=c++17"],
    linkopts = [
        "-L/usr/local/cuda/lib64",
        "-lcusparse",
        "-lcudart",
    ],
    deps = [
        ":ANCF3243Data",
        ":cpu_utils",
        ":solvers_syncednesterov",
        ":solvers_syncedadamw",
        ":solvers_syncednewton",
        "@eigen//:eigen",
        ":mesh_utils",
    ],
)

cc_binary(
    name = "test_ancf3443_nesterov",
    srcs = ["lib_bin/test_ancf3443_nesterov.cc"],
    copts = ["--std=c++17"],
    linkopts = [
        "-L/usr/local/cuda/lib64",
        "-lcusparse",
        "-lcudart",
    ],
    deps = [
        ":ANCF3443Data",
        ":cpu_utils",
        ":solvers_syncednesterov",
        ":solvers_syncedadamw",
        ":solvers_syncednewton",
        "@eigen//:eigen",
    ],
)

cc_binary(
    name = "test_ancf3243_adamw",
    srcs = ["lib_bin/test_ancf3243_adamw.cc"],
    copts = ["--std=c++17"],
    linkopts = [
        "-L/usr/local/cuda/lib64",
        "-lcusparse",
        "-lcudart",
    ],
    deps = [
        ":ANCF3243Data",
        ":cpu_utils",
        ":mesh_utils",
        ":solvers_syncednesterov",
        ":solvers_syncedadamw",
        ":solvers_syncednewton",
        "@eigen//:eigen",
    ],
)

cc_binary(
    name = "test_ancf3443_adamw",
    srcs = ["lib_bin/test_ancf3443_adamw.cc"],
    copts = ["--std=c++17"],
    linkopts = [
        "-L/usr/local/cuda/lib64",
        "-lcusparse",
        "-lcudart",
    ],
    deps = [
        ":ANCF3443Data",
        ":cpu_utils",
        ":solvers_syncednesterov",
        ":solvers_syncedadamw",
        ":solvers_syncednewton",
        "@eigen//:eigen",
    ],
)

cc_binary(
    name = "test_feat10_adamw",
    srcs = ["lib_bin/test_feat10_adamw.cc"],
    copts = ["--std=c++17"],
    linkopts = [
        "-L/usr/local/cuda/lib64",
        "-lcusparse",
        "-lcudart",
    ],
    deps = [
        ":FEAT10Data",
        ":cpu_utils",
        ":solvers_syncednesterov",
        ":solvers_syncedadamw",
        ":solvers_syncednewton",
        "@eigen//:eigen",
    ],
)

cc_binary(
    name = "test_feat10_resolution_adamw",
    srcs = ["lib_bin/test_feat10_resolution_adamw.cc"],
    copts = ["--std=c++17"],
    linkopts = [
        "-L/usr/local/cuda/lib64",
        "-lcusparse",
        "-lcudart",
    ],
    deps = [
        ":FEAT10Data",
        ":cpu_utils",
        ":solvers_syncednesterov",
        ":solvers_syncedadamw",
        ":solvers_syncednewton",
        "@eigen//:eigen",
    ],
)


cc_binary(
    name = "test_feat10_resolution_adamw_soft",
    srcs = ["lib_bin/test_feat10_resolution_adamw_soft.cc"],
    copts = ["--std=c++17"],
    linkopts = [
        "-L/usr/local/cuda/lib64",
        "-lcusparse",
        "-lcudart",
    ],
    deps = [
        ":FEAT10Data",
        ":cpu_utils",
        ":solvers_syncednesterov",
        ":solvers_syncedadamw",
        ":solvers_syncednewton",
        "@eigen//:eigen",
    ],
)


cc_binary(
    name = "test_feat10_nesterov",
    srcs = ["lib_bin/test_feat10_nesterov.cc"],
    copts = ["--std=c++17"],
    linkopts = [
        "-L/usr/local/cuda/lib64",
        "-lcusparse",
        "-lcudart",
    ],
    deps = [
        ":FEAT10Data",
        ":cpu_utils",
        ":solvers_syncednesterov",
        ":solvers_syncedadamw",
        ":solvers_syncednewton",
        "@eigen//:eigen",
    ],
)

cc_binary(
    name = "test_feat10_bunny_adamw",
    srcs = ["lib_bin/test_feat10_bunny_adamw.cc"],
    copts = ["--std=c++17"],
    linkopts = [
        "-L/usr/local/cuda/lib64",
        "-lcusparse",
        "-lcudart",
    ],
    deps = [
        ":FEAT10Data",
        ":cpu_utils",
        ":solvers_syncednesterov",
        ":solvers_syncedadamw",
        ":solvers_syncednewton",
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
    linkopts = [
        "-L/usr/local/cuda/lib64",
        "-lcusparse",
        "-lcudart",                
    ],
    data = glob([
        "data/utest/**/*",
    ]),
    deps = [
        ":ANCF3243Data",
        ":cpu_utils",
        "@eigen//:eigen",
        "@googletest//:gtest_main",
        ":csv_utils",
        ":mesh_utils",
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

cc_test(
    name = "utest_sparse_mass",
    srcs = ["lib_utest/utest_sparse_mass.cc"],
    copts = ["--std=c++17"],
    linkopts = [
        "-L/usr/local/cuda/lib64",
        "-lcusparse",
        "-lcudart",                
    ],
    data = glob([
        "data/utest/**/*",
        "data/meshes/**",
    ]),
    deps = [
        ":FEAT10Data",
        ":ANCF3443Data",
        ":ANCF3243Data",
        ":cpu_utils",
        "@eigen//:eigen",
        "@googletest//:gtest_main",
        ":csv_utils",
        ":mesh_utils",
    ],
)


cc_test(
    name = "utest_feat10_mesh",
    srcs = ["lib_utest/utest_feat10_mesh.cc"],
    copts = ["--std=c++17"],
    linkopts = [
        "-L/usr/local/cuda/lib64",
        "-lcusparse",
        "-lcudart",                
    ],
    data = glob([
        "data/utest/**/*",
        "data/meshes/**",
    ]),
    deps = [
        ":FEAT10Data",
        ":cpu_utils",
        "@eigen//:eigen",
        "@googletest//:gtest_main",
        ":solvers_syncednesterov",
        ":solvers_syncedadamw",
        ":solvers_syncednewton",
        ":csv_utils",
    ],
)
# ========================================