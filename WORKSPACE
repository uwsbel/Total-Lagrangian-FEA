load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "eigen",
    urls = ["https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"],
    strip_prefix = "eigen-3.4.0",
    build_file_content = """
cc_library(
    name = "eigen",
    hdrs = glob(["Eigen/**"]),
    includes = ["."],
    visibility = ["//visibility:public"],
)
""",
)