#!/usr/bin/env bash
set -euo pipefail

# Format only first-party sources; avoid rewriting vendored code and Bazel
# output trees.
find . \
  -path './third_party' -prune -o \
  -path './bazel-*' -prune -o \
  -path './.bazel_*' -prune -o \
  -path './.bazel_root' -prune -o \
  -type f \( -name '*.cc' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \) -print0 \
  | xargs -0 -n 25 -P 8 clang-format -i --style=file
