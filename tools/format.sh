find . -type f \( -name '*.cc' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \) \
  -print0 | xargs -0 -n 25 -P 8 clang-format -i --style=file
