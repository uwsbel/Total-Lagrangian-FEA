#pragma once

#include <cuda_runtime.h>
#if __has_include(<cudss.h>)
#include <cudss.h>
#endif
#include <cusparse.h>

#include <cstdio>
#include <cstdlib>

#ifndef HANDLE_ERROR_MACRO
#define HANDLE_ERROR_MACRO
static inline void HandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
#endif

#ifndef CHECK_CUSPARSE_MACRO
#define CHECK_CUSPARSE_MACRO
#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, \
             cusparseGetErrorString(status), status);                          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
#endif

#if __has_include(<cudss.h>)
#ifndef CHECK_CUDSS_MACRO
#define CHECK_CUDSS_MACRO
#define CUDSS_OK(call)                                              \
  do {                                                              \
    cudssStatus_t status = call;                                    \
    if (status != CUDSS_STATUS_SUCCESS) {                           \
      std::cerr << "cuDSS error at " << __FILE__ << ":" << __LINE__ \
                << std::endl;                                       \
      exit(1);                                                      \
    }                                                               \
  } while (0)
#endif
#endif