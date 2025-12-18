#pragma once

#include <cuda_runtime.h>

struct CollisionPair {
  int idA;
  int idB;

  __host__ __device__ CollisionPair() : idA(-1), idB(-1) {}
  __host__ __device__ CollisionPair(int a, int b) : idA(a), idB(b) {}

  __host__ __device__ bool isValid() const {
    return idA >= 0 && idB >= 0;
  }
};
