/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    HydroelasticCollisionTypes.cuh
 * Brief:   Defines shared POD types for hydroelastic collision detection and
 *          contact processing (e.g., collision pair identifiers).
 *==============================================================
 *==============================================================*/

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
