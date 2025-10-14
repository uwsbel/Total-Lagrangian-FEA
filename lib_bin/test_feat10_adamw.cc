#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <iomanip>
#include <iostream>

#include "../../lib_utils/quadrature_utils.h"
#include "../lib_src/elements/FEAT10Data.cuh"
#include "../lib_src/solvers/SyncedAdamW.cuh"
#include "../lib_utils/cpu_utils.h"

const double E    = 7e8;   // Young's modulus
const double nu   = 0.33;  // Poisson's ratio
const double rho0 = 2700;  // Density

int main() {
  int n_beam = 2;  // this is working
  int n_coef = 10;

  GPU_FEAT10_Data gpu_t10_data(n_beam, n_coef);

  std::cout << "gpu_t10_data created" << std::endl;

  gpu_t10_data.Initialize();

  std::cout << "gpu_t10_data initialized" << std::endl;

  gpu_t10_data.Destroy();

  std::cout << "gpu_t10_data destroyed" << std::endl;

  return 0;
}
