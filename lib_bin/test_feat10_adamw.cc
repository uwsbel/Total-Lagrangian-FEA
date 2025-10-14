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
  // Read mesh data
  Eigen::MatrixXd nodes;
  Eigen::MatrixXi elements;

  int n_nodes =
      ANCFCPUUtils::FEAT10_read_nodes("data/meshes/T10/cube.1.node", nodes);
  int n_elems = ANCFCPUUtils::FEAT10_read_elements("data/meshes/T10/cube.1.ele",
                                                   elements);

  std::cout << "mesh read nodes: " << n_nodes << std::endl;
  std::cout << "mesh read elements: " << n_elems << std::endl;

  // print nodes and elements matrix
  std::cout << "nodes matrix:" << std::endl;
  std::cout << nodes << std::endl;
  std::cout << "elements matrix:" << std::endl;
  std::cout << elements << std::endl;

  GPU_FEAT10_Data gpu_t10_data(n_elems, n_nodes);

  std::cout << "gpu_t10_data created" << std::endl;

  gpu_t10_data.Initialize();

  std::cout << "gpu_t10_data initialized" << std::endl;

  // Extract coordinate vectors from nodes matrix
  Eigen::VectorXd h_x12(n_nodes), h_y12(n_nodes), h_z12(n_nodes);
  for (int i = 0; i < n_nodes; i++) {
    h_x12(i) = nodes(i, 0);  // X coordinates
    h_y12(i) = nodes(i, 1);  // Y coordinates
    h_z12(i) = nodes(i, 2);  // Z coordinates
  }

  // Get quadrature data from quadrature_utils.h
  const Eigen::VectorXd& tet5pt_x_host       = Quadrature::tet5pt_x;
  const Eigen::VectorXd& tet5pt_y_host       = Quadrature::tet5pt_y;
  const Eigen::VectorXd& tet5pt_z_host       = Quadrature::tet5pt_z;
  const Eigen::VectorXd& tet5pt_weights_host = Quadrature::tet5pt_weights;

  // Call Setup with all required parameters
  gpu_t10_data.Setup(rho0, nu, E,          // Material properties
                     tet5pt_x_host,        // Quadrature points
                     tet5pt_y_host,        // Quadrature points
                     tet5pt_z_host,        // Quadrature points
                     tet5pt_weights_host,  // Quadrature weights
                     h_x12, h_y12, h_z12,  // Node coordinates
                     elements);            // Element connectivity

  std::cout << "gpu_t10_data setup complete" << std::endl;

  gpu_t10_data.CalcDnDuPre();

  std::cout << "gpu_t10_data dndu pre complete" << std::endl;

  // 2. Retrieve results
  std::vector<std::vector<Eigen::MatrixXd>> ref_grads;
  gpu_t10_data.RetrieveDnDuPreToCPU(ref_grads);

  std::cout << "ref_grads:" << std::endl;
  for (size_t i = 0; i < ref_grads.size(); i++) {
    for (size_t j = 0; j < ref_grads[i].size(); j++) {
      std::cout << ref_grads[i][j] << std::endl;
    }
  }
  std::cout << "done retrieving ref_grads" << std::endl;

  gpu_t10_data.Destroy();

  std::cout << "gpu_t10_data destroyed" << std::endl;

  return 0;
}
