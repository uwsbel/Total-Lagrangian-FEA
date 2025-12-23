/**
 * FEAT10 Bunny Newton Test
 *
 * Author: Json Zhou
 * Email:  zzhou292@wisc.edu
 *
 * This simulation loads a FEAT10 bunny mesh, clamps nodes near the base,
 * applies strong downward loads on nodes near the ears, and advances the
 * configuration with the synchronized Newton solver. It is used to stress
 * test FEAT10 internal force assembly, constraint handling, Newton
 * convergence, and VTK output under large deformations.
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "../../lib_utils/quadrature_utils.h"
#include "../lib_src/elements/FEAT10Data.cuh"
#include "../lib_src/solvers/SyncedNewton.cuh"
#include "../lib_utils/cpu_utils.h"

const double E    = 3.0e8;  // Pa  (~0.3 GPa, between 0.7 GPa and 0.13 GPa)
const double nu   = 0.40;   // polymers tend to be higher than metals
const double rho0 = 920.0;  // kg/m^3, typical polyethylene density

enum MATERIAL_MODEL { MAT_SVK, MAT_MOONEY_RIVLIN };

int main() {
  // Read mesh data
  Eigen::MatrixXd nodes;
  Eigen::MatrixXi elements;

  int n_nodes = ANCFCPUUtils::FEAT10_read_nodes(
      "data/meshes/T10/bunny_ascii_26.1.node", nodes);
  int n_elems = ANCFCPUUtils::FEAT10_read_elements(
      "data/meshes/T10/bunny_ascii_26.1.ele", elements);

  MATERIAL_MODEL material = MAT_SVK;

  GPU_FEAT10_Data gpu_t10_data(n_elems, n_nodes);

  gpu_t10_data.Initialize();

  // Extract coordinate vectors from nodes matrix
  Eigen::VectorXd h_x12(n_nodes), h_y12(n_nodes), h_z12(n_nodes);
  for (int i = 0; i < n_nodes; i++) {
    h_x12(i) = nodes(i, 0);  // X coordinates
    h_y12(i) = nodes(i, 1);  // Y coordinates
    h_z12(i) = nodes(i, 2);  // Z coordinates
  }

  // Find all nodes with z < -4
  std::vector<int> fixed_node_indices;
  for (int i = 0; i < h_z12.size(); ++i) {
    if (h_z12(i) < -4.0) {  // Fix nodes with z coordinate less than -4
      fixed_node_indices.push_back(i);
    }
  }

  // Convert to Eigen::VectorXi
  Eigen::VectorXi h_fixed_nodes(fixed_node_indices.size());
  for (size_t i = 0; i < fixed_node_indices.size(); ++i) {
    h_fixed_nodes(i) = fixed_node_indices[i];
  }

  // Set fixed nodes
  gpu_t10_data.SetNodalFixed(h_fixed_nodes);

  // set external force: -1000N in z direction for all nodes above z=4
  Eigen::VectorXd h_f_ext(gpu_t10_data.get_n_coef() * 3);
  h_f_ext.setZero();

  for (int i = 0; i < h_z12.size(); ++i) {
    if (h_z12(i) > 4.0) {
      h_f_ext(3 * i + 2) = -35000.0;  // z direction
    }
  }
  gpu_t10_data.SetExternalForce(h_f_ext);

  // Get quadrature data from quadrature_utils.h
  const Eigen::VectorXd& tet5pt_x_host       = Quadrature::tet5pt_x;
  const Eigen::VectorXd& tet5pt_y_host       = Quadrature::tet5pt_y;
  const Eigen::VectorXd& tet5pt_z_host       = Quadrature::tet5pt_z;
  const Eigen::VectorXd& tet5pt_weights_host = Quadrature::tet5pt_weights;

  // Call Setup with all required parameters
  gpu_t10_data.Setup(tet5pt_x_host, tet5pt_y_host, tet5pt_z_host,
                     tet5pt_weights_host, h_x12, h_y12, h_z12, elements);

  gpu_t10_data.SetDensity(rho0);
  gpu_t10_data.SetDamping(0.0, 0.0);

  if (material == MAT_SVK) {
    gpu_t10_data.SetSVK(E, nu);
  } else {
    const double mu    = E / (2.0 * (1.0 + nu));
    const double K     = E / (3.0 * (1.0 - 2.0 * nu));
    const double kappa = 1.5 * K;
    const double mu10  = 0.30 * mu;
    const double mu01  = 0.20 * mu;
    gpu_t10_data.SetMooneyRivlin(mu10, mu01, kappa);
  }

  gpu_t10_data.CalcDnDuPre();

  // 2. Retrieve results
  std::vector<std::vector<Eigen::MatrixXd>> ref_grads;
  gpu_t10_data.RetrieveDnDuPreToCPU(ref_grads);

  std::vector<std::vector<double>> detJ;
  gpu_t10_data.RetrieveDetJToCPU(detJ);

  gpu_t10_data.CalcMassMatrix();

  gpu_t10_data.CalcConstraintData();

  gpu_t10_data.ConvertToCSR_ConstraintJacT();

  gpu_t10_data.BuildConstraintJacobianCSR();


  // calculate p
  gpu_t10_data.CalcP();

  // retrieve p
  std::vector<std::vector<Eigen::MatrixXd>> p_from_F;
  gpu_t10_data.RetrievePFromFToCPU(p_from_F);

  // calculate internal force
  gpu_t10_data.CalcInternalForce();

  // retrieve internal force
  Eigen::VectorXd f_int;
  gpu_t10_data.RetrieveInternalForceToCPU(f_int);
  SyncedNewtonParams params = {1e-4, 1e-4, 1e-4, 1e14, 5, 10, 1e-3};
  SyncedNewtonSolver solver(&gpu_t10_data, gpu_t10_data.get_n_constraint());
  solver.Setup();
  solver.SetParameters(&params);

  solver.AnalyzeHessianSparsity();
  solver.SetFixedSparsityPattern(true);  // Enable analysis reuse for fixed structure

  // Prepare CSV file for node positions
  std::ofstream csv_file("output/bunny_node_positions.csv");
  if (csv_file.is_open()) {
    csv_file << "step,node_id,x,y,z" << std::endl;
    csv_file << std::fixed << std::setprecision(10);
  }

  for (int i = 0; i < 8000; i++) {
    // Reset external force to zero after 1000 steps
    if (i == 1000) {
      Eigen::VectorXd h_zero(gpu_t10_data.get_n_coef() * 3);
      h_zero.setZero();
      gpu_t10_data.SetExternalForce(h_zero);
    }

    solver.Solve();

    // Save node positions at every step to CSV
    if (csv_file.is_open()) {
      Eigen::VectorXd x12, y12, z12;
      gpu_t10_data.RetrievePositionToCPU(x12, y12, z12);
      for (int node_idx = 0; node_idx < x12.size(); ++node_idx) {
        csv_file << i << "," << node_idx << "," << x12(node_idx) << ","
                 << y12(node_idx) << "," << z12(node_idx) << "\n";
      }
    }
  }

  if (csv_file.is_open()) {
    csv_file.close();
  }

  gpu_t10_data.Destroy();

  return 0;
}
