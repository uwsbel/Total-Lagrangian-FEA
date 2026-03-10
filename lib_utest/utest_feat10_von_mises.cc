/**
 * FEAT10 Von Mises Stress Unit Tests
 *
 * Validates the ComputeVonMises() GPU kernel against analytical solutions
 * and independent CPU recomputation.
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>

#include "../../lib_utils/quadrature_utils.h"
#include "../lib_src/elements/FEAT10Data.cuh"
#include "../lib_utils/cpu_utils.h"

namespace {

// Independent CPU computation of von Mises stress from F and P.
double cpu_von_mises(const Eigen::MatrixXd& F, const Eigen::MatrixXd& P) {
  double J = F.determinant();
  if (std::abs(J) < 1e-12)
    return 0.0;
  Eigen::MatrixXd sigma = (1.0 / J) * P * F.transpose();
  double s00 = sigma(0, 0), s11 = sigma(1, 1), s22 = sigma(2, 2);
  double s01 = sigma(0, 1), s02 = sigma(0, 2), s12 = sigma(1, 2);
  double vm2 = 0.5 * ((s00 - s11) * (s00 - s11) + (s11 - s22) * (s11 - s22) +
                      (s22 - s00) * (s22 - s00)) +
               3.0 * (s01 * s01 + s02 * s02 + s12 * s12);
  return std::sqrt(std::abs(vm2));
}

class VonMisesTest : public ::testing::Test {
 protected:
  GPU_FEAT10_Data* gpu_data = nullptr;
  int n_nodes               = 0;
  int n_elems               = 0;
  Eigen::VectorXd h_x12, h_y12, h_z12;

  void SetUp() override {
    Eigen::MatrixXd nodes;
    Eigen::MatrixXi elements;
    n_nodes = ANCFCPUUtils::FEAT10_read_nodes(
        "data/meshes/T10/beam_3x2x1.1.node", nodes);
    n_elems = ANCFCPUUtils::FEAT10_read_elements(
        "data/meshes/T10/beam_3x2x1.1.ele", elements);

    h_x12.resize(n_nodes);
    h_y12.resize(n_nodes);
    h_z12.resize(n_nodes);
    for (int i = 0; i < n_nodes; i++) {
      h_x12(i) = nodes(i, 0);
      h_y12(i) = nodes(i, 1);
      h_z12(i) = nodes(i, 2);
    }

    gpu_data = new GPU_FEAT10_Data(n_elems, n_nodes);
    gpu_data->Initialize();
    gpu_data->Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y,
                    Quadrature::tet5pt_z, Quadrature::tet5pt_weights, h_x12,
                    h_y12, h_z12, elements);

    SolidMaterialProperties mat =
        SolidMaterialProperties::SVK(7e8, 0.33, 2700.0, 0.0, 0.0);
    gpu_data->ApplyMaterial(mat);
    gpu_data->CalcDnDuPre();
  }

  void TearDown() override {
    if (gpu_data) {
      gpu_data->Destroy();
      delete gpu_data;
    }
  }
};

TEST_F(VonMisesTest, VonMisesZeroAtRest) {
  gpu_data->CalcP();
  gpu_data->ComputeVonMises();

  Eigen::VectorXd vm;
  gpu_data->RetrieveVonMisesToCPU(vm);

  for (int i = 0; i < n_elems; i++) {
    EXPECT_NEAR(vm(i), 0.0, 1e-6) << "Element " << i;
  }
}

TEST_F(VonMisesTest, VonMisesUniformTension) {
  Eigen::VectorXd x_new = h_x12 * 1.01;
  gpu_data->UpdatePositions(x_new, h_y12, h_z12);
  gpu_data->CalcP();
  gpu_data->ComputeVonMises();

  // Analytical SVK von Mises for F = diag(1.01, 1, 1)
  double E_val  = 7e8;
  double nu     = 0.33;
  double lambda = E_val * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
  double mu     = E_val / (2.0 * (1.0 + nu));
  double f      = 1.01;
  double e_val  = 0.5 * (f * f - 1.0);  // Green strain E_11
  // Cauchy: sigma = diag(f*(lambda+2mu)*e, lambda*e/f, lambda*e/f)
  double s00           = f * (lambda + 2.0 * mu) * e_val;
  double s11           = lambda * e_val / f;
  double vm_analytical = std::abs(s00 - s11);

  Eigen::VectorXd vm;
  gpu_data->RetrieveVonMisesToCPU(vm);

  for (int i = 0; i < n_elems; i++) {
    EXPECT_NEAR(vm(i), vm_analytical, 1e-6) << "Element " << i;
  }
}

TEST_F(VonMisesTest, VonMisesHydrostatic) {
  double alpha = 1.05;
  gpu_data->UpdatePositions(h_x12 * alpha, h_y12 * alpha, h_z12 * alpha);
  gpu_data->CalcP();
  gpu_data->ComputeVonMises();

  Eigen::VectorXd vm;
  gpu_data->RetrieveVonMisesToCPU(vm);

  for (int i = 0; i < n_elems; i++) {
    EXPECT_NEAR(vm(i), 0.0, 1e-5) << "Element " << i;
  }
}

TEST_F(VonMisesTest, VonMisesCPURecomputation) {
  Eigen::VectorXd x_new = h_x12 + 0.01 * h_y12;
  gpu_data->UpdatePositions(x_new, h_y12, h_z12);
  gpu_data->CalcP();
  gpu_data->ComputeVonMises();

  Eigen::VectorXd vm_gpu;
  gpu_data->RetrieveVonMisesToCPU(vm_gpu);

  std::vector<std::vector<Eigen::MatrixXd>> F_all, P_all;
  gpu_data->RetrieveDeformationGradientToCPU(F_all);
  gpu_data->RetrievePFromFToCPU(P_all);

  for (int elem = 0; elem < n_elems; elem++) {
    double vm_sum = 0.0;
    for (int qp = 0; qp < Quadrature::N_QP_T10_5; qp++) {
      vm_sum += cpu_von_mises(F_all[elem][qp], P_all[elem][qp]);
    }
    double vm_cpu = vm_sum / static_cast<double>(Quadrature::N_QP_T10_5);
    EXPECT_NEAR(vm_gpu(elem), vm_cpu, 1e-6) << "Element " << elem;
  }
}

}  // namespace
