/*
 * HRZ lumped mass for T10 elements, standard (FEAT10Data) and FEAT10Opt.
 *
 * For a straight-sided T10 element the HRZ shares are exact: each corner node
 * gets 1/36 of the element mass and each edge node gets 4/27.
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <fstream>

#include "lib_src/elements/FEAT10Data.cuh"
#include "lib_src/elements/FEAT10DataOpt.cuh"
#include "lib_utils/cpu_utils.h"
#include "lib_utils/quadrature_utils.h"

namespace {

// Unit tetrahedron with edge nodes at the edge midpoints, in T10 order
// (edges 01, 12, 02, 03, 13, 23). Volume 1/6.
Eigen::MatrixXd UnitT10Nodes() {
  Eigen::MatrixXd X(10, 3);
  // clang-format off
  X << 0.0, 0.0, 0.0,
       1.0, 0.0, 0.0,
       0.0, 1.0, 0.0,
       0.0, 0.0, 1.0,
       0.5, 0.0, 0.0,
       0.5, 0.5, 0.0,
       0.0, 0.5, 0.0,
       0.0, 0.0, 0.5,
       0.5, 0.0, 0.5,
       0.0, 0.5, 0.5;
  // clang-format on
  return X;
}

Eigen::MatrixXi OneElement() {
  Eigen::MatrixXi elements(1, 10);
  elements << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
  return elements;
}

Eigen::VectorXd StandardLumpedMass(const Eigen::MatrixXd& nodes,
                                   const Eigen::MatrixXi& elements,
                                   double rho) {
  GPU_FEAT10_Data element(elements.rows(), nodes.rows());
  element.Initialize();
  element.Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y,
                Quadrature::tet5pt_z, Quadrature::tet5pt_weights, nodes.col(0),
                nodes.col(1), nodes.col(2), elements);
  element.SetDensity(rho);
  element.CalcDnDuPre();
  element.CalcLumpedMassHRZ();

  Eigen::VectorXd mass;
  element.RetrieveLumpedMassToCPU(mass);
  element.Destroy();
  return mass;
}

// FEAT10Opt stores the inverse lumped mass in float.
Eigen::VectorXd OptLumpedMass(const Eigen::MatrixXd& nodes,
                              const Eigen::MatrixXi& elements, double rho) {
  GPU_FEAT10Opt_Data element;
  element.Initialize(elements.rows(), nodes.rows());
  element.Setup(nodes, elements);
  element.SetDensity(rho);
  element.ComputePrecomputation();
  element.ComputeLumpedMassHRZ();

  Eigen::VectorXf inv_mass;
  element.RetrieveInvLumpedMassToCPU(inv_mass);
  element.Destroy();
  return inv_mass.cast<double>().cwiseInverse();
}

void ExpectExactShares(const Eigen::VectorXd& mass, double m, double tol) {
  ASSERT_EQ(mass.size(), 10);
  for (int i = 0; i < 4; i++) {
    EXPECT_NEAR(mass(i), m / 36.0, tol * m) << "corner node " << i;
  }
  for (int i = 4; i < 10; i++) {
    EXPECT_NEAR(mass(i), m * 4.0 / 27.0, tol * m) << "edge node " << i;
  }
}

}  // namespace

TEST(HRZMass, SingleTet_ExactShares) {
  const double rho = 1200.0;
  const double m   = rho / 6.0;
  ExpectExactShares(StandardLumpedMass(UnitT10Nodes(), OneElement(), rho), m,
                    1e-12);
}

TEST(HRZMass, SingleTet_ExactShares_Opt) {
  const double rho = 1200.0;
  const double m   = rho / 6.0;
  ExpectExactShares(OptLumpedMass(UnitT10Nodes(), OneElement(), rho), m, 1e-6);
}

TEST(HRZMass, BeamMesh) {
  // 3 x 2 x 1 beam: volume 6.
  Eigen::MatrixXd nodes;
  Eigen::MatrixXi elements;
  ASSERT_GT(ANCFCPUUtils::FEAT10_read_nodes("data/meshes/T10/beam_3x2x1.1.node",
                                            nodes),
            0);
  ASSERT_GT(ANCFCPUUtils::FEAT10_read_elements(
                "data/meshes/T10/beam_3x2x1.1.ele", elements),
            0);

  const double rho         = 2700.0;
  const double total       = rho * 6.0;
  Eigen::VectorXd mass     = StandardLumpedMass(nodes, elements, rho);
  Eigen::VectorXd mass_opt = OptLumpedMass(nodes, elements, rho);

  EXPECT_NEAR(mass.sum(), total, 1e-12 * total);
  EXPECT_GT(mass.minCoeff(), 0.0);

  // Saved reference: data/utest/hrz_mass_reference.csv, one mass per node.
  std::ifstream file("data/utest/hrz_mass_reference.csv");
  ASSERT_TRUE(file.is_open());
  for (int i = 0; i < mass.size(); i++) {
    double ref;
    ASSERT_TRUE(file >> ref) << "reference has fewer rows than nodes";
    EXPECT_NEAR(mass(i), ref, 1e-12 * ref) << "node " << i;
  }

  for (int i = 0; i < mass.size(); i++) {
    EXPECT_NEAR(mass_opt(i), mass(i), 1e-6 * mass(i)) << "node " << i;
  }
}
