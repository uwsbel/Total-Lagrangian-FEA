/**
 * FEAT10 Double Pendulum Revolute-Joint Demo
 *
 * Builds a two-link T10 double pendulum from the pendulum beam mesh.
 * The upper link is attached to the world through a revolute joint, and the
 * lower link is attached to the upper link through a second revolute joint.
 * The combined mesh is exported to VTU for visualization.
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../../lib_src/elements/FEAT10Data.cuh"
#include "../../lib_src/solvers/FEAT10ConstraintManager.h"
#include "../../lib_src/solvers/SyncedNewton.cuh"
#include "../../lib_utils/cpu_utils.h"
#include "../../lib_utils/mesh_manager.h"
#include "../../lib_utils/quadrature_utils.h"

namespace {

constexpr double kPi             = 3.14159265358979323846;
constexpr double kGravity        = -9.81;
constexpr double kDt             = 2e-4;
constexpr int kNumStepsDefault   = 40000;
constexpr int kExportIntervalDef = 50;

constexpr double kBeamLength           = 0.5;
constexpr double kUpperHingeLocalZ     = 0.0;
constexpr double kLowerHingeLocalZ     = kBeamLength;
constexpr double kLowerLinkHingeLocalZ = 0.0;
constexpr double kJointOffset          = 0.001;

constexpr double kUpperAngleDeg = 35.0;
constexpr double kLowerAngleDeg = -25.0;

const SolidMaterialProperties kLinkMaterial =
    SolidMaterialProperties::SVK(1.0e7,   // E: pendulum links
                                 0.30,    // nu
                                 1200.0,  // rho0
                                 1.0e4,   // eta_damp
                                 1.0e4    // lambda_damp
    );

double DegToRad(double angle_deg) {
  return angle_deg * kPi / 180.0;
}

Eigen::Vector3d TransformPoint(const Eigen::Matrix4d& transform,
                               const Eigen::Vector3d& point) {
  const Eigen::Vector4d p_h(point.x(), point.y(), point.z(), 1.0);
  const Eigen::Vector4d mapped = transform * p_h;
  return mapped.head<3>();
}

Eigen::Matrix4d MakeBeamTransform(double angle_y,
                                  const Eigen::Vector3d& hinge_point,
                                  double hinge_local_z) {
  const Eigen::Matrix4d rotation = ANCFCPUUtils::rotationY(angle_y);
  const Eigen::Vector3d rotated_hinge =
      TransformPoint(rotation, Eigen::Vector3d(0.0, 0.0, hinge_local_z));
  return ANCFCPUUtils::translation(hinge_point.x() - rotated_hinge.x(),
                                   hinge_point.y() - rotated_hinge.y(),
                                   hinge_point.z() - rotated_hinge.z()) *
         rotation;
}

FEAT10ConstraintManager::ElementRange MakeElementRange(
    const ANCFCPUUtils::MeshInstance& instance) {
  return FEAT10ConstraintManager::ElementRange{
      instance.element_offset, instance.element_offset + instance.num_elements};
}

std::vector<double> ComputeLumpedMass(GPU_FEAT10_Data& data) {
  std::vector<int> offsets;
  std::vector<int> columns;
  std::vector<double> values;
  data.RetrieveMassCSRToCPU(offsets, columns, values);

  std::vector<double> lumped_mass(static_cast<size_t>(data.get_n_coef()), 0.0);
  if (offsets.size() != static_cast<size_t>(data.get_n_coef() + 1)) {
    std::fill(lumped_mass.begin(), lumped_mass.end(), 1.0);
    return lumped_mass;
  }

  for (int row = 0; row < data.get_n_coef(); ++row) {
    double row_sum = 0.0;
    for (int idx = offsets[static_cast<size_t>(row)];
         idx < offsets[static_cast<size_t>(row + 1)]; ++idx) {
      row_sum += values[static_cast<size_t>(idx)];
    }
    lumped_mass[static_cast<size_t>(row)] = row_sum;
  }
  return lumped_mass;
}

void AppendGravityForInstance(Eigen::VectorXd* h_f_ext,
                              const std::vector<double>& lumped_mass,
                              const ANCFCPUUtils::MeshInstance& instance,
                              double gravity) {
  for (int local_node = 0; local_node < instance.num_nodes; ++local_node) {
    const int global_node = instance.node_offset + local_node;
    (*h_f_ext)(3 * global_node + 2) +=
        lumped_mass[static_cast<size_t>(global_node)] * gravity;
  }
}

std::string MakeOutputPath(int frame) {
  std::ostringstream oss;
  oss << "output/engineering_joint/double_pendulum_" << std::setw(6)
      << std::setfill('0') << frame << ".vtu";
  return oss.str();
}

Eigen::Vector3d EvaluateCurrentPointPosition(
    const FEAT10ConstraintManager::ReferencePoint& point,
    const Eigen::MatrixXi& connectivity, const Eigen::VectorXd& x,
    const Eigen::VectorXd& y, const Eigen::VectorXd& z) {
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  for (int local_node = 0; local_node < Quadrature::N_NODE_T10_10;
       ++local_node) {
    const double weight = point.shape(local_node);
    if (weight == 0.0) {
      continue;
    }
    const int node = connectivity(point.element_idx, local_node);
    position += weight * Eigen::Vector3d(x(node), y(node), z(node));
  }
  return position;
}

}  // namespace

int main(int argc, char** argv) {
  int max_steps       = kNumStepsDefault;
  int export_interval = kExportIntervalDef;
  if (argc > 1) {
    const int parsed_steps = std::atoi(argv[1]);
    if (parsed_steps > 0) {
      max_steps = parsed_steps;
    }
  }
  if (argc > 2) {
    const int parsed_interval = std::atoi(argv[2]);
    if (parsed_interval > 0) {
      export_interval = parsed_interval;
    }
  }

  std::cout << "========================================\n";
  std::cout << "FEAT10 Double Pendulum Revolute-Joint Demo\n";
  std::cout << "========================================\n";
  std::cout << "steps=" << max_steps << " export_interval=" << export_interval
            << "\n";

  std::filesystem::create_directories("output/engineering_joint");

  const std::string mesh_prefix =
      "data/meshes/T10/double_pendulum/pendulum_beam.1";

  ANCFCPUUtils::MeshManager mesh_manager;
  const int mesh_upper = mesh_manager.LoadMesh(
      mesh_prefix + ".node", mesh_prefix + ".ele", "upper", kLinkMaterial);
  const int mesh_lower = mesh_manager.LoadMesh(
      mesh_prefix + ".node", mesh_prefix + ".ele", "lower", kLinkMaterial);
  if (mesh_upper < 0 || mesh_lower < 0) {
    std::cerr << "Failed to load pendulum beam meshes from " << mesh_prefix
              << std::endl;
    return 1;
  }

  const auto& inst_upper = mesh_manager.GetMeshInstance(mesh_upper);
  const auto& inst_lower = mesh_manager.GetMeshInstance(mesh_lower);

  const Eigen::Vector3d top_hinge(0.0, 0.0, 0.7);
  const double upper_angle = kPi - DegToRad(kUpperAngleDeg);
  const double lower_angle = kPi - DegToRad(kLowerAngleDeg);

  const Eigen::Matrix4d upper_transform =
      MakeBeamTransform(upper_angle, top_hinge, kUpperHingeLocalZ);
  mesh_manager.TransformMesh(mesh_upper, upper_transform);

  const Eigen::Vector3d lower_hinge = TransformPoint(
      upper_transform, Eigen::Vector3d(0.0, 0.0, kLowerHingeLocalZ));
  const Eigen::Matrix4d lower_transform =
      MakeBeamTransform(lower_angle, lower_hinge, kLowerLinkHingeLocalZ);
  mesh_manager.TransformMesh(mesh_lower, lower_transform);

  const Eigen::MatrixXd& all_nodes = mesh_manager.GetAllNodes();
  const Eigen::MatrixXi& all_elems = mesh_manager.GetAllElements();
  const int n_nodes                = mesh_manager.GetTotalNodes();
  const int n_elems                = mesh_manager.GetTotalElements();

  std::cout << "upper:   " << inst_upper.num_nodes << " nodes, "
            << inst_upper.num_elements << " elements\n";
  std::cout << "lower:   " << inst_lower.num_nodes << " nodes, "
            << inst_lower.num_elements << " elements\n";
  std::cout << "total:   " << n_nodes << " nodes, " << n_elems << " elements\n";
  std::cout << "top hinge:   [" << top_hinge.transpose() << "]\n";
  std::cout << "lower hinge: [" << lower_hinge.transpose() << "]\n";

  GPU_FEAT10_Data gpu_t10_data(n_elems, n_nodes);
  gpu_t10_data.Initialize();

  Eigen::VectorXd h_x12(n_nodes), h_y12(n_nodes), h_z12(n_nodes);
  for (int i = 0; i < n_nodes; ++i) {
    h_x12(i) = all_nodes(i, 0);
    h_y12(i) = all_nodes(i, 1);
    h_z12(i) = all_nodes(i, 2);
  }

  gpu_t10_data.Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y,
                     Quadrature::tet5pt_z, Quadrature::tet5pt_weights, h_x12,
                     h_y12, h_z12, all_elems);
  gpu_t10_data.ApplyMaterialsFromMeshManager(mesh_manager);
  gpu_t10_data.CalcDnDuPre();
  gpu_t10_data.CalcMassMatrix();

  const std::vector<double> lumped_mass = ComputeLumpedMass(gpu_t10_data);
  Eigen::VectorXd h_f_ext               = Eigen::VectorXd::Zero(n_nodes * 3);
  AppendGravityForInstance(&h_f_ext, lumped_mass, inst_upper, kGravity);
  AppendGravityForInstance(&h_f_ext, lumped_mass, inst_lower, kGravity);
  gpu_t10_data.SetExternalForce(h_f_ext);

  FEAT10ConstraintManager constraint_manager(&gpu_t10_data);
  const auto lower_hinge_on_upper = constraint_manager.LocateReferencePoint(
      lower_hinge, MakeElementRange(inst_upper));
  const auto lower_hinge_on_lower = constraint_manager.LocateReferencePoint(
      lower_hinge, MakeElementRange(inst_lower));
  constraint_manager.AddRevoluteJointToWorld(
      MakeElementRange(inst_upper), top_hinge, Eigen::Vector3d::UnitY(),
      kJointOffset, 1.0);
  constraint_manager.AddRevoluteJoint(
      MakeElementRange(inst_upper), MakeElementRange(inst_lower), lower_hinge,
      Eigen::Vector3d::UnitY(), kJointOffset, 1);
  constraint_manager.Finalize();

  gpu_t10_data.CalcConstraintData();
  gpu_t10_data.ConvertToCSR_ConstraintJacT();
  gpu_t10_data.BuildConstraintJacobianCSR();
  gpu_t10_data.CalcP();
  gpu_t10_data.CalcInternalForce();

  SyncedNewtonParams params = {1e-4, 1e-4, 1e-6, 1e9, 8, 10, kDt, false};
  SyncedNewtonSolver solver(&gpu_t10_data, gpu_t10_data.get_n_constraint());
  solver.Setup();
  solver.SetParameters(&params);
  solver.AnalyzeHessianSparsity();
  solver.SetFixedSparsityPattern(true);

  std::cout << "constraints: " << gpu_t10_data.get_n_constraint() << "\n";
  Eigen::VectorXd x_curr, y_curr, z_curr;
  gpu_t10_data.RetrievePositionToCPU(x_curr, y_curr, z_curr);
  const Eigen::Vector3d lower_upper_initial = EvaluateCurrentPointPosition(
      lower_hinge_on_upper, all_elems, x_curr, y_curr, z_curr);
  const Eigen::Vector3d lower_lower_initial = EvaluateCurrentPointPosition(
      lower_hinge_on_lower, all_elems, x_curr, y_curr, z_curr);
  std::cout << "initial lower hinge on upper: ["
            << lower_upper_initial.transpose() << "]\n";
  std::cout << "initial lower hinge on lower: ["
            << lower_lower_initial.transpose() << "]\n";
  std::cout << "initial lower hinge mismatch norm: "
            << (lower_upper_initial - lower_lower_initial).norm() << "\n";
  std::cout << "writing initial frame to " << MakeOutputPath(0) << "\n";
  gpu_t10_data.WriteOutputVTU(MakeOutputPath(0));

  int output_frame = 1;
  for (int step = 1; step <= max_steps; ++step) {
    solver.Solve();
    gpu_t10_data.RetrievePositionToCPU(x_curr, y_curr, z_curr);
    const Eigen::Vector3d lower_upper_current = EvaluateCurrentPointPosition(
        lower_hinge_on_upper, all_elems, x_curr, y_curr, z_curr);
    const Eigen::Vector3d lower_lower_current = EvaluateCurrentPointPosition(
        lower_hinge_on_lower, all_elems, x_curr, y_curr, z_curr);
    std::cout << "step " << step << " lower hinge on upper: ["
              << lower_upper_current.transpose() << "]\n";
    std::cout << "step " << step << " lower hinge on lower: ["
              << lower_lower_current.transpose() << "]\n";
    std::cout << "step " << step << " lower hinge mismatch norm: "
              << (lower_upper_current - lower_lower_current).norm() << "\n";
    std::cout << "step " << step << " lower hinge travel from reference: "
              << (lower_upper_current - lower_hinge).norm() << "\n";
    if (step % export_interval == 0) {
      gpu_t10_data.WriteOutputVTU(MakeOutputPath(output_frame));
      ++output_frame;
    }
  }

  gpu_t10_data.Destroy();
  std::cout << "Done. Output written to output/engineering_joint/\n";
  return 0;
}
