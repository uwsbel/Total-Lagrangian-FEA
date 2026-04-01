/**
 * FEAT10 Double Pendulum Revolute-Joint Vertical Pull Demo
 *
 * Builds a two-link T10 double pendulum from the pendulum beam mesh with both
 * links initially hanging perfectly vertical. A distributed downward pull is
 * applied to the bottom region of the lower beam, and joint reaction forces are
 * exported for validation.
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "../../lib_src/elements/FEAT10Data.cuh"
#include "../../lib_src/solvers/FEAT10ConstraintManager.h"
#include "../../lib_src/solvers/SyncedNewton.cuh"
#include "../../lib_utils/cpu_utils.h"
#include "../../lib_utils/mesh_manager.h"
#include "../../lib_utils/quadrature_utils.h"
#include "double_pendulum_csv_utils.h"

namespace {

constexpr double kPi             = 3.14159265358979323846;
constexpr double kGravity        = -9.81;
constexpr double kDt             = 5e-4;
constexpr int kNumStepsDefault   = 1000;
constexpr int kExportIntervalDef = 50;

constexpr double kBeamLength           = 0.5;
constexpr double kUpperHingeLocalZ     = 0.0;
constexpr double kLowerHingeLocalZ     = kBeamLength;
constexpr double kLowerLinkHingeLocalZ = 0.0;
constexpr double kJointOffset          = 0.001;

constexpr double kVerticalAngleRad  = kPi;
constexpr double kPullForceZ        = 25.0;
constexpr double kPullRegionLength  = 0.05;
constexpr int kPullRampSteps        = 200;
constexpr double kTopHingeHeight    = 1.1;

const SolidMaterialProperties kLinkMaterial =
    SolidMaterialProperties::SVK(1.0e7,   // E: pendulum links
                                 0.30,    // nu
                                 1200.0,  // rho0
                                 1.0e4,   // eta_damp
                                 1.0e4    // lambda_damp
    );

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

double ComputeInstanceMass(const std::vector<double>& lumped_mass,
                           const ANCFCPUUtils::MeshInstance& instance) {
  double total_mass = 0.0;
  for (int local_node = 0; local_node < instance.num_nodes; ++local_node) {
    const int global_node = instance.node_offset + local_node;
    if (global_node < 0 ||
        global_node >= static_cast<int>(lumped_mass.size())) {
      continue;
    }
    total_mass += lumped_mass[static_cast<size_t>(global_node)];
  }
  return total_mass;
}

void AddDistributedForce(Eigen::VectorXd* h_f_ext,
                         const std::vector<int>& nodes,
                         const Eigen::Vector3d& total_force) {
  if (nodes.empty()) {
    return;
  }
  const Eigen::Vector3d force_per_node =
      total_force / static_cast<double>(nodes.size());
  for (int node : nodes) {
    (*h_f_ext)(3 * node + 0) += force_per_node.x();
    (*h_f_ext)(3 * node + 1) += force_per_node.y();
    (*h_f_ext)(3 * node + 2) += force_per_node.z();
  }
}

std::vector<int> SelectLowerTipRegionNodes(
    const Eigen::MatrixXd& all_nodes,
    const ANCFCPUUtils::MeshInstance& lower_instance, double region_length) {
  double min_z = std::numeric_limits<double>::infinity();
  for (int local_node = 0; local_node < lower_instance.num_nodes; ++local_node) {
    const int global_node = lower_instance.node_offset + local_node;
    min_z = std::min(min_z, all_nodes(global_node, 2));
  }

  std::vector<int> selected_nodes;
  selected_nodes.reserve(static_cast<size_t>(lower_instance.num_nodes));
  for (int local_node = 0; local_node < lower_instance.num_nodes; ++local_node) {
    const int global_node = lower_instance.node_offset + local_node;
    if (all_nodes(global_node, 2) <= min_z + region_length) {
      selected_nodes.push_back(global_node);
    }
  }

  if (selected_nodes.empty()) {
    throw std::runtime_error("Lower tip region node selection returned no nodes");
  }
  return selected_nodes;
}

Eigen::Vector3d ComputeNodeCentroid(const std::vector<int>& nodes,
                                    const Eigen::VectorXd& x,
                                    const Eigen::VectorXd& y,
                                    const Eigen::VectorXd& z) {
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  for (int node : nodes) {
    center += Eigen::Vector3d(x(node), y(node), z(node));
  }
  return center / static_cast<double>(nodes.size());
}

std::string MakeOutputPath(int frame) {
  std::ostringstream oss;
  oss << "output/engineering_joint/double_pendulum_revolute_vertical_"
      << std::setw(6) << std::setfill('0') << frame << ".vtu";
  return oss.str();
}

std::string MakeCsvOutputPath() {
  return "output/engineering_joint/double_pendulum_revolute_vertical_metrics.csv";
}

bool TryParsePositiveInt(const std::string& arg, int* value) {
  char* end_ptr   = nullptr;
  const long raw  = std::strtol(arg.c_str(), &end_ptr, 10);
  const bool okay = end_ptr != arg.c_str() && end_ptr != nullptr &&
                    *end_ptr == '\0' && raw > 0 &&
                    raw <= std::numeric_limits<int>::max();
  if (!okay) {
    return false;
  }
  *value = static_cast<int>(raw);
  return true;
}

bool TryParseNonnegativeDouble(const std::string& arg, double* value) {
  char* end_ptr    = nullptr;
  const double raw = std::strtod(arg.c_str(), &end_ptr);
  const bool okay  = end_ptr != arg.c_str() && end_ptr != nullptr &&
                    *end_ptr == '\0' && std::isfinite(raw) && raw >= 0.0;
  if (!okay) {
    return false;
  }
  *value = raw;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  int max_steps       = kNumStepsDefault;
  int export_interval = kExportIntervalDef;
  bool write_csv      = false;
  std::string csv_path;
  double pull_force_z = kPullForceZ;

  int positional_index = 0;
  for (int argi = 1; argi < argc; ++argi) {
    const std::string arg(argv[argi]);
    if (arg == "--csv") {
      write_csv = true;
      continue;
    }

    if (arg.rfind("--csv=", 0) == 0) {
      write_csv = true;
      csv_path  = arg.substr(std::string("--csv=").size());
      continue;
    }

    if (arg.rfind("--pull_force_z=", 0) == 0) {
      const std::string value_text =
          arg.substr(std::string("--pull_force_z=").size());
      if (!TryParseNonnegativeDouble(value_text, &pull_force_z)) {
        std::cerr << "Invalid --pull_force_z: " << value_text << "\n"
                  << "Expected a nonnegative finite magnitude in N."
                  << std::endl;
        return 1;
      }
      continue;
    }

    int parsed_value = 0;
    if (TryParsePositiveInt(arg, &parsed_value)) {
      if (positional_index == 0) {
        max_steps = parsed_value;
      } else if (positional_index == 1) {
        export_interval = parsed_value;
      } else {
        std::cerr << "Unexpected extra positional argument: " << arg << "\n"
                  << "Usage: " << argv[0]
                  << " [max_steps] [export_interval] [--csv[=path]]"
                  << " [--pull_force_z=FZ]"
                  << std::endl;
        return 1;
      }
      ++positional_index;
      continue;
    }

    std::cerr << "Unknown argument: " << arg << "\n"
              << "Usage: " << argv[0]
              << " [max_steps] [export_interval] [--csv[=path]]"
              << " [--pull_force_z=FZ]" << std::endl;
    return 1;
  }

  if (write_csv && csv_path.empty()) {
    csv_path = MakeCsvOutputPath();
  }

  std::cout << "====================================================\n";
  std::cout << "FEAT10 Double Pendulum Revolute Vertical Pull Demo\n";
  std::cout << "====================================================\n";
  std::cout << "steps=" << max_steps << " export_interval=" << export_interval
            << "\n";
  std::cout << "pull_force_z=-" << pull_force_z
            << " pull_region_length=" << kPullRegionLength
            << " pull_ramp_steps=" << kPullRampSteps << "\n";
  if (write_csv) {
    std::cout << "csv=" << csv_path << "\n";
  }

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

  const Eigen::Vector3d top_hinge(0.0, 0.0, kTopHingeHeight);

  const Eigen::Matrix4d upper_transform =
      MakeBeamTransform(kVerticalAngleRad, top_hinge, kUpperHingeLocalZ);
  mesh_manager.TransformMesh(mesh_upper, upper_transform);

  const Eigen::Vector3d lower_hinge = TransformPoint(
      upper_transform, Eigen::Vector3d(0.0, 0.0, kLowerHingeLocalZ));
  const Eigen::Matrix4d lower_transform =
      MakeBeamTransform(kVerticalAngleRad, lower_hinge, kLowerLinkHingeLocalZ);
  mesh_manager.TransformMesh(mesh_lower, lower_transform);

  const Eigen::MatrixXd& all_nodes = mesh_manager.GetAllNodes();
  const Eigen::MatrixXi& all_elems = mesh_manager.GetAllElements();
  const int n_nodes                = mesh_manager.GetTotalNodes();
  const int n_elems                = mesh_manager.GetTotalElements();

  const std::vector<int> lower_tip_region_nodes =
      SelectLowerTipRegionNodes(all_nodes, inst_lower, kPullRegionLength);

  std::cout << "upper:   " << inst_upper.num_nodes << " nodes, "
            << inst_upper.num_elements << " elements\n";
  std::cout << "lower:   " << inst_lower.num_nodes << " nodes, "
            << inst_lower.num_elements << " elements\n";
  std::cout << "total:   " << n_nodes << " nodes, " << n_elems << " elements\n";
  std::cout << "top hinge:   [" << top_hinge.transpose() << "]\n";
  std::cout << "lower hinge: [" << lower_hinge.transpose() << "]\n";
  std::cout << "lower tip pull nodes: " << lower_tip_region_nodes.size()
            << "\n";

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
  const double upper_mass               = ComputeInstanceMass(lumped_mass, inst_upper);
  const double lower_mass               = ComputeInstanceMass(lumped_mass, inst_lower);
  Eigen::VectorXd gravity_f_ext         = Eigen::VectorXd::Zero(n_nodes * 3);
  AppendGravityForInstance(&gravity_f_ext, lumped_mass, inst_upper, kGravity);
  AppendGravityForInstance(&gravity_f_ext, lumped_mass, inst_lower, kGravity);
  gpu_t10_data.SetExternalForce(gravity_f_ext);

  std::cout << "upper beam mass: " << upper_mass << "\n";
  std::cout << "lower beam mass: " << lower_mass << "\n";
  std::cout << "total mass:      " << (upper_mass + lower_mass) << "\n";

  FEAT10ConstraintManager constraint_manager(&gpu_t10_data);
  const auto lower_hinge_on_upper = constraint_manager.LocateReferencePoint(
      lower_hinge, MakeElementRange(inst_upper));
  const auto lower_hinge_on_lower = constraint_manager.LocateReferencePoint(
      lower_hinge, MakeElementRange(inst_lower));
  const auto upper_tip_on_upper = constraint_manager.LocateReferencePoint(
      lower_hinge, MakeElementRange(inst_upper));
  const auto lower_tip_on_lower = constraint_manager.LocateReferencePoint(
      TransformPoint(lower_transform, Eigen::Vector3d(0.0, 0.0, kBeamLength)),
      MakeElementRange(inst_lower));
  constraint_manager.AddRevoluteJointToWorld(
      MakeElementRange(inst_upper), top_hinge, Eigen::Vector3d::UnitY(),
      kJointOffset, 1.0);
  constraint_manager.AddRevoluteJoint(
      MakeElementRange(inst_upper), MakeElementRange(inst_lower), lower_hinge,
      Eigen::Vector3d::UnitY(), kJointOffset, 1.0);
  constraint_manager.Finalize();

  gpu_t10_data.CalcConstraintData();
  gpu_t10_data.ConvertToCSR_ConstraintJacT();
  gpu_t10_data.BuildConstraintJacobianCSR();
  gpu_t10_data.CalcP();
  gpu_t10_data.CalcInternalForce();

  SyncedNewtonParams params = {1e-6, 1e-6, 1e-8, 1e10, 8, 10, kDt, false};
  SyncedNewtonSolver solver(&gpu_t10_data, gpu_t10_data.get_n_constraint());
  solver.Setup();
  solver.SetParameters(&params);
  solver.AnalyzeHessianSparsity();
  solver.SetFixedSparsityPattern(true);

  std::cout << "constraints: " << gpu_t10_data.get_n_constraint() << "\n";
  constexpr int kRowsPerRevoluteJoint = 5;
  constexpr int kPositionRowsPerJoint = 3;

  std::vector<int> position_constraint_rows;
  std::vector<int> orientation_constraint_rows;
  for (int joint_row_offset = 0;
       joint_row_offset < gpu_t10_data.get_n_constraint();
       joint_row_offset += kRowsPerRevoluteJoint) {
    for (int row = 0; row < kPositionRowsPerJoint; ++row) {
      position_constraint_rows.push_back(joint_row_offset + row);
    }
    for (int row = kPositionRowsPerJoint; row < kRowsPerRevoluteJoint; ++row) {
      orientation_constraint_rows.push_back(joint_row_offset + row);
    }
  }

  engineering_joint::DoublePendulumCsvWriter csv_writer;
  Eigen::VectorXd constraint_values;
  Eigen::VectorXd lambda_values =
      Eigen::VectorXd::Zero(gpu_t10_data.get_n_constraint());
  Eigen::VectorXd augmented_dual_values =
      Eigen::VectorXd::Zero(gpu_t10_data.get_n_constraint());
  Eigen::VectorXd velocity_xyz = Eigen::VectorXd::Zero(n_nodes * 3);
  std::vector<int> constraint_j_offsets;
  std::vector<int> constraint_j_columns;
  std::vector<double> constraint_j_values;
  Eigen::VectorXd x_curr, y_curr, z_curr;
  gpu_t10_data.RetrievePositionToCPU(x_curr, y_curr, z_curr);
  gpu_t10_data.CalcConstraintData();
  gpu_t10_data.BuildConstraintJacobianCSR();
  gpu_t10_data.RetrieveConstraintDataToCPU(constraint_values);
  gpu_t10_data.RetrieveConstraintJacobianCSRToCPU(
      constraint_j_offsets, constraint_j_columns, constraint_j_values);

  const Eigen::Vector3d lower_upper_initial =
      engineering_joint::EvaluateCurrentPointPosition(lower_hinge_on_upper,
                                                      all_elems, x_curr, y_curr,
                                                      z_curr);
  const Eigen::Vector3d lower_lower_initial =
      engineering_joint::EvaluateCurrentPointPosition(lower_hinge_on_lower,
                                                      all_elems, x_curr, y_curr,
                                                      z_curr);
  const Eigen::Vector3d pull_region_centroid_initial =
      ComputeNodeCentroid(lower_tip_region_nodes, x_curr, y_curr, z_curr);

  std::cout << "initial lower hinge mismatch norm: "
            << (lower_upper_initial - lower_lower_initial).norm() << "\n";
  std::cout << "initial pull-region centroid: ["
            << pull_region_centroid_initial.transpose() << "]\n";
  std::cout << "writing initial frame to " << MakeOutputPath(0) << "\n";
  gpu_t10_data.WriteOutputVTU(MakeOutputPath(0));

  if (write_csv) {
    const std::filesystem::path csv_parent =
        std::filesystem::path(csv_path).parent_path();
    if (!csv_parent.empty()) {
      std::filesystem::create_directories(csv_parent);
    }
    csv_writer.Open(csv_path);
    std::cout << "writing csv metrics to " << csv_path << "\n";
  }

  int output_frame = 1;
  for (int step = 1; step <= max_steps; ++step) {
    Eigen::VectorXd step_f_ext = gravity_f_ext;
    const double load_scale =
        std::min(1.0, static_cast<double>(step) /
                          static_cast<double>(kPullRampSteps));
    AddDistributedForce(&step_f_ext, lower_tip_region_nodes,
                        Eigen::Vector3d(0.0, 0.0, -pull_force_z * load_scale));
    gpu_t10_data.SetExternalForce(step_f_ext);

    solver.Solve();
    gpu_t10_data.RetrievePositionToCPU(x_curr, y_curr, z_curr);
    gpu_t10_data.CalcConstraintData();
    gpu_t10_data.BuildConstraintJacobianCSR();
    gpu_t10_data.RetrieveConstraintDataToCPU(constraint_values);
    gpu_t10_data.RetrieveConstraintJacobianCSRToCPU(
        constraint_j_offsets, constraint_j_columns, constraint_j_values);
    HANDLE_ERROR(
        cudaMemcpy(lambda_values.data(), solver.GetLambdaGuessDevicePtr(),
                   static_cast<size_t>(lambda_values.size()) * sizeof(double),
                   cudaMemcpyDeviceToHost));
    augmented_dual_values = lambda_values + params.rho * constraint_values;
    HANDLE_ERROR(
        cudaMemcpy(velocity_xyz.data(), solver.GetVelocityGuessDevicePtr(),
                   static_cast<size_t>(velocity_xyz.size()) * sizeof(double),
                   cudaMemcpyDeviceToHost));

    const Eigen::Vector3d lower_upper_current =
        engineering_joint::EvaluateCurrentPointPosition(lower_hinge_on_upper,
                                                        all_elems, x_curr,
                                                        y_curr, z_curr);
    const Eigen::Vector3d lower_lower_current =
        engineering_joint::EvaluateCurrentPointPosition(lower_hinge_on_lower,
                                                        all_elems, x_curr,
                                                        y_curr, z_curr);
    const Eigen::Vector3d upper_tip_current =
        engineering_joint::EvaluateCurrentPointPosition(upper_tip_on_upper,
                                                        all_elems, x_curr,
                                                        y_curr, z_curr);
    const Eigen::Vector3d lower_tip_current =
        engineering_joint::EvaluateCurrentPointPosition(lower_tip_on_lower,
                                                        all_elems, x_curr,
                                                        y_curr, z_curr);
    const Eigen::Vector3d upper_dir_current = upper_tip_current - top_hinge;
    const Eigen::Vector3d lower_dir_current =
        lower_tip_current - lower_lower_current;
    const Eigen::Vector3d lower_hinge_current =
        engineering_joint::AveragePoint(lower_upper_current,
                                        lower_lower_current);
    const Eigen::Vector3d pull_region_centroid_current =
        ComputeNodeCentroid(lower_tip_region_nodes, x_curr, y_curr, z_curr);
    const double lower_hinge_mismatch_norm =
        (lower_upper_current - lower_lower_current).norm();

    if (step == 1 || step % export_interval == 0) {
      std::cout << "step " << step
                << " pull_region_z=" << pull_region_centroid_current.z()
                << " lower_hinge_mismatch=" << lower_hinge_mismatch_norm
                << " load_scale=" << load_scale << "\n";
    }

    if (write_csv) {
      const double potential_energy =
          engineering_joint::ComputePotentialEnergy(lumped_mass, z_curr,
                                                    kGravity);
      const double elastic_strain_energy =
          engineering_joint::ComputeElasticStrainEnergy(gpu_t10_data,
                                                        kLinkMaterial);
      const double kinetic_energy =
          engineering_joint::ComputeKineticEnergy(lumped_mass, velocity_xyz);
      const double total_energy =
          kinetic_energy + potential_energy + elastic_strain_energy;
      const engineering_joint::HingeWrench upper_joint_reaction =
          engineering_joint::ScaleHingeWrench(
              engineering_joint::RecoverSeparatedRevoluteJointWrenchFromCSR(
              n_nodes * 3, constraint_j_offsets, constraint_j_columns,
              constraint_j_values, augmented_dual_values, 0,
              kPositionRowsPerJoint,
              kRowsPerRevoluteJoint - kPositionRowsPerJoint,
              inst_upper.node_offset, inst_upper.num_nodes, x_curr, y_curr,
              z_curr, top_hinge),
              kDt);
      const engineering_joint::HingeWrench lower_joint_reaction =
          engineering_joint::ScaleHingeWrench(
              engineering_joint::RecoverSeparatedRevoluteJointWrenchFromCSR(
              n_nodes * 3, constraint_j_offsets, constraint_j_columns,
              constraint_j_values, augmented_dual_values,
              kRowsPerRevoluteJoint, kPositionRowsPerJoint,
              kRowsPerRevoluteJoint - kPositionRowsPerJoint,
              inst_lower.node_offset, inst_lower.num_nodes, x_curr, y_curr,
              z_curr, lower_hinge_current),
              kDt);

      csv_writer.WriteRow(
          step, step * kDt,
          engineering_joint::SignedAngleAboutAxis(Eigen::Vector3d::UnitZ(),
                                                  upper_dir_current,
                                                  Eigen::Vector3d::UnitY()),
          engineering_joint::SignedAngleAboutAxis(Eigen::Vector3d::UnitZ(),
                                                  lower_dir_current,
                                                  Eigen::Vector3d::UnitY()),
          engineering_joint::SignedAngleAboutAxis(
              upper_dir_current, lower_dir_current, Eigen::Vector3d::UnitY()),
          constraint_values.norm(),
          engineering_joint::ComputeInfinityNorm(constraint_values),
          engineering_joint::ComputeIndexedL2Norm(constraint_values,
                                                  position_constraint_rows),
          engineering_joint::ComputeIndexedInfinityNorm(
              constraint_values, position_constraint_rows),
          engineering_joint::ComputeIndexedL2Norm(constraint_values,
                                                  orientation_constraint_rows),
          engineering_joint::ComputeIndexedInfinityNorm(
              constraint_values, orientation_constraint_rows),
          total_energy, kinetic_energy, potential_energy, elastic_strain_energy,
          upper_joint_reaction, lower_joint_reaction, lower_hinge_mismatch_norm,
          lower_tip_current);
    }

    if (step % export_interval == 0) {
      gpu_t10_data.WriteOutputVTU(MakeOutputPath(output_frame));
      ++output_frame;
    }
  }

  gpu_t10_data.Destroy();
  std::cout << "Done. Output written to output/engineering_joint/\n";
  return 0;
}
