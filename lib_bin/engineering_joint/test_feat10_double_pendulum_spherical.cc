/**
 * FEAT10 Double Pendulum Spherical-Joint Demo
 *
 * Builds a two-link T10 double pendulum from the pendulum beam mesh.
 * The upper link is attached to the world through a spherical joint, and the
 * lower link is attached to the upper link through a second spherical joint.
 * The combined mesh is exported to VTU for visualization.
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
constexpr int kNumStepsDefault   = 5000;
constexpr int kExportIntervalDef = 50;

constexpr double kBeamLength           = 0.5;
constexpr double kUpperHingeLocalZ     = 0.0;
constexpr double kLowerHingeLocalZ     = kBeamLength;
constexpr double kLowerLinkHingeLocalZ = 0.0;
constexpr double kUpperAngleDeg        = 35.0;
constexpr double kLowerAngleDeg        = -25.0;
constexpr double kUpperOutOfPlaneDeg   = 18.0;
constexpr double kLowerOutOfPlaneDeg   = -27.0;

constexpr double kYoungsModulus     = 2.0e6;
constexpr double kPoissonsRatio     = 0.30;
constexpr double kDensity           = 1200.0;
constexpr double kEtaDampDefault    = 1.0e4;
constexpr double kLambdaDampDefault = 1.0e4;

double DegToRad(double angle_deg) {
  return angle_deg * kPi / 180.0;
}

Eigen::Vector3d TransformPoint(const Eigen::Matrix4d& transform,
                               const Eigen::Vector3d& point) {
  const Eigen::Vector4d p_h(point.x(), point.y(), point.z(), 1.0);
  const Eigen::Vector4d mapped = transform * p_h;
  return mapped.head<3>();
}

Eigen::Matrix4d MakeBeamTransform(double angle_x, double angle_y,
                                  const Eigen::Vector3d& hinge_point,
                                  double hinge_local_z) {
  // Start from the planar Y rotation used by the revolute demo, then add an
  // X tilt so the spherical-joint motion is initialized in full 3D.
  const Eigen::Matrix4d rotation =
      ANCFCPUUtils::rotationX(angle_x) * ANCFCPUUtils::rotationY(angle_y);
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

std::string FormatDoubleForPath(double value) {
  if (std::abs(value) < 1e-12) {
    return "0";
  }
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(1) << value;
  return oss.str();
}

std::string MakeOutputDirectory(double eta_damp, double lambda_damp) {
  std::ostringstream oss;
  oss << "output/engineering_joint/double_pendulum_spherical_eta_"
      << FormatDoubleForPath(eta_damp) << "_lambda_"
      << FormatDoubleForPath(lambda_damp);
  return oss.str();
}

std::string MakeOutputPath(const std::string& output_dir, int frame) {
  std::ostringstream oss;
  oss << output_dir << "/double_pendulum_spherical_" << std::setw(6)
      << std::setfill('0') << frame << ".vtu";
  return oss.str();
}

std::string MakeCsvOutputPath(const std::string& output_dir) {
  return output_dir + "/double_pendulum_spherical_metrics.csv";
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
  char* end_ptr        = nullptr;
  const double raw     = std::strtod(arg.c_str(), &end_ptr);
  const bool is_finite = std::isfinite(raw);
  const bool okay      = end_ptr != arg.c_str() && end_ptr != nullptr &&
                         *end_ptr == '\0' && raw >= 0.0 && is_finite;
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
  double eta_damp    = kEtaDampDefault;
  double lambda_damp = kLambdaDampDefault;

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

    if (arg.rfind("--eta_damp=", 0) == 0) {
      const std::string value = arg.substr(std::string("--eta_damp=").size());
      if (!TryParseNonnegativeDouble(value, &eta_damp)) {
        std::cerr << "Invalid --eta_damp value: " << value << "\n";
        return 1;
      }
      continue;
    }

    if (arg.rfind("--lambda_damp=", 0) == 0) {
      const std::string value =
          arg.substr(std::string("--lambda_damp=").size());
      if (!TryParseNonnegativeDouble(value, &lambda_damp)) {
        std::cerr << "Invalid --lambda_damp value: " << value << "\n";
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
                  << " [--eta_damp=value] [--lambda_damp=value]" << std::endl;
        return 1;
      }
      ++positional_index;
      continue;
    }

    std::cerr << "Unknown argument: " << arg << "\n"
              << "Usage: " << argv[0]
              << " [max_steps] [export_interval] [--csv[=path]]"
              << " [--eta_damp=value] [--lambda_damp=value]" << std::endl;
    return 1;
  }

  const std::string output_dir = MakeOutputDirectory(eta_damp, lambda_damp);
  if (write_csv && csv_path.empty()) {
    csv_path = MakeCsvOutputPath(output_dir);
  }

  std::cout << "========================================\n";
  std::cout << "FEAT10 Double Pendulum Spherical-Joint Demo\n";
  std::cout << "========================================\n";
  std::cout << "steps=" << max_steps << " export_interval=" << export_interval
            << "\n";
  std::cout << "eta_damp=" << eta_damp << " lambda_damp=" << lambda_damp
            << "\n";
  std::cout << "output_dir=" << output_dir << "\n";
  if (write_csv) {
    std::cout << "csv=" << csv_path << "\n";
  }

  std::filesystem::create_directories(output_dir);

  const std::string mesh_prefix =
      "data/meshes/T10/double_pendulum/pendulum_beam.1";
  const SolidMaterialProperties link_material = SolidMaterialProperties::SVK(
      kYoungsModulus, kPoissonsRatio, kDensity, eta_damp, lambda_damp);

  ANCFCPUUtils::MeshManager mesh_manager;
  const int mesh_upper = mesh_manager.LoadMesh(
      mesh_prefix + ".node", mesh_prefix + ".ele", "upper", link_material);
  const int mesh_lower = mesh_manager.LoadMesh(
      mesh_prefix + ".node", mesh_prefix + ".ele", "lower", link_material);
  if (mesh_upper < 0 || mesh_lower < 0) {
    std::cerr << "Failed to load pendulum beam meshes from " << mesh_prefix
              << std::endl;
    return 1;
  }

  const auto& inst_upper = mesh_manager.GetMeshInstance(mesh_upper);
  const auto& inst_lower = mesh_manager.GetMeshInstance(mesh_lower);

  const Eigen::Vector3d top_hinge(0.0, 0.0, 0.7);
  const double upper_angle        = kPi - DegToRad(kUpperAngleDeg);
  const double lower_angle        = kPi - DegToRad(kLowerAngleDeg);
  const double upper_out_of_plane = DegToRad(kUpperOutOfPlaneDeg);
  const double lower_out_of_plane = DegToRad(kLowerOutOfPlaneDeg);

  const Eigen::Matrix4d upper_transform = MakeBeamTransform(
      upper_out_of_plane, upper_angle, top_hinge, kUpperHingeLocalZ);
  mesh_manager.TransformMesh(mesh_upper, upper_transform);

  const Eigen::Vector3d lower_hinge = TransformPoint(
      upper_transform, Eigen::Vector3d(0.0, 0.0, kLowerHingeLocalZ));
  const Eigen::Matrix4d lower_transform = MakeBeamTransform(
      lower_out_of_plane, lower_angle, lower_hinge, kLowerLinkHingeLocalZ);
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
  std::cout << "upper out-of-plane tilt: " << kUpperOutOfPlaneDeg << " deg\n";
  std::cout << "lower out-of-plane tilt: " << kLowerOutOfPlaneDeg << " deg\n";

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
  const auto lower_tip_on_lower = constraint_manager.LocateReferencePoint(
      TransformPoint(lower_transform, Eigen::Vector3d(0.0, 0.0, kBeamLength)),
      MakeElementRange(inst_lower));
  constraint_manager.AddSphericalJointToWorld(MakeElementRange(inst_upper),
                                              top_hinge);
  constraint_manager.AddSphericalJoint(
      MakeElementRange(inst_upper), MakeElementRange(inst_lower), lower_hinge);
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
  constexpr int kRowsPerSphericalJoint               = 3;
  const std::vector<int> upper_joint_constraint_rows = {0, 1, 2};
  const std::vector<int> lower_joint_constraint_rows = {3, 4, 5};
  if (gpu_t10_data.get_n_constraint() != 2 * kRowsPerSphericalJoint) {
    std::cerr << "Unexpected spherical constraint count: "
              << gpu_t10_data.get_n_constraint() << " (expected "
              << 2 * kRowsPerSphericalJoint << ")" << std::endl;
    return 1;
  }

  engineering_joint::DoublePendulumSphericalCsvWriter csv_writer;
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
      engineering_joint::EvaluateCurrentPointPosition(
          lower_hinge_on_upper, all_elems, x_curr, y_curr, z_curr);
  const Eigen::Vector3d lower_lower_initial =
      engineering_joint::EvaluateCurrentPointPosition(
          lower_hinge_on_lower, all_elems, x_curr, y_curr, z_curr);
  const Eigen::Vector3d lower_tip_initial =
      engineering_joint::EvaluateCurrentPointPosition(
          lower_tip_on_lower, all_elems, x_curr, y_curr, z_curr);
  const Eigen::Vector3d upper_dir_initial = lower_upper_initial - top_hinge;
  const Eigen::Vector3d lower_dir_initial =
      lower_tip_initial - lower_lower_initial;
  std::cout << "initial lower hinge on upper: ["
            << lower_upper_initial.transpose() << "]\n";
  std::cout << "initial lower hinge on lower: ["
            << lower_lower_initial.transpose() << "]\n";
  std::cout << "initial lower hinge mismatch norm: "
            << (lower_upper_initial - lower_lower_initial).norm() << "\n";
  std::cout << "writing initial frame to " << MakeOutputPath(output_dir, 0)
            << "\n";
  gpu_t10_data.WriteOutputVTU(MakeOutputPath(output_dir, 0));

  if (write_csv) {
    const std::filesystem::path csv_parent =
        std::filesystem::path(csv_path).parent_path();
    if (!csv_parent.empty()) {
      std::filesystem::create_directories(csv_parent);
    }
    csv_writer.Open(csv_path);
    const double potential_energy = engineering_joint::ComputePotentialEnergy(
        lumped_mass, z_curr, kGravity);
    const double elastic_strain_energy =
        engineering_joint::ComputeElasticStrainEnergy(gpu_t10_data,
                                                      link_material);
    const double kinetic_energy =
        engineering_joint::ComputeKineticEnergy(lumped_mass, velocity_xyz);
    const double total_energy =
        kinetic_energy + potential_energy + elastic_strain_energy;
    const engineering_joint::HingeWrench zero_wrench;
    csv_writer.WriteRow(
        0, 0.0,
        engineering_joint::ComputeSwingAngleFromNegativeZ(upper_dir_initial),
        engineering_joint::ComputeAzimuth(upper_dir_initial),
        engineering_joint::ComputeSwingAngleFromNegativeZ(lower_dir_initial),
        engineering_joint::ComputeAzimuth(lower_dir_initial),
        engineering_joint::ComputeAngleBetween(upper_dir_initial,
                                               lower_dir_initial),
        constraint_values.norm(),
        engineering_joint::ComputeInfinityNorm(constraint_values),
        engineering_joint::ComputeSegmentL2Norm(constraint_values, 0,
                                                kRowsPerSphericalJoint),
        engineering_joint::ComputeIndexedInfinityNorm(
            constraint_values, upper_joint_constraint_rows),
        engineering_joint::ComputeSegmentL2Norm(
            constraint_values, kRowsPerSphericalJoint, kRowsPerSphericalJoint),
        engineering_joint::ComputeIndexedInfinityNorm(
            constraint_values, lower_joint_constraint_rows),
        total_energy, kinetic_energy, potential_energy, elastic_strain_energy,
        zero_wrench, zero_wrench,
        (lower_upper_initial - lower_lower_initial).norm(), upper_dir_initial,
        lower_dir_initial, lower_tip_initial);
    std::cout << "writing csv metrics to " << csv_path << "\n";
  }

  int output_frame = 1;
  for (int step = 1; step <= max_steps; ++step) {
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
        engineering_joint::EvaluateCurrentPointPosition(
            lower_hinge_on_upper, all_elems, x_curr, y_curr, z_curr);
    const Eigen::Vector3d lower_lower_current =
        engineering_joint::EvaluateCurrentPointPosition(
            lower_hinge_on_lower, all_elems, x_curr, y_curr, z_curr);
    const Eigen::Vector3d lower_tip_current =
        engineering_joint::EvaluateCurrentPointPosition(
            lower_tip_on_lower, all_elems, x_curr, y_curr, z_curr);
    const Eigen::Vector3d upper_dir_current = lower_upper_current - top_hinge;
    const Eigen::Vector3d lower_dir_current =
        lower_tip_current - lower_lower_current;
    const Eigen::Vector3d lower_hinge_current = engineering_joint::AveragePoint(
        lower_upper_current, lower_lower_current);
    const double lower_hinge_mismatch_norm =
        (lower_upper_current - lower_lower_current).norm();

    if (write_csv) {
      const double potential_energy = engineering_joint::ComputePotentialEnergy(
          lumped_mass, z_curr, kGravity);
      const double elastic_strain_energy =
          engineering_joint::ComputeElasticStrainEnergy(gpu_t10_data,
                                                        link_material);
      const double kinetic_energy =
          engineering_joint::ComputeKineticEnergy(lumped_mass, velocity_xyz);
      const double total_energy =
          kinetic_energy + potential_energy + elastic_strain_energy;
      const Eigen::VectorXd upper_joint_reaction_vector =
          engineering_joint::ComputeGeneralizedReactionFromCSR(
              n_nodes * 3, constraint_j_offsets, constraint_j_columns,
              constraint_j_values, augmented_dual_values, 0,
              kRowsPerSphericalJoint);
      const Eigen::VectorXd lower_joint_reaction_vector =
          engineering_joint::ComputeGeneralizedReactionFromCSR(
              n_nodes * 3, constraint_j_offsets, constraint_j_columns,
              constraint_j_values, augmented_dual_values,
              kRowsPerSphericalJoint, 2 * kRowsPerSphericalJoint);
      const engineering_joint::HingeWrench upper_joint_reaction =
          engineering_joint::ScaleHingeWrench(
              engineering_joint::EstimateHingeWrench(
                  upper_joint_reaction_vector, inst_upper.node_offset,
                  inst_upper.num_nodes, x_curr, y_curr, z_curr, top_hinge),
              kDt);
      const engineering_joint::HingeWrench lower_joint_reaction =
          engineering_joint::ScaleHingeWrench(
              engineering_joint::EstimateHingeWrench(
                  lower_joint_reaction_vector, inst_lower.node_offset,
                  inst_lower.num_nodes, x_curr, y_curr, z_curr,
                  lower_hinge_current),
              kDt);
      csv_writer.WriteRow(
          step, step * kDt,
          engineering_joint::ComputeSwingAngleFromNegativeZ(upper_dir_current),
          engineering_joint::ComputeAzimuth(upper_dir_current),
          engineering_joint::ComputeSwingAngleFromNegativeZ(lower_dir_current),
          engineering_joint::ComputeAzimuth(lower_dir_current),
          engineering_joint::ComputeAngleBetween(upper_dir_current,
                                                 lower_dir_current),
          constraint_values.norm(),
          engineering_joint::ComputeInfinityNorm(constraint_values),
          engineering_joint::ComputeSegmentL2Norm(constraint_values, 0,
                                                  kRowsPerSphericalJoint),
          engineering_joint::ComputeIndexedInfinityNorm(
              constraint_values, upper_joint_constraint_rows),
          engineering_joint::ComputeSegmentL2Norm(constraint_values,
                                                  kRowsPerSphericalJoint,
                                                  kRowsPerSphericalJoint),
          engineering_joint::ComputeIndexedInfinityNorm(
              constraint_values, lower_joint_constraint_rows),
          total_energy, kinetic_energy, potential_energy, elastic_strain_energy,
          upper_joint_reaction, lower_joint_reaction, lower_hinge_mismatch_norm,
          upper_dir_current, lower_dir_current, lower_tip_current);
    }

    if (step % export_interval == 0) {
      gpu_t10_data.WriteOutputVTU(MakeOutputPath(output_dir, output_frame));
      ++output_frame;
    }
  }

  gpu_t10_data.Destroy();
  std::cout << "Done. Output written to " << output_dir << "/\n";
  return 0;
}
