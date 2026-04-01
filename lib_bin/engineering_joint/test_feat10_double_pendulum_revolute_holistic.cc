/**
 * FEAT10 Double Pendulum Revolute-Joint Demo (Holistic Solver)
 *
 * Builds a two-link T10 double pendulum from two separate FEAT10 blocks.
 * The upper link is attached to the world through a revolute joint, and the
 * lower link is attached to the upper link through a second revolute joint.
 *
 * This demo exists as a temporary exerciser for the new holistic mixed
 * constraint path:
 *   FEMultiElementProblem + MixedConstraintSystem + HolisticNewtonSolver
 *
 * Output:
 *   output/engineering_joint/double_pendulum_holistic_upper_XXXXXX.vtu
 *   output/engineering_joint/double_pendulum_holistic_lower_XXXXXX.vtu
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../lib_src/constraints/MixedConstraintSystem.h"
#include "../../lib_src/elements/FEAT10Data.cuh"
#include "../../lib_src/solvers/FEMultiElementProblem.h"
#include "../../lib_src/solvers/HolisticNewton.cuh"
#include "../../lib_utils/cpu_utils.h"
#include "../../lib_utils/cuda_utils.h"
#include "../../lib_utils/mesh_manager.h"
#include "../../lib_utils/quadrature_utils.h"

namespace {

constexpr double kPi             = 3.14159265358979323846;
constexpr double kGravity        = -9.81;
constexpr double kDt             = 2e-4;
constexpr int kNumStepsDefault   = 3000;
constexpr int kExportIntervalDef = 50;

constexpr double kBeamLength           = 0.5;
constexpr double kUpperHingeLocalZ     = 0.0;
constexpr double kLowerHingeLocalZ     = kBeamLength;
constexpr double kLowerLinkHingeLocalZ = 0.0;
constexpr double kJointOffset          = 0.001;

constexpr double kUpperAngleDeg = 35.0;
constexpr double kLowerAngleDeg = -25.0;

const SolidMaterialProperties kLinkMaterial =
    SolidMaterialProperties::SVK(1.0e7,   // E
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
  const Eigen::Vector4d point_h(point.x(), point.y(), point.z(), 1.0);
  return (transform * point_h).head<3>();
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

Eigen::MatrixXd ExtractLocalNodes(const Eigen::MatrixXd& global_nodes,
                                  const ANCFCPUUtils::MeshInstance& inst) {
  return global_nodes.middleRows(inst.node_offset, inst.num_nodes);
}

Eigen::MatrixXi ExtractLocalElements(const Eigen::MatrixXi& global_elements,
                                     const ANCFCPUUtils::MeshInstance& inst) {
  Eigen::MatrixXi local =
      global_elements.middleRows(inst.element_offset, inst.num_elements);
  local.array() -= inst.node_offset;
  return local;
}

Eigen::VectorXd ExtractAxis(const Eigen::MatrixXd& nodes, int axis) {
  Eigen::VectorXd values(nodes.rows());
  for (int i = 0; i < nodes.rows(); ++i) {
    values(i) = nodes(i, axis);
  }
  return values;
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

void AppendGravityForBlock(Eigen::VectorXd* h_f_ext,
                           const std::vector<double>& lumped_mass,
                           int coef_offset, double gravity) {
  for (int local_node = 0; local_node < static_cast<int>(lumped_mass.size());
       ++local_node) {
    const int global_node = coef_offset + local_node;
    (*h_f_ext)(3 * global_node + 2) +=
        lumped_mass[static_cast<size_t>(local_node)] * gravity;
  }
}

Eigen::Vector3d BuildPerpendicularAxis1(const Eigen::Vector3d& axis) {
  const Eigen::Vector3d trial =
      (std::abs(axis.z()) < 0.9) ? Eigen::Vector3d::UnitZ()
                                 : Eigen::Vector3d::UnitX();
  const Eigen::Vector3d p1 = axis.cross(trial);
  const double norm = p1.norm();
  if (norm < 1e-12) {
    throw std::runtime_error("Failed to construct perpendicular axis");
  }
  return p1 / norm;
}

MixedConstraintPointBinding LocateWithAdaptiveOffset(
    const MixedConstraintSystem& constraints, int block_idx,
    const Eigen::Vector3d& base_point, const Eigen::Vector3d& direction,
    double initial_offset) {
  double offset = initial_offset;
  for (int attempt = 0; attempt < 8; ++attempt) {
    try {
      return constraints.LocateReferencePoint(block_idx,
                                              base_point + offset * direction);
    } catch (const std::runtime_error&) {
      offset *= 0.5;
    }
  }
  throw std::runtime_error("Failed to locate offset reference point");
}

void AddRevoluteJointToWorld(MixedConstraintSystem* constraints, int block_idx,
                             const Eigen::Vector3d& hinge_point,
                             const Eigen::Vector3d& hinge_axis, double offset,
                             double dp1_weight) {
  const double axis_norm = hinge_axis.norm();
  if (axis_norm < 1e-12) {
    throw std::invalid_argument("World revolute axis must be non-zero");
  }

  const Eigen::Vector3d axis = hinge_axis / axis_norm;
  const MixedConstraintPointBinding p =
      constraints->LocateReferencePoint(block_idx, hinge_point);
  const MixedConstraintPointBinding q = LocateWithAdaptiveOffset(
      *constraints, block_idx, hinge_point, axis, offset);

  constraints->AddPointToWorldCDAxis(p, 0, hinge_point.x());
  constraints->AddPointToWorldCDAxis(p, 1, hinge_point.y());
  constraints->AddPointToWorldCDAxis(p, 2, hinge_point.z());

  const Eigen::Vector3d p1 = BuildPerpendicularAxis1(axis);
  const Eigen::Vector3d p2 = axis.cross(p1).normalized();
  constraints->AddWorldDP1Constraint(p, q, p1, 0.0, dp1_weight);
  constraints->AddWorldDP1Constraint(p, q, p2, 0.0, dp1_weight);
}

Eigen::Vector3d EvaluateCurrentPointPosition(
    const MixedConstraintPointBinding& point, const Eigen::VectorXd& x,
    const Eigen::VectorXd& y, const Eigen::VectorXd& z) {
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  for (int i = 0; i < point.count; ++i) {
    const int coef = point.coef_indices[i];
    const double weight = point.weights[i];
    position += weight * Eigen::Vector3d(x(coef), y(coef), z(coef));
  }
  return position;
}

void RetrieveUnifiedPositions(const FEStateBuffer& state, Eigen::VectorXd* x,
                              Eigen::VectorXd* y, Eigen::VectorXd* z) {
  x->resize(state.total_coef);
  y->resize(state.total_coef);
  z->resize(state.total_coef);
  HANDLE_ERROR(cudaMemcpy(x->data(), state.d_x12,
                          static_cast<size_t>(state.total_coef) *
                              sizeof(double),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(y->data(), state.d_y12,
                          static_cast<size_t>(state.total_coef) *
                              sizeof(double),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(z->data(), state.d_z12,
                          static_cast<size_t>(state.total_coef) *
                              sizeof(double),
                          cudaMemcpyDeviceToHost));
}

std::string MakeOutputPath(const std::string& body_name, int frame) {
  std::ostringstream oss;
  oss << "output/engineering_joint/double_pendulum_holistic_" << body_name
      << "_" << std::setw(6) << std::setfill('0') << frame << ".vtu";
  return oss.str();
}

}  // namespace

int main(int argc, char** argv) {
  int max_steps = kNumStepsDefault;
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
  std::cout << "FEAT10 Double Pendulum Revolute (Holistic)\n";
  std::cout << "========================================\n";
  std::cout << "steps=" << max_steps
            << " export_interval=" << export_interval << "\n";

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

  const Eigen::MatrixXd upper_nodes = ExtractLocalNodes(all_nodes, inst_upper);
  const Eigen::MatrixXd lower_nodes = ExtractLocalNodes(all_nodes, inst_lower);
  const Eigen::MatrixXi upper_elems = ExtractLocalElements(all_elems, inst_upper);
  const Eigen::MatrixXi lower_elems = ExtractLocalElements(all_elems, inst_lower);

  auto upper_data =
      std::make_unique<GPU_FEAT10_Data>(inst_upper.num_elements,
                                        inst_upper.num_nodes);
  upper_data->Initialize();
  upper_data->Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y,
                    Quadrature::tet5pt_z, Quadrature::tet5pt_weights,
                    ExtractAxis(upper_nodes, 0), ExtractAxis(upper_nodes, 1),
                    ExtractAxis(upper_nodes, 2), upper_elems);
  upper_data->ApplyMaterial(kLinkMaterial);
  upper_data->CalcDnDuPre();
  upper_data->CalcMassMatrix();

  auto lower_data =
      std::make_unique<GPU_FEAT10_Data>(inst_lower.num_elements,
                                        inst_lower.num_nodes);
  lower_data->Initialize();
  lower_data->Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y,
                    Quadrature::tet5pt_z, Quadrature::tet5pt_weights,
                    ExtractAxis(lower_nodes, 0), ExtractAxis(lower_nodes, 1),
                    ExtractAxis(lower_nodes, 2), lower_elems);
  lower_data->ApplyMaterial(kLinkMaterial);
  lower_data->CalcDnDuPre();
  lower_data->CalcMassMatrix();

  const std::vector<double> upper_lumped_mass = ComputeLumpedMass(*upper_data);
  const std::vector<double> lower_lumped_mass = ComputeLumpedMass(*lower_data);

  FEMultiElementProblem problem;
  const int upper_block = problem.AddElementBlock(upper_data.get(), TYPE_T10);
  const int lower_block = problem.AddElementBlock(lower_data.get(), TYPE_T10);
  problem.Finalize();

  Eigen::VectorXd h_f_ext = Eigen::VectorXd::Zero(problem.GetTotalDofs());
  const FEStateBuffer& state = problem.GetStateBuffer();
  AppendGravityForBlock(&h_f_ext, upper_lumped_mass,
                        state.blocks[static_cast<size_t>(upper_block)].coef_offset,
                        kGravity);
  AppendGravityForBlock(&h_f_ext, lower_lumped_mass,
                        state.blocks[static_cast<size_t>(lower_block)].coef_offset,
                        kGravity);
  HANDLE_ERROR(cudaMemcpy(state.d_f_ext, h_f_ext.data(),
                          static_cast<size_t>(problem.GetTotalDofs()) *
                              sizeof(double),
                          cudaMemcpyHostToDevice));

  MixedConstraintSystem constraints(&problem);
  AddRevoluteJointToWorld(&constraints, upper_block, top_hinge,
                          Eigen::Vector3d::UnitY(), kJointOffset, 1.0);
  constraints.AddRevoluteJoint(upper_block, lower_block, lower_hinge,
                               Eigen::Vector3d::UnitY(), kJointOffset, 1.0);

  const MixedConstraintPointBinding lower_hinge_on_upper =
      constraints.LocateReferencePoint(upper_block, lower_hinge);
  const MixedConstraintPointBinding lower_hinge_on_lower =
      constraints.LocateReferencePoint(lower_block, lower_hinge);
  constraints.Finalize();

  HolisticNewtonParams params;
  params.inner_atol = 1e-4;
  params.inner_rtol = 1e-4;
  params.outer_tol = 1e-6;
  params.rho = 1e9;
  params.max_outer = 8;
  params.max_inner = 10;
  params.time_step = kDt;
  params.enable_line_search = false;

  HolisticNewtonSolver solver(&problem, &constraints);
  solver.SetParameters(&params);
  solver.Setup();

  Eigen::VectorXd x_curr, y_curr, z_curr;
  RetrieveUnifiedPositions(problem.GetStateBuffer(), &x_curr, &y_curr, &z_curr);
  const Eigen::Vector3d lower_upper_initial =
      EvaluateCurrentPointPosition(lower_hinge_on_upper, x_curr, y_curr, z_curr);
  const Eigen::Vector3d lower_lower_initial =
      EvaluateCurrentPointPosition(lower_hinge_on_lower, x_curr, y_curr, z_curr);

  std::cout << "constraints: " << constraints.num_constraints() << "\n";
  std::cout << "initial lower hinge mismatch norm: "
            << (lower_upper_initial - lower_lower_initial).norm() << "\n";

  upper_data->WriteOutputVTU(MakeOutputPath("upper", 0));
  lower_data->WriteOutputVTU(MakeOutputPath("lower", 0));

  int output_frame = 1;
  for (int step = 1; step <= max_steps; ++step) {
    solver.Solve();

    RetrieveUnifiedPositions(problem.GetStateBuffer(), &x_curr, &y_curr,
                             &z_curr);
    const Eigen::Vector3d lower_upper_current =
        EvaluateCurrentPointPosition(lower_hinge_on_upper, x_curr, y_curr,
                                     z_curr);
    const Eigen::Vector3d lower_lower_current =
        EvaluateCurrentPointPosition(lower_hinge_on_lower, x_curr, y_curr,
                                     z_curr);

    Eigen::VectorXd constraint_values(constraints.num_constraints());
    HANDLE_ERROR(cudaMemcpy(constraint_values.data(),
                            constraints.GetConstraintDevicePtr(),
                            static_cast<size_t>(constraints.num_constraints()) *
                                sizeof(double),
                            cudaMemcpyDeviceToHost));

    std::cout << "step " << step
              << " constraint_norm=" << constraint_values.norm()
              << " lower_hinge_mismatch="
              << (lower_upper_current - lower_lower_current).norm() << "\n";

    if (step % export_interval == 0) {
      upper_data->WriteOutputVTU(MakeOutputPath("upper", output_frame));
      lower_data->WriteOutputVTU(MakeOutputPath("lower", output_frame));
      ++output_frame;
    }
  }

  upper_data->Destroy();
  lower_data->Destroy();

  std::cout << "Done. Output written to output/engineering_joint/\n";
  return 0;
}
