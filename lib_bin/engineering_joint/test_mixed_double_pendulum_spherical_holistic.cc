/**
 * Mixed ANCF3243 + FEAT10 Double Pendulum — Spherical Both Ends (Holistic)
 *
 * Upper link:  ANCF3243 beam.
 * Lower link:  FEAT10 T10 block.
 *
 * World anchor: 3-axis CD lock on ANCF node 0 (ball-in-socket to world).
 * Elbow:        spherical joint (3 CD rows) between ANCF tip and T10 hinge.
 *
 * ANCF-side bindings use MakeANCF3243NodeBinding at the root and tip nodes
 * — a direct single-coef reference to each node's position slot.  This
 * bypasses LocateReferencePointANCF3243's off-centerline Newton-search
 * limitation entirely.
 *
 * T10-side bindings go through LocateReferencePoint as usual — the T10
 * locator is reliable.
 *
 * Output:
 *   output/engineering_joint/mixed_double_pendulum_spherical_holistic_upper_XXXXXX.vtu
 *   output/engineering_joint/mixed_double_pendulum_spherical_holistic_lower_XXXXXX.vtu
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
#include "../../lib_src/elements/ANCF3243Data.cuh"
#include "../../lib_src/elements/FEAT10Data.cuh"
#include "../../lib_src/solvers/FEMultiElementProblem.h"
#include "../../lib_src/solvers/HolisticNewton.cuh"
#include "../../lib_utils/cpu_utils.h"
#include "../../lib_utils/cuda_utils.h"
#include "../../lib_utils/mesh_manager.h"
#include "../../lib_utils/mesh_utils.h"
#include "../../lib_utils/quadrature_utils.h"
#include "../../lib_utils/visualization_utils.h"

namespace {

constexpr double kPi             = 3.14159265358979323846;
constexpr double kGravity        = -9.81;
constexpr double kDt             = 2e-4;
constexpr int kNumStepsDefault   = 3000;
constexpr int kExportIntervalDef = 50;

// ANCF upper beam matches the T10 block's cross-section (0.04 x 0.04) and
// density, so both links carry the same translational mass (0.96 kg).
// kBeamE is 7.5e6 (nominal 1.0e7) — a 0.75x scale calibrated offline
// against the T10 block's self-weight deflection to compensate for
// ANCF3243's ~25% overstiffness at matched E.
constexpr int kBeamElements  = 5;
constexpr double kBeamLength = 0.5;
constexpr double kBeamL      = kBeamLength / kBeamElements;
constexpr double kBeamW      = 0.04;
constexpr double kBeamH      = 0.04;
constexpr double kBeamE      = 7.5e6;  // calibrated; nominal is 1.0e7
constexpr double kBeamNu     = 0.30;
constexpr double kBeamRho    = 1200.0;
constexpr double kBeamEta    = 1.0e4;
constexpr double kBeamLambda = 1.0e4;

// Pose — matches test_feat10_double_pendulum_spherical_holistic exactly.
constexpr double kUpperHingeLocalZ     = 0.0;
constexpr double kLowerHingeLocalZ     = kBeamLength;
constexpr double kLowerLinkHingeLocalZ = 0.0;
constexpr double kUpperAngleDeg        = 35.0;
constexpr double kLowerAngleDeg        = -25.0;
constexpr double kUpperOutOfPlaneDeg   = 18.0;
constexpr double kLowerOutOfPlaneDeg   = -27.0;

const SolidMaterialProperties kLinkMaterial =
    SolidMaterialProperties::SVK(1.0e7, 0.30, 1200.0, 1.0e4, 1.0e4);

double DegToRad(double angle_deg) { return angle_deg * kPi / 180.0; }

Eigen::Vector3d TransformPoint(const Eigen::Matrix4d& transform,
                               const Eigen::Vector3d& point) {
  const Eigen::Vector4d point_h(point.x(), point.y(), point.z(), 1.0);
  return (transform * point_h).head<3>();
}

Eigen::Matrix4d MakeBeamTransform(double angle_x, double angle_y,
                                  const Eigen::Vector3d& hinge_point,
                                  double hinge_local_z) {
  const Eigen::Matrix4d rotation =
      ANCFCPUUtils::rotationX(angle_x) * ANCFCPUUtils::rotationY(angle_y);
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
  for (int i = 0; i < nodes.rows(); ++i) values(i) = nodes(i, axis);
  return values;
}

template <typename ElementData>
std::vector<double> ComputeLumpedMass(ElementData& data) {
  std::vector<int> offsets;
  std::vector<int> columns;
  std::vector<double> values;
  data.RetrieveMassCSRToCPU(offsets, columns, values);

  const int n_coef = data.get_n_coef();
  std::vector<double> lumped_mass(static_cast<size_t>(n_coef), 0.0);
  if (offsets.size() != static_cast<size_t>(n_coef + 1)) {
    std::fill(lumped_mass.begin(), lumped_mass.end(), 1.0);
    return lumped_mass;
  }
  for (int row = 0; row < n_coef; ++row) {
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
  oss << "output/engineering_joint/mixed_double_pendulum_spherical_holistic_"
      << body_name << "_" << std::setw(6) << std::setfill('0') << frame
      << ".vtu";
  return oss.str();
}

// Rotate + translate the ANCF beam coefficients so that:
//   - Node 0's position lands at `hinge_point`.
//   - The beam extends in the direction `R_target * +Z` (matching the T10
//     convention where the beam's local long axis is +Z from the hinge).
// GridMeshGenerator emits the beam along +X starting at raw_root, so we
// pre-apply rotationY(-π/2) to re-align the beam's +X axis to +Z before
// applying the target rotation.  All four coefficient slots per node
// (position + 3 slopes) are rotated; only the position slot is translated.
void TransformANCFBeam(Eigen::VectorXd* h_x12, Eigen::VectorXd* h_y12,
                       Eigen::VectorXd* h_z12, int n_nodes,
                       const Eigen::Matrix3d& R_target,
                       const Eigen::Vector3d& hinge_point) {
  constexpr int kDofsPerNode = 4;
  // rotationY(-π/2): maps +X → +Z, preserves +Y, maps +Z → -X.
  const double c = 0.0;   // cos(-π/2)
  const double s = -1.0;  // sin(-π/2)
  Eigen::Matrix3d R_x_to_z;
  R_x_to_z <<  c, 0.0,  s,
             0.0, 1.0, 0.0,
              -s, 0.0,  c;
  const Eigen::Matrix3d R_full = R_target * R_x_to_z;

  const Eigen::Vector3d raw_root(
      (*h_x12)(0), (*h_y12)(0), (*h_z12)(0));
  const Eigen::Vector3d t = hinge_point - R_full * raw_root;

  for (int node = 0; node < n_nodes; ++node) {
    for (int slot = 0; slot < kDofsPerNode; ++slot) {
      const int idx = node * kDofsPerNode + slot;
      const Eigen::Vector3d v((*h_x12)(idx), (*h_y12)(idx), (*h_z12)(idx));
      Eigen::Vector3d v_new = R_full * v;
      if (slot == 0) v_new += t;
      (*h_x12)(idx) = v_new.x();
      (*h_y12)(idx) = v_new.y();
      (*h_z12)(idx) = v_new.z();
    }
  }
}

Eigen::Vector3d EvaluateANCFTip(const Eigen::VectorXd& x,
                                const Eigen::VectorXd& y,
                                const Eigen::VectorXd& z, int beam_coef_offset,
                                int n_nodes) {
  const int tip_node = n_nodes - 1;
  const int tip_pos_coef = beam_coef_offset + tip_node * 4;
  return Eigen::Vector3d(x(tip_pos_coef), y(tip_pos_coef), z(tip_pos_coef));
}

bool WriteANCFVTU(GPU_ANCF3243_Data& data,
                  const Eigen::MatrixXi& h_element_connectivity,
                  const std::string& path) {
  Eigen::VectorXd x12(data.get_n_coef());
  Eigen::VectorXd y12(data.get_n_coef());
  Eigen::VectorXd z12(data.get_n_coef());
  data.RetrievePositionToCPU(x12, y12, z12);

  Eigen::VectorXd vm;
  data.ComputeVonMises();
  data.RetrieveVonMisesToCPU(vm);

  return ANCFCPUUtils::VisualizationUtils::ExportANCF3243ToVTU(
      x12, y12, z12, h_element_connectivity, kBeamW, kBeamH, path, &vm);
}

void PrintBinding(const std::string& name,
                  const MixedConstraintPointBinding& b) {
  std::cout << "  " << name << ": count=" << b.count << "  ";
  for (int i = 0; i < b.count; ++i) {
    std::cout << "(coef=" << b.coef_indices[i]
              << ",w=" << b.weights[i] << ") ";
  }
  std::cout << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  int max_steps = kNumStepsDefault;
  int export_interval = kExportIntervalDef;
  if (argc > 1) {
    const int parsed_steps = std::atoi(argv[1]);
    if (parsed_steps > 0) max_steps = parsed_steps;
  }
  if (argc > 2) {
    const int parsed_interval = std::atoi(argv[2]);
    if (parsed_interval > 0) export_interval = parsed_interval;
  }

  std::cout << "========================================\n";
  std::cout << "Mixed ANCF+T10 — Spherical Both Ends (Holistic)\n";
  std::cout << "========================================\n";
  std::cout << "steps=" << max_steps
            << " export_interval=" << export_interval << "\n";

  std::filesystem::create_directories("output/engineering_joint");

  // -------- Build ANCF3243 upper beam (3D-tilted pose matching T10 demo) ---
  const Eigen::Vector3d top_hinge(0.0, 0.0, 0.7);

  ANCFCPUUtils::GridMeshGenerator grid_gen(kBeamLength, 0.0, kBeamL, true,
                                           false);
  grid_gen.generate_mesh();
  const int beam_n_nodes    = grid_gen.get_num_nodes();
  const int beam_n_elements = grid_gen.get_num_elements();

  auto beam_data =
      std::make_unique<GPU_ANCF3243_Data>(beam_n_nodes, beam_n_elements);
  beam_data->Initialize();

  Eigen::VectorXd h_x12(beam_data->get_n_coef());
  Eigen::VectorXd h_y12(beam_data->get_n_coef());
  Eigen::VectorXd h_z12(beam_data->get_n_coef());
  grid_gen.get_coordinates(h_x12, h_y12, h_z12);

  // Build the T10-style rotation matrix (rotationX * rotationY) that places
  // the upper link hinge at top_hinge and orients local +Z along the beam.
  const double upper_angle        = kPi - DegToRad(kUpperAngleDeg);
  const double upper_out_of_plane = DegToRad(kUpperOutOfPlaneDeg);
  const Eigen::Matrix4d upper_transform_4d =
      MakeBeamTransform(upper_out_of_plane, upper_angle, top_hinge,
                        kUpperHingeLocalZ);
  const Eigen::Matrix3d R_upper = upper_transform_4d.topLeftCorner<3, 3>();

  TransformANCFBeam(&h_x12, &h_y12, &h_z12, beam_n_nodes, R_upper, top_hinge);

  Eigen::MatrixXi h_beam_connectivity;
  grid_gen.get_element_connectivity(h_beam_connectivity);

  beam_data->Setup(kBeamL, kBeamW, kBeamH, Quadrature::gauss_xi_m_6,
                   Quadrature::gauss_xi_3, Quadrature::gauss_eta_2,
                   Quadrature::gauss_zeta_2, Quadrature::weight_xi_m_6,
                   Quadrature::weight_xi_3, Quadrature::weight_eta_2,
                   Quadrature::weight_zeta_2, h_x12, h_y12, h_z12,
                   h_beam_connectivity);
  beam_data->SetDensity(kBeamRho);
  beam_data->SetDamping(kBeamEta, kBeamLambda);
  beam_data->SetSVK(kBeamE, kBeamNu);
  beam_data->CalcDsDuPre();
  beam_data->CalcMassMatrix();

  // -------- Build T10 lower block (same pipeline as T10 demo) -------------
  const Eigen::Vector3d lower_hinge = TransformPoint(
      upper_transform_4d, Eigen::Vector3d(0.0, 0.0, kLowerHingeLocalZ));

  const std::string mesh_prefix =
      "data/meshes/T10/double_pendulum/pendulum_beam.1";
  ANCFCPUUtils::MeshManager mesh_manager;
  const int mesh_lower = mesh_manager.LoadMesh(
      mesh_prefix + ".node", mesh_prefix + ".ele", "lower", kLinkMaterial);
  if (mesh_lower < 0) {
    std::cerr << "Failed to load T10 mesh from " << mesh_prefix << std::endl;
    return 1;
  }
  const auto& inst_lower = mesh_manager.GetMeshInstance(mesh_lower);

  const double lower_angle        = kPi - DegToRad(kLowerAngleDeg);
  const double lower_out_of_plane = DegToRad(kLowerOutOfPlaneDeg);
  const Eigen::Matrix4d lower_transform = MakeBeamTransform(
      lower_out_of_plane, lower_angle, lower_hinge, kLowerLinkHingeLocalZ);
  mesh_manager.TransformMesh(mesh_lower, lower_transform);

  const Eigen::MatrixXd& all_nodes = mesh_manager.GetAllNodes();
  const Eigen::MatrixXi& all_elems = mesh_manager.GetAllElements();
  const Eigen::MatrixXd lower_nodes = ExtractLocalNodes(all_nodes, inst_lower);
  const Eigen::MatrixXi lower_elems =
      ExtractLocalElements(all_elems, inst_lower);

  auto lower_data = std::make_unique<GPU_FEAT10_Data>(inst_lower.num_elements,
                                                       inst_lower.num_nodes);
  lower_data->Initialize();
  lower_data->Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y,
                    Quadrature::tet5pt_z, Quadrature::tet5pt_weights,
                    ExtractAxis(lower_nodes, 0), ExtractAxis(lower_nodes, 1),
                    ExtractAxis(lower_nodes, 2), lower_elems);
  lower_data->ApplyMaterial(kLinkMaterial);
  lower_data->CalcDnDuPre();
  lower_data->CalcMassMatrix();

  const std::vector<double> beam_lumped_mass  = ComputeLumpedMass(*beam_data);
  const std::vector<double> lower_lumped_mass = ComputeLumpedMass(*lower_data);

  // -------- Register blocks and apply gravity -----------------------------
  FEMultiElementProblem problem;
  const int upper_block = problem.AddElementBlock(beam_data.get(), TYPE_3243);
  const int lower_block = problem.AddElementBlock(lower_data.get(), TYPE_T10);
  problem.Finalize();

  Eigen::VectorXd h_f_ext = Eigen::VectorXd::Zero(problem.GetTotalDofs());
  const FEStateBuffer& state = problem.GetStateBuffer();
  const int beam_coef_offset =
      state.blocks[static_cast<size_t>(upper_block)].coef_offset;
  const int lower_coef_offset =
      state.blocks[static_cast<size_t>(lower_block)].coef_offset;
  AppendGravityForBlock(&h_f_ext, beam_lumped_mass, beam_coef_offset,
                        kGravity);
  AppendGravityForBlock(&h_f_ext, lower_lumped_mass, lower_coef_offset,
                        kGravity);
  HANDLE_ERROR(cudaMemcpy(state.d_f_ext, h_f_ext.data(),
                          static_cast<size_t>(problem.GetTotalDofs()) *
                              sizeof(double),
                          cudaMemcpyHostToDevice));

  // Constraints: ANCF node bindings + LocateReferencePoint for T10.
  MixedConstraintSystem constraints(&problem);

  const MixedConstraintPointBinding p_world =
      constraints.MakeANCF3243NodeBinding(upper_block, 0);
  const MixedConstraintPointBinding p_elbow_upper =
      constraints.MakeANCF3243NodeBinding(upper_block, beam_n_nodes - 1);

  const MixedConstraintPointBinding p_elbow_lower =
      constraints.LocateReferencePoint(lower_block, lower_hinge);

  // World anchor: 3 CD (ANCF root position locked to top_hinge).
  constraints.AddPointToWorldCDAxis(p_world, 0, top_hinge.x());
  constraints.AddPointToWorldCDAxis(p_world, 1, top_hinge.y());
  constraints.AddPointToWorldCDAxis(p_world, 2, top_hinge.z());

  // Elbow: 3 CD (ANCF tip = T10 reference point).
  constraints.AddSphericalJoint(p_elbow_upper, p_elbow_lower);

  constraints.Finalize();

  std::cout << "bindings:\n";
  PrintBinding("p_world       (ANCF root)", p_world);
  PrintBinding("p_elbow_upper (ANCF tip) ", p_elbow_upper);
  PrintBinding("p_elbow_lower (T10)      ", p_elbow_lower);

  // -------- Solver --------------------------------------------------------
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
  RetrieveUnifiedPositions(state, &x_curr, &y_curr, &z_curr);
  const Eigen::Vector3d elbow_upper_initial =
      EvaluateCurrentPointPosition(p_elbow_upper, x_curr, y_curr, z_curr);
  const Eigen::Vector3d elbow_lower_initial =
      EvaluateCurrentPointPosition(p_elbow_lower, x_curr, y_curr, z_curr);

  std::cout << "constraints: " << constraints.num_constraints() << "\n";
  std::cout << "initial elbow mismatch norm: "
            << (elbow_upper_initial - elbow_lower_initial).norm() << "\n";
  std::cout << "ancf_beam: n_nodes=" << beam_n_nodes
            << " n_elements=" << beam_n_elements
            << " coef_offset=" << beam_coef_offset << "\n";

  WriteANCFVTU(*beam_data, h_beam_connectivity, MakeOutputPath("upper", 0));
  lower_data->WriteOutputVTU(MakeOutputPath("lower", 0));

  int output_frame = 1;
  for (int step = 1; step <= max_steps; ++step) {
    solver.Solve();

    RetrieveUnifiedPositions(state, &x_curr, &y_curr, &z_curr);
    const Eigen::Vector3d elbow_upper_current =
        EvaluateCurrentPointPosition(p_elbow_upper, x_curr, y_curr, z_curr);
    const Eigen::Vector3d elbow_lower_current =
        EvaluateCurrentPointPosition(p_elbow_lower, x_curr, y_curr, z_curr);
    const Eigen::Vector3d ancf_tip =
        EvaluateANCFTip(x_curr, y_curr, z_curr, beam_coef_offset,
                        beam_n_nodes);

    Eigen::VectorXd constraint_values(constraints.num_constraints());
    HANDLE_ERROR(cudaMemcpy(
        constraint_values.data(), constraints.GetConstraintDevicePtr(),
        static_cast<size_t>(constraints.num_constraints()) * sizeof(double),
        cudaMemcpyDeviceToHost));

    std::cout << "step " << step
              << " constraint_norm=" << constraint_values.norm()
              << " elbow_mismatch="
              << (elbow_upper_current - elbow_lower_current).norm()
              << " ancf_tip_z=" << ancf_tip.z() << "\n";

    if (step % export_interval == 0) {
      WriteANCFVTU(*beam_data, h_beam_connectivity,
                   MakeOutputPath("upper", output_frame));
      lower_data->WriteOutputVTU(MakeOutputPath("lower", output_frame));
      ++output_frame;
    }
  }

  beam_data->Destroy();
  lower_data->Destroy();

  std::cout << "Done. Output written to output/engineering_joint/\n";
  return 0;
}
