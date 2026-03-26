/**
 * Mixed T10 + ANCF3243 Tendon-Driven Finger Curl Demo
 *
 * Builds a four-block finger from the T10 finger_block mesh. The proximal
 * block is grounded by fixing its back face, the remaining blocks are linked
 * by revolute joints along a near-bottom hinge line, and an ANCF3243 tendon
 * beam is routed through one cylindrical guide per block. The tendon is welded
 * to the distal block and pulled from its proximal tail to curl the finger.
 *
 * The full multi-body system is solved monolithically through:
 *   FEMultiElementProblem + MixedConstraintSystem + HolisticNewtonSolver
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <algorithm>
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

constexpr double kDt             = 1e-4;
constexpr int kNumStepsDefault   = 2500;
constexpr int kExportIntervalDef = 25;
constexpr int kPullRampSteps     = 400;

constexpr int kNumFingerBlocks = 4;

constexpr double kBlockLength = 0.025;
constexpr double kBlockWidth  = 0.020;
constexpr double kBlockHeight = 0.016;

constexpr double kTendonLength    = 0.125;
constexpr double kTendonWidth     = 0.005;
constexpr double kTendonThickness = 0.003;

constexpr double kJointOffset        = 0.001;
constexpr double kFaceTolerance      = 1e-10;
constexpr double kBasePatchTolerance = 1e-10;
constexpr double kHingeZ             = 2.5e-4;
constexpr double kGuideY             = 0.5 * kBlockWidth;
constexpr double kGuideZ             = 0.014;
constexpr double kDistalAttachX      = 0.0;

constexpr double kTendonPullForce = 0.8;

const SolidMaterialProperties kFingerMaterial =
    SolidMaterialProperties::SVK(1.0e7,   // E
                                 0.32,    // nu
                                 1200.0,  // rho0
                                 1.0e3,   // eta_damp
                                 1.0e3    // lambda_damp
    );

const SolidMaterialProperties kTendonMaterial =
    SolidMaterialProperties::SVK(5.0e6,  // E
                                 0.30,   // nu
                                 500.0,  // rho0
                                 5.0e2,  // eta_damp
                                 5.0e2   // lambda_damp
    );

std::vector<int> SelectNodesOnPlane(const Eigen::MatrixXd& all_nodes,
                                    const ANCFCPUUtils::MeshInstance& instance,
                                    int axis, double target, double tol) {
  std::vector<int> nodes;
  for (int local_node = 0; local_node < instance.num_nodes; ++local_node) {
    const int global_node = instance.node_offset + local_node;
    if (std::abs(all_nodes(global_node, axis) - target) <= tol) {
      nodes.push_back(global_node);
    }
  }
  return nodes;
}

std::vector<int> SelectANCF3243NodesOnPlane(
    const ANCFCPUUtils::ANCF3243Mesh& mesh, int axis, double target,
    double tol) {
  std::vector<int> nodes;
  nodes.reserve(static_cast<size_t>(mesh.n_nodes));
  for (int nid = 0; nid < mesh.n_nodes; ++nid) {
    const int coef     = 4 * nid;
    const double value = (axis == 0)
                             ? mesh.x12(coef)
                             : ((axis == 1) ? mesh.y12(coef) : mesh.z12(coef));
    if (std::abs(value - target) <= tol) {
      nodes.push_back(nid);
    }
  }
  return nodes;
}

void TranslateANCF3243Mesh(ANCFCPUUtils::ANCF3243Mesh* mesh, double dx,
                           double dy, double dz) {
  if (mesh == nullptr) {
    return;
  }
  for (int nid = 0; nid < mesh->n_nodes; ++nid) {
    const int base = 4 * nid;
    mesh->x12(base + 0) += dx;
    mesh->y12(base + 0) += dy;
    mesh->z12(base + 0) += dz;
  }
}

Eigen::MatrixXd ExtractLocalNodes(const Eigen::MatrixXd& global_nodes,
                                  const ANCFCPUUtils::MeshInstance& instance) {
  return global_nodes.middleRows(instance.node_offset, instance.num_nodes);
}

Eigen::MatrixXi ExtractLocalElements(
    const Eigen::MatrixXi& global_elements,
    const ANCFCPUUtils::MeshInstance& instance) {
  Eigen::MatrixXi local = global_elements.middleRows(instance.element_offset,
                                                     instance.num_elements);
  local.array() -= instance.node_offset;
  return local;
}

Eigen::VectorXd ExtractAxis(const Eigen::MatrixXd& nodes, int axis) {
  Eigen::VectorXd values(nodes.rows());
  for (int i = 0; i < nodes.rows(); ++i) {
    values(i) = nodes(i, axis);
  }
  return values;
}

Eigen::VectorXd ComputeANCF3243ElementLengths(
    const ANCFCPUUtils::ANCF3243Mesh& mesh) {
  Eigen::VectorXd lengths(mesh.n_elements);
  for (int elem = 0; elem < mesh.n_elements; ++elem) {
    const int n0 = mesh.element_connectivity(elem, 0);
    const int n1 = mesh.element_connectivity(elem, 1);
    const int i0 = 4 * n0;
    const int i1 = 4 * n1;
    const Eigen::Vector3d p0(mesh.x12(i0), mesh.y12(i0), mesh.z12(i0));
    const Eigen::Vector3d p1(mesh.x12(i1), mesh.y12(i1), mesh.z12(i1));
    lengths(elem) = (p1 - p0).norm();
  }
  return lengths;
}

Eigen::Vector3d ComputeCurrentCoefficientCentroid(
    const std::vector<int>& global_coef_indices, const Eigen::VectorXd& x,
    const Eigen::VectorXd& y, const Eigen::VectorXd& z) {
  if (global_coef_indices.empty()) {
    throw std::runtime_error(
        "Cannot compute centroid of an empty coefficient set");
  }

  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  for (int coef : global_coef_indices) {
    center += Eigen::Vector3d(x(coef), y(coef), z(coef));
  }
  return center / static_cast<double>(global_coef_indices.size());
}

Eigen::Vector3d EvaluateCurrentPointPosition(
    const MixedConstraintPointBinding& point, const Eigen::VectorXd& x,
    const Eigen::VectorXd& y, const Eigen::VectorXd& z) {
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  for (int i = 0; i < point.count; ++i) {
    const int coef      = point.coef_indices[i];
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
  HANDLE_ERROR(
      cudaMemcpy(x->data(), state.d_x12,
                 static_cast<size_t>(state.total_coef) * sizeof(double),
                 cudaMemcpyDeviceToHost));
  HANDLE_ERROR(
      cudaMemcpy(y->data(), state.d_y12,
                 static_cast<size_t>(state.total_coef) * sizeof(double),
                 cudaMemcpyDeviceToHost));
  HANDLE_ERROR(
      cudaMemcpy(z->data(), state.d_z12,
                 static_cast<size_t>(state.total_coef) * sizeof(double),
                 cudaMemcpyDeviceToHost));
}

Eigen::VectorXd ExtractBlockCoefficients(const Eigen::VectorXd& values,
                                         int coef_offset, int coef_count) {
  return values.segment(coef_offset, coef_count);
}

void AddNodeToWorldCD(MixedConstraintSystem* constraints, int global_coef,
                      const Eigen::Vector3d& world_point) {
  const MixedConstraintPointBinding point =
      constraints->MakeCoefficientBinding(global_coef);
  constraints->AddPointToWorldCDAxis(point, 0, world_point.x());
  constraints->AddPointToWorldCDAxis(point, 1, world_point.y());
  constraints->AddPointToWorldCDAxis(point, 2, world_point.z());
}

void AddFingerBlockRevoluteJoint(MixedConstraintSystem* constraints,
                                 int block_a, int block_b,
                                 const Eigen::Vector3d& hinge_point,
                                 double offset, double dp1_weight) {
  const Eigen::Vector3d hinge_axis = -Eigen::Vector3d::UnitY();
  const MixedConstraintPointBinding p =
      constraints->LocateReferencePoint(block_a, hinge_point);
  const MixedConstraintPointBinding r =
      constraints->LocateReferencePoint(block_b, hinge_point);
  const MixedConstraintPointBinding q = constraints->LocateReferencePoint(
      block_a, hinge_point + offset * hinge_axis);

  // For the finger layout, the child block always lies in +x from the hinge,
  // so these two interior directions remain safely inside the child body.
  const MixedConstraintPointBinding s = constraints->LocateReferencePoint(
      block_b, hinge_point + offset * Eigen::Vector3d::UnitX());
  const MixedConstraintPointBinding t = constraints->LocateReferencePoint(
      block_b, hinge_point + offset * Eigen::Vector3d::UnitZ());
  constraints->AddRevoluteJoint(p, q, r, s, t, 0.0, 0.0, dp1_weight);
}

void AddDistalBeamTipFixedJoint(MixedConstraintSystem* constraints,
                                int tendon_block, int distal_block,
                                const Eigen::Vector3d& joint_point,
                                double offset, double dp1_weight) {
  const Eigen::Vector3d beam_axis_1  = Eigen::Vector3d::UnitX();
  const Eigen::Vector3d beam_axis_2  = Eigen::Vector3d::UnitY();
  const Eigen::Vector3d block_axis_1 = Eigen::Vector3d::UnitX();
  const Eigen::Vector3d block_axis_2 = Eigen::Vector3d::UnitY();

  const MixedConstraintPointBinding p =
      constraints->LocateReferencePoint(tendon_block, joint_point);
  const MixedConstraintPointBinding r =
      constraints->LocateReferencePoint(distal_block, joint_point);
  const MixedConstraintPointBinding q = constraints->LocateReferencePoint(
      tendon_block, joint_point + offset * beam_axis_1);
  const MixedConstraintPointBinding w = constraints->LocateReferencePoint(
      tendon_block, joint_point + offset * beam_axis_2);
  const MixedConstraintPointBinding s = constraints->LocateReferencePoint(
      distal_block, joint_point + offset * block_axis_1);
  const MixedConstraintPointBinding t = constraints->LocateReferencePoint(
      distal_block, joint_point + offset * block_axis_2);

  const double f1 = std::pow(offset, 2) * beam_axis_1.dot(block_axis_1);
  const double f2 = std::pow(offset, 2) * beam_axis_1.dot(block_axis_2);
  const double f3 = std::pow(offset, 2) * beam_axis_2.dot(block_axis_2);
  constraints->AddFixedJoint(p, q, w, r, s, t, f1, f2, f3, dp1_weight);
}

std::vector<int> ToLocalNodeIndices(const std::vector<int>& global_nodes,
                                    int node_offset, int node_count) {
  std::vector<int> local_nodes;
  local_nodes.reserve(global_nodes.size());
  for (int global_node : global_nodes) {
    const int local_node = global_node - node_offset;
    if (local_node < 0 || local_node >= node_count) {
      throw std::out_of_range(
          "Global node does not belong to the requested T10 block");
    }
    local_nodes.push_back(local_node);
  }
  return local_nodes;
}

std::vector<int> MakeGlobalCoefIndicesFromNodes(
    const std::vector<int>& local_nodes, int coef_offset) {
  std::vector<int> global_coef_indices;
  global_coef_indices.reserve(local_nodes.size());
  for (int local_node : local_nodes) {
    global_coef_indices.push_back(coef_offset + local_node);
  }
  return global_coef_indices;
}

std::vector<int> MakeGlobalPosCoefIndicesFromANCFNodes(
    const std::vector<int>& local_nodes, int coef_offset) {
  std::vector<int> global_coef_indices;
  global_coef_indices.reserve(local_nodes.size());
  for (int local_node : local_nodes) {
    global_coef_indices.push_back(coef_offset + 4 * local_node);
  }
  return global_coef_indices;
}

struct GeneralizedForceWeight {
  int global_coef = -1;
  double weight   = 0.0;
};

Eigen::Matrix<double, Quadrature::N_SHAPE_3243, 1> EvaluateANCF3243Shape(
    double xi, double eta, double zeta, double length, double width,
    double height, const Eigen::Matrix<double, 8, 8>& b_inv) {
  const double u = 0.5 * length * xi;
  const double v = 0.5 * width * eta;
  const double w = 0.5 * height * zeta;

  Eigen::Matrix<double, Quadrature::N_SHAPE_3243, 1> basis;
  basis << 1.0, u, v, w, u * v, u * w, u * u, u * u * u;
  return b_inv * basis;
}

Eigen::Matrix<double, Quadrature::N_SHAPE_3243, 3>
EvaluateANCF3243ShapeDerivatives(double xi, double eta, double zeta,
                                 double length, double width, double height,
                                 const Eigen::Matrix<double, 8, 8>& b_inv) {
  Eigen::Matrix<double, Quadrature::N_SHAPE_3243, 1> db_dxi;
  Eigen::Matrix<double, Quadrature::N_SHAPE_3243, 1> db_deta;
  Eigen::Matrix<double, Quadrature::N_SHAPE_3243, 1> db_dzeta;

  db_dxi << 0.0, length / 2.0, 0.0, 0.0, (length * width / 4.0) * eta,
      (length * height / 4.0) * zeta, (length * length / 2.0) * xi,
      (3.0 * length * length * length / 8.0) * xi * xi;
  db_deta << 0.0, 0.0, width / 2.0, 0.0, (length * width / 4.0) * xi, 0.0, 0.0,
      0.0;
  db_dzeta << 0.0, 0.0, 0.0, height / 2.0, 0.0, (length * height / 4.0) * xi,
      0.0, 0.0;

  Eigen::Matrix<double, Quadrature::N_SHAPE_3243, 3> deriv;
  deriv.col(0) = b_inv * db_dxi;
  deriv.col(1) = b_inv * db_deta;
  deriv.col(2) = b_inv * db_dzeta;
  return deriv;
}

Eigen::Matrix<double, Quadrature::N_SHAPE_3243, 3>
GatherANCF3243ElementReferenceCoefficients(
    const ANCFCPUUtils::ANCF3243Mesh& mesh, int elem_idx) {
  Eigen::Matrix<double, Quadrature::N_SHAPE_3243, 3> coeffs;
  for (int local_coef = 0; local_coef < Quadrature::N_SHAPE_3243;
       ++local_coef) {
    const int node_local  = (local_coef < 4) ? 0 : 1;
    const int dof_local   = local_coef % 4;
    const int node_global = mesh.element_connectivity(elem_idx, node_local);
    const int coef_idx    = 4 * node_global + dof_local;
    coeffs(local_coef, 0) = mesh.x12(coef_idx);
    coeffs(local_coef, 1) = mesh.y12(coef_idx);
    coeffs(local_coef, 2) = mesh.z12(coef_idx);
  }
  return coeffs;
}

std::vector<GeneralizedForceWeight> BuildANCF3243EndFaceTractionWeights(
    const ANCFCPUUtils::ANCF3243Mesh& mesh, int elem_idx, int coef_offset,
    bool use_max_x_face) {
  if (elem_idx < 0 || elem_idx >= mesh.n_elements) {
    throw std::out_of_range("ANCF3243 end-face traction element out of range");
  }

  const int node0 = mesh.element_connectivity(elem_idx, 0);
  const int node1 = mesh.element_connectivity(elem_idx, 1);
  const int i0    = 4 * node0;
  const int i1    = 4 * node1;

  const Eigen::Vector3d p0(mesh.x12(i0), mesh.y12(i0), mesh.z12(i0));
  const Eigen::Vector3d p1(mesh.x12(i1), mesh.y12(i1), mesh.z12(i1));
  const double length = (p1 - p0).norm();
  if (length <= 0.0) {
    throw std::runtime_error(
        "ANCF3243 end-face traction: invalid element length");
  }

  Eigen::MatrixXd b_inv_dynamic;
  ANCFCPUUtils::ANCF3243_B12_matrix(length, kTendonWidth, kTendonThickness,
                                    b_inv_dynamic, Quadrature::N_SHAPE_3243);
  const Eigen::Matrix<double, 8, 8> b_inv = b_inv_dynamic;
  const auto coeffs =
      GatherANCF3243ElementReferenceCoefficients(mesh, elem_idx);

  const double xi              = use_max_x_face ? 1.0 : -1.0;
  constexpr double kGauss2[2]  = {-0.5773502691896257, 0.5773502691896257};
  constexpr double kWeight2[2] = {1.0, 1.0};

  Eigen::Matrix<double, Quadrature::N_SHAPE_3243, 1> integrated_weights =
      Eigen::Matrix<double, Quadrature::N_SHAPE_3243, 1>::Zero();
  double face_area = 0.0;

  for (int eta_q = 0; eta_q < 2; ++eta_q) {
    for (int zeta_q = 0; zeta_q < 2; ++zeta_q) {
      const double eta  = kGauss2[eta_q];
      const double zeta = kGauss2[zeta_q];
      const auto shape  = EvaluateANCF3243Shape(
          xi, eta, zeta, length, kTendonWidth, kTendonThickness, b_inv);
      const auto deriv = EvaluateANCF3243ShapeDerivatives(
          xi, eta, zeta, length, kTendonWidth, kTendonThickness, b_inv);

      Eigen::Vector3d dx_deta  = Eigen::Vector3d::Zero();
      Eigen::Vector3d dx_dzeta = Eigen::Vector3d::Zero();
      for (int i = 0; i < Quadrature::N_SHAPE_3243; ++i) {
        dx_deta += coeffs.row(i).transpose() * deriv(i, 1);
        dx_dzeta += coeffs.row(i).transpose() * deriv(i, 2);
      }

      const double d_area =
          dx_deta.cross(dx_dzeta).norm() * kWeight2[eta_q] * kWeight2[zeta_q];
      integrated_weights += shape * d_area;
      face_area += d_area;
    }
  }

  if (face_area <= 0.0) {
    throw std::runtime_error("ANCF3243 end-face traction: invalid face area");
  }

  std::vector<GeneralizedForceWeight> weights;
  weights.reserve(Quadrature::N_SHAPE_3243);
  for (int i = 0; i < Quadrature::N_SHAPE_3243; ++i) {
    const int node_local  = (i < 4) ? 0 : 1;
    const int dof_local   = i % 4;
    const int node_global = mesh.element_connectivity(elem_idx, node_local);
    weights.push_back({coef_offset + 4 * node_global + dof_local,
                       integrated_weights(i) / face_area});
  }
  return weights;
}

void AddWeightedForce(Eigen::VectorXd* h_f_ext,
                      const std::vector<GeneralizedForceWeight>& weights,
                      const Eigen::Vector3d& total_force) {
  for (const GeneralizedForceWeight& entry : weights) {
    (*h_f_ext)(3 * entry.global_coef + 0) += entry.weight * total_force.x();
    (*h_f_ext)(3 * entry.global_coef + 1) += entry.weight * total_force.y();
    (*h_f_ext)(3 * entry.global_coef + 2) += entry.weight * total_force.z();
  }
}

std::string MakeOutputPath(const std::string& body_name, int frame) {
  std::ostringstream oss;
  oss << "output/engineering_joint/finger_tendon_curl_" << body_name << "_"
      << std::setw(6) << std::setfill('0') << frame << ".vtu";
  return oss.str();
}

void WriteTendonOutputVTU(const ANCFCPUUtils::ANCF3243Mesh& mesh,
                          const Eigen::VectorXd& x_global,
                          const Eigen::VectorXd& y_global,
                          const Eigen::VectorXd& z_global, int coef_offset,
                          int coef_count, int frame) {
  const Eigen::VectorXd x_local =
      ExtractBlockCoefficients(x_global, coef_offset, coef_count);
  const Eigen::VectorXd y_local =
      ExtractBlockCoefficients(y_global, coef_offset, coef_count);
  const Eigen::VectorXd z_local =
      ExtractBlockCoefficients(z_global, coef_offset, coef_count);

  ANCFCPUUtils::VisualizationUtils::ExportANCF3243ToVTU(
      x_local, y_local, z_local, mesh.element_connectivity, kTendonWidth,
      kTendonThickness, MakeOutputPath("tendon", frame));
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
  std::cout << "Mixed T10 + ANCF3243 Finger Curl Demo\n";
  std::cout << "========================================\n";
  std::cout << "steps=" << max_steps << " export_interval=" << export_interval
            << "\n";

  std::filesystem::create_directories("output/engineering_joint");

  const std::string t10_mesh_dir = "data/meshes/T10/finger";
  const std::string block_prefix = t10_mesh_dir + "/finger_block.1";
  const std::string tendon_mesh_path =
      "data/meshes/ANCF3243/finger_tendon.ancf3243mesh";

  ANCFCPUUtils::MeshManager mesh_manager;
  std::vector<int> block_mesh_ids;
  block_mesh_ids.reserve(kNumFingerBlocks);
  for (int block_idx = 0; block_idx < kNumFingerBlocks; ++block_idx) {
    const int mesh_id = mesh_manager.LoadMesh(
        block_prefix + ".node", block_prefix + ".ele",
        "finger_block_" + std::to_string(block_idx), kFingerMaterial);
    if (mesh_id < 0) {
      std::cerr << "Failed to load finger block mesh from " << block_prefix
                << std::endl;
      return 1;
    }
    mesh_manager.TranslateMesh(mesh_id, block_idx * kBlockLength, 0.0, 0.0);
    block_mesh_ids.push_back(mesh_id);
  }

  ANCFCPUUtils::ANCF3243Mesh tendon_mesh;
  std::string tendon_err;
  if (!ANCFCPUUtils::ReadANCF3243MeshFromFile(tendon_mesh_path, tendon_mesh,
                                              &tendon_err)) {
    std::cerr << tendon_err << std::endl;
    return 1;
  }
  TranslateANCF3243Mesh(&tendon_mesh, 0.0, kGuideY - 0.5 * kTendonWidth,
                        kGuideZ - 0.5 * kTendonThickness);

  std::vector<ANCFCPUUtils::MeshInstance> block_instances;
  block_instances.reserve(kNumFingerBlocks);
  for (int mesh_id : block_mesh_ids) {
    block_instances.push_back(mesh_manager.GetMeshInstance(mesh_id));
  }
  const auto& base_instance = block_instances.back();

  const Eigen::MatrixXd& all_nodes = mesh_manager.GetAllNodes();
  const Eigen::MatrixXi& all_elems = mesh_manager.GetAllElements();

  const double base_x_max                 = kNumFingerBlocks * kBlockLength;
  const double tendon_x_max               = kTendonLength;
  const std::vector<int> fixed_base_nodes = SelectNodesOnPlane(
      all_nodes, base_instance, 0, base_x_max, kBasePatchTolerance);
  const std::vector<int> tendon_pull_nodes =
      SelectANCF3243NodesOnPlane(tendon_mesh, 0, tendon_x_max, kFaceTolerance);

  if (fixed_base_nodes.empty()) {
    throw std::runtime_error("Failed to identify the grounded base-face nodes");
  }
  if (tendon_pull_nodes.empty()) {
    throw std::runtime_error("Failed to identify the tendon pull node(s)");
  }

  std::cout << "finger blocks: " << kNumFingerBlocks << "\n";
  for (int block_idx = 0; block_idx < kNumFingerBlocks; ++block_idx) {
    const auto& inst = block_instances[block_idx];
    std::cout << "  block[" << block_idx << "]: " << inst.num_nodes
              << " nodes, " << inst.num_elements << " elements"
              << " x=[" << block_idx * kBlockLength << ", "
              << (block_idx + 1) * kBlockLength << "]\n";
  }
  std::cout << "tendon beam: " << tendon_mesh.n_nodes << " nodes, "
            << tendon_mesh.n_elements << " elements"
            << " x=[0, " << kTendonLength << "] y=["
            << (kGuideY - 0.5 * kTendonWidth) << ", "
            << (kGuideY + 0.5 * kTendonWidth) << "] z=["
            << (kGuideZ - 0.5 * kTendonThickness) << ", "
            << (kGuideZ + 0.5 * kTendonThickness) << "]\n";
  std::cout << "fixed base nodes: " << fixed_base_nodes.size() << "\n";
  std::cout << "tendon pull nodes: " << tendon_pull_nodes.size() << "\n";

  std::vector<std::unique_ptr<GPU_FEAT10_Data>> finger_blocks;
  finger_blocks.reserve(kNumFingerBlocks);
  for (int block_idx = 0; block_idx < kNumFingerBlocks; ++block_idx) {
    const auto& inst                  = block_instances[block_idx];
    const Eigen::MatrixXd block_nodes = ExtractLocalNodes(all_nodes, inst);
    const Eigen::MatrixXi block_elems = ExtractLocalElements(all_elems, inst);

    auto data =
        std::make_unique<GPU_FEAT10_Data>(inst.num_elements, inst.num_nodes);
    data->Initialize();
    data->Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y,
                Quadrature::tet5pt_z, Quadrature::tet5pt_weights,
                ExtractAxis(block_nodes, 0), ExtractAxis(block_nodes, 1),
                ExtractAxis(block_nodes, 2), block_elems);
    data->ApplyMaterial(kFingerMaterial);
    data->CalcDnDuPre();
    data->CalcMassMatrix();
    finger_blocks.push_back(std::move(data));
  }

  GPU_ANCF3243_Data tendon_data(tendon_mesh.n_nodes, tendon_mesh.n_elements);
  tendon_data.Initialize();
  tendon_data.SetExternalForce(
      Eigen::VectorXd::Zero(4 * tendon_mesh.n_nodes * 3));

  const Eigen::VectorXd tendon_length =
      ComputeANCF3243ElementLengths(tendon_mesh);
  const Eigen::VectorXd tendon_width =
      Eigen::VectorXd::Constant(tendon_mesh.n_elements, kTendonWidth);
  const Eigen::VectorXd tendon_height =
      Eigen::VectorXd::Constant(tendon_mesh.n_elements, kTendonThickness);

  tendon_data.Setup(
      tendon_length, tendon_width, tendon_height, Quadrature::gauss_xi_m_6,
      Quadrature::gauss_xi_3, Quadrature::gauss_eta_2, Quadrature::gauss_zeta_2,
      Quadrature::weight_xi_m_6, Quadrature::weight_xi_3,
      Quadrature::weight_eta_2, Quadrature::weight_zeta_2, tendon_mesh.x12,
      tendon_mesh.y12, tendon_mesh.z12, tendon_mesh.element_connectivity);
  tendon_data.SetDensity(kTendonMaterial.rho0);
  tendon_data.SetDamping(kTendonMaterial.eta_damp, kTendonMaterial.lambda_damp);
  tendon_data.SetSVK(kTendonMaterial.E, kTendonMaterial.nu);
  tendon_data.CalcDsDuPre();
  tendon_data.CalcMassMatrix();

  FEMultiElementProblem problem;
  std::vector<int> finger_block_ids;
  finger_block_ids.reserve(kNumFingerBlocks);
  for (const auto& block : finger_blocks) {
    finger_block_ids.push_back(problem.AddElementBlock(block.get(), TYPE_T10));
  }
  const int tendon_block = problem.AddElementBlock(&tendon_data, TYPE_3243);
  problem.Finalize();

  FEStateBuffer& state = problem.GetStateBuffer();

  const std::vector<int> fixed_base_nodes_local = ToLocalNodeIndices(
      fixed_base_nodes, base_instance.node_offset, base_instance.num_nodes);
  const int base_block_id = finger_block_ids.back();
  const int base_coef_offset =
      state.blocks[static_cast<size_t>(base_block_id)].coef_offset;
  const std::vector<int> fixed_base_coef_indices =
      MakeGlobalCoefIndicesFromNodes(fixed_base_nodes_local, base_coef_offset);

  const int distal_block_id = finger_block_ids.front();
  const int tendon_coef_offset =
      state.blocks[static_cast<size_t>(tendon_block)].coef_offset;
  const int tendon_coef_count =
      state.blocks[static_cast<size_t>(tendon_block)].coef_count;
  const std::vector<int> tendon_pull_position_coef_indices =
      MakeGlobalPosCoefIndicesFromANCFNodes(tendon_pull_nodes,
                                            tendon_coef_offset);
  const std::vector<GeneralizedForceWeight> tendon_pull_force_weights =
      BuildANCF3243EndFaceTractionWeights(
          tendon_mesh, tendon_mesh.n_elements - 1, tendon_coef_offset, true);

  MixedConstraintSystem constraints(&problem);
  for (size_t i = 0; i < fixed_base_nodes_local.size(); ++i) {
    const int global_node = fixed_base_nodes[static_cast<size_t>(i)];
    const int global_coef = fixed_base_coef_indices[static_cast<size_t>(i)];
    AddNodeToWorldCD(&constraints, global_coef,
                     all_nodes.row(global_node).transpose());
  }

  for (int joint_idx = 0; joint_idx < kNumFingerBlocks - 1; ++joint_idx) {
    const double hinge_x = (joint_idx + 1) * kBlockLength;
    const Eigen::Vector3d hinge_point(hinge_x, kGuideY, kHingeZ);
    AddFingerBlockRevoluteJoint(&constraints, finger_block_ids[joint_idx],
                                finger_block_ids[joint_idx + 1], hinge_point,
                                kJointOffset, 1.0);
  }

  for (int block_idx = 0; block_idx < kNumFingerBlocks; ++block_idx) {
    const double guide_x = (block_idx + 0.5) * kBlockLength;
    const Eigen::Vector3d guide_point(guide_x, kGuideY, kGuideZ);
    // For mixed beam-T10 cylindrical joints, place the ANCF3243 tendon on the
    // body-b side so the DP2 offset-collinearity rows follow Appendix A.9.4.
    constraints.AddCylindricalJoint(
        tendon_block, finger_block_ids[block_idx], guide_point, guide_point,
        Eigen::Vector3d::UnitX(), kJointOffset, 1.0, 1.0);
  }

  const Eigen::Vector3d distal_attach_point(kDistalAttachX, kGuideY, kGuideZ);
  // The tendon tip weld lies on the x=0 boundary of both bodies, so build the
  // fixed joint from explicit inward points instead of the generic helper,
  // which may place one of its orientation points outside the distal block.
  AddDistalBeamTipFixedJoint(&constraints, tendon_block, distal_block_id,
                             distal_attach_point, kJointOffset, 1.0);

  const MixedConstraintPointBinding tip_reference =
      constraints.LocateReferencePoint(
          distal_block_id,
          Eigen::Vector3d(0.001, kGuideY, kBlockHeight - 0.001));
  const MixedConstraintPointBinding tail_reference =
      constraints.LocateReferencePoint(
          tendon_block,
          Eigen::Vector3d(kTendonLength - 0.001, kGuideY, kGuideZ));
  constraints.Finalize();

  HolisticNewtonParams params;
  params.inner_atol         = 1e-4;
  params.inner_rtol         = 1e-4;
  params.outer_tol          = 1e-8;
  params.rho                = 1e8;
  params.max_outer          = 8;
  params.max_inner          = 12;
  params.time_step          = kDt;
  params.enable_line_search = false;

  HolisticNewtonSolver solver(&problem, &constraints);
  solver.SetParameters(&params);
  solver.Setup();

  Eigen::VectorXd x_curr, y_curr, z_curr;
  RetrieveUnifiedPositions(state, &x_curr, &y_curr, &z_curr);
  const Eigen::Vector3d tip_initial =
      EvaluateCurrentPointPosition(tip_reference, x_curr, y_curr, z_curr);
  const Eigen::Vector3d tail_initial =
      EvaluateCurrentPointPosition(tail_reference, x_curr, y_curr, z_curr);
  const Eigen::Vector3d pull_face_initial = ComputeCurrentCoefficientCentroid(
      tendon_pull_position_coef_indices, x_curr, y_curr, z_curr);

  std::cout << "constraints: " << constraints.num_constraints() << "\n";
  std::cout << "initial fingertip: [" << tip_initial.transpose() << "]\n";
  std::cout << "initial tendon tail point: [" << tail_initial.transpose()
            << "]\n";
  std::cout << "initial tendon pull-face centroid: ["
            << pull_face_initial.transpose() << "]\n";

  for (int block_idx = 0; block_idx < kNumFingerBlocks; ++block_idx) {
    finger_blocks[static_cast<size_t>(block_idx)]->WriteOutputVTU(
        MakeOutputPath("block_" + std::to_string(block_idx), 0));
  }
  WriteTendonOutputVTU(tendon_mesh, x_curr, y_curr, z_curr, tendon_coef_offset,
                       tendon_coef_count, 0);

  int output_frame = 1;
  for (int step = 1; step <= max_steps; ++step) {
    const double ramp =
        std::min(1.0, static_cast<double>(step) / kPullRampSteps);
    Eigen::VectorXd step_f_ext = Eigen::VectorXd::Zero(problem.GetTotalDofs());
    AddWeightedForce(&step_f_ext, tendon_pull_force_weights,
                     ramp * kTendonPullForce * Eigen::Vector3d::UnitX());
    HANDLE_ERROR(
        cudaMemcpy(state.d_f_ext, step_f_ext.data(),
                   static_cast<size_t>(problem.GetTotalDofs()) * sizeof(double),
                   cudaMemcpyHostToDevice));

    solver.Solve();
    RetrieveUnifiedPositions(state, &x_curr, &y_curr, &z_curr);

    if (step % export_interval == 0 || step == max_steps) {
      const Eigen::Vector3d tip_current =
          EvaluateCurrentPointPosition(tip_reference, x_curr, y_curr, z_curr);
      const Eigen::Vector3d tail_current =
          EvaluateCurrentPointPosition(tail_reference, x_curr, y_curr, z_curr);
      const Eigen::Vector3d pull_face_current =
          ComputeCurrentCoefficientCentroid(tendon_pull_position_coef_indices,
                                            x_curr, y_curr, z_curr);

      Eigen::VectorXd constraint_values(constraints.num_constraints());
      HANDLE_ERROR(cudaMemcpy(
          constraint_values.data(), constraints.GetConstraintDevicePtr(),
          static_cast<size_t>(constraints.num_constraints()) * sizeof(double),
          cudaMemcpyDeviceToHost));

      std::cout << "step " << step << " ramp=" << ramp
                << " constraint_norm=" << constraint_values.norm() << " tip=["
                << tip_current.transpose() << "]"
                << " tip_disp=[" << (tip_current - tip_initial).transpose()
                << "]"
                << " tail_disp=[" << (tail_current - tail_initial).transpose()
                << "]"
                << " pull_face_disp=["
                << (pull_face_current - pull_face_initial).transpose() << "]\n";

      for (int block_idx = 0; block_idx < kNumFingerBlocks; ++block_idx) {
        finger_blocks[static_cast<size_t>(block_idx)]->WriteOutputVTU(
            MakeOutputPath("block_" + std::to_string(block_idx), output_frame));
      }
      WriteTendonOutputVTU(tendon_mesh, x_curr, y_curr, z_curr,
                           tendon_coef_offset, tendon_coef_count, output_frame);
      ++output_frame;
    }
  }

  for (auto& block : finger_blocks) {
    block->Destroy();
  }
  tendon_data.Destroy();

  std::cout << "Done. Output written to output/engineering_joint/\n";
  return 0;
}
