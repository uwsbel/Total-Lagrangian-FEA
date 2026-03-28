/**
 * FEAT10 Tendon-Driven Finger Curl Demo
 *
 * Builds a four-block finger from the T10 finger_block mesh. The proximal
 * block is grounded by fixing its back face, the remaining blocks are linked
 * by revolute joints along a near-bottom hinge line, and a single tendon body
 * is routed through one cylindrical guide per block. The tendon is welded to
 * the distal block and pulled from its proximal tail to curl the finger.
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../lib_src/elements/FEAT10Data.cuh"
#include "../../lib_src/solvers/FEAT10ConstraintManager.h"
#include "../../lib_src/solvers/SyncedNewton.cuh"
#include "../../lib_utils/cpu_utils.h"
#include "../../lib_utils/mesh_manager.h"
#include "../../lib_utils/quadrature_utils.h"

namespace {

constexpr double kDt             = 1e-4;
constexpr int kNumStepsDefault   = 2500;
constexpr int kExportIntervalDef = 25;
constexpr int kPullRampSteps     = 400;

constexpr int kNumFingerBlocks = 4;

constexpr double kBlockLength = 0.025;
constexpr double kBlockWidth  = 0.020;
constexpr double kBlockHeight = 0.016;

constexpr double kTendonLength    = 0.150;
constexpr double kTendonWidth     = 0.005;
constexpr double kTendonThickness = 0.003;

constexpr double kJointOffset        = 0.001;
constexpr double kFaceTolerance      = 1e-10;
constexpr double kBasePatchTolerance = 1e-10;
constexpr double kHingeZ             = 2.5e-4;
constexpr double kGuideY             = 0.5 * kBlockWidth;
constexpr double kGuideZ             = 0.014;
constexpr double kDistalAttachX      = 0.0025;

constexpr double kTendonPullForce = 0.8;

const SolidMaterialProperties kFingerMaterial =
    SolidMaterialProperties::SVK(5.0e6,   // E
                                 0.32,    // nu
                                 1200.0,  // rho0
                                 1.0e3,   // eta_damp
                                 1.0e3    // lambda_damp
    );

const SolidMaterialProperties kTendonMaterial =
    SolidMaterialProperties::SVK(5.0e7,   // E
                                 0.30,    // nu
                                 1100.0,  // rho0
                                 5.0e2,   // eta_damp
                                 5.0e2    // lambda_damp
    );

FEAT10ConstraintManager::ElementRange MakeElementRange(
    const ANCFCPUUtils::MeshInstance& instance) {
  return FEAT10ConstraintManager::ElementRange{
      instance.element_offset, instance.element_offset + instance.num_elements};
}

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

void AddDistributedForce(Eigen::VectorXd* h_f_ext, const std::vector<int>& nodes,
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

Eigen::Vector3d ComputeCurrentNodeCentroid(const std::vector<int>& nodes,
                                           const Eigen::VectorXd& x,
                                           const Eigen::VectorXd& y,
                                           const Eigen::VectorXd& z) {
  if (nodes.empty()) {
    throw std::runtime_error("Cannot compute centroid of an empty node set");
  }

  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  for (int node : nodes) {
    center += Eigen::Vector3d(x(node), y(node), z(node));
  }
  return center / static_cast<double>(nodes.size());
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

std::string MakeOutputPath(int frame) {
  std::ostringstream oss;
  oss << "output/engineering_joint/finger_tendon_curl_" << std::setw(6)
      << std::setfill('0') << frame << ".vtu";
  return oss.str();
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
  std::cout << "FEAT10 Tendon-Driven Finger Curl Demo\n";
  std::cout << "========================================\n";
  std::cout << "steps=" << max_steps << " export_interval=" << export_interval
            << "\n";

  std::filesystem::create_directories("output/engineering_joint");

  const std::string mesh_dir = "data/meshes/T10/hand_simple";
  const std::string block_prefix = mesh_dir + "/finger_block.1";
  const std::string tendon_prefix = mesh_dir + "/tendon.1";

  ANCFCPUUtils::MeshManager mesh_manager;
  std::vector<int> block_mesh_ids;
  block_mesh_ids.reserve(kNumFingerBlocks);
  for (int block_idx = 0; block_idx < kNumFingerBlocks; ++block_idx) {
    const int mesh_id =
        mesh_manager.LoadMesh(block_prefix + ".node", block_prefix + ".ele",
                              "finger_block_" + std::to_string(block_idx),
                              kFingerMaterial);
    if (mesh_id < 0) {
      std::cerr << "Failed to load finger block mesh from " << block_prefix
                << std::endl;
      return 1;
    }
    mesh_manager.TranslateMesh(mesh_id, block_idx * kBlockLength, 0.0, 0.0);
    block_mesh_ids.push_back(mesh_id);
  }

  const int tendon_mesh =
      mesh_manager.LoadMesh(tendon_prefix + ".node", tendon_prefix + ".ele",
                            "tendon", kTendonMaterial);
  if (tendon_mesh < 0) {
    std::cerr << "Failed to load tendon mesh from " << tendon_prefix
              << std::endl;
    return 1;
  }
  mesh_manager.TranslateMesh(tendon_mesh, 0.0, kGuideY - 0.5 * kTendonWidth,
                             kGuideZ - 0.5 * kTendonThickness);

  std::vector<ANCFCPUUtils::MeshInstance> block_instances;
  block_instances.reserve(kNumFingerBlocks);
  for (int mesh_id : block_mesh_ids) {
    block_instances.push_back(mesh_manager.GetMeshInstance(mesh_id));
  }
  const auto& tendon_instance = mesh_manager.GetMeshInstance(tendon_mesh);
  const auto& base_instance   = block_instances.back();
  const auto& distal_instance = block_instances.front();

  const Eigen::MatrixXd& all_nodes = mesh_manager.GetAllNodes();
  const Eigen::MatrixXi& all_elems = mesh_manager.GetAllElements();
  const int n_nodes                = mesh_manager.GetTotalNodes();
  const int n_elems                = mesh_manager.GetTotalElements();

  const double base_x_max = (kNumFingerBlocks)*kBlockLength;
  const double tendon_x_max = kTendonLength;
  const std::vector<int> fixed_base_nodes = SelectNodesOnPlane(
      all_nodes, base_instance, 0, base_x_max, kBasePatchTolerance);
  const std::vector<int> tendon_pull_nodes = SelectNodesOnPlane(
      all_nodes, tendon_instance, 0, tendon_x_max, kFaceTolerance);

  if (fixed_base_nodes.empty()) {
    throw std::runtime_error("Failed to identify the grounded base-face nodes");
  }
  if (tendon_pull_nodes.empty()) {
    throw std::runtime_error("Failed to identify the tendon pull-face nodes");
  }

  std::cout << "finger blocks: " << kNumFingerBlocks << "\n";
  for (int block_idx = 0; block_idx < kNumFingerBlocks; ++block_idx) {
    const auto& inst = block_instances[block_idx];
    std::cout << "  block[" << block_idx << "]: " << inst.num_nodes
              << " nodes, " << inst.num_elements << " elements"
              << " x=[" << block_idx * kBlockLength << ", "
              << (block_idx + 1) * kBlockLength << "]\n";
  }
  std::cout << "tendon: " << tendon_instance.num_nodes << " nodes, "
            << tendon_instance.num_elements << " elements"
            << " x=[0, " << kTendonLength << "] y=["
            << (kGuideY - 0.5 * kTendonWidth) << ", "
            << (kGuideY + 0.5 * kTendonWidth) << "] z=["
            << (kGuideZ - 0.5 * kTendonThickness) << ", "
            << (kGuideZ + 0.5 * kTendonThickness) << "]\n";
  std::cout << "total: " << n_nodes << " nodes, " << n_elems << " elements\n";
  std::cout << "fixed base nodes: " << fixed_base_nodes.size() << "\n";
  std::cout << "tendon pull nodes: " << tendon_pull_nodes.size() << "\n";

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
  gpu_t10_data.SetExternalForce(Eigen::VectorXd::Zero(n_nodes * 3));

  FEAT10ConstraintManager constraint_manager(&gpu_t10_data);
  for (int node : fixed_base_nodes) {
    constraint_manager.AddNodeToWorldCD(node);
  }

  const Eigen::Vector3d hinge_axis = -Eigen::Vector3d::UnitY();
  for (int joint_idx = 0; joint_idx < kNumFingerBlocks - 1; ++joint_idx) {
    const double hinge_x = (joint_idx + 1) * kBlockLength;
    const Eigen::Vector3d hinge_point(hinge_x, kGuideY, kHingeZ);
    constraint_manager.AddRevoluteJoint(
        MakeElementRange(block_instances[joint_idx]),
        MakeElementRange(block_instances[joint_idx + 1]), hinge_point,
        hinge_axis, kJointOffset, 1.0);
  }

  for (int block_idx = 0; block_idx < kNumFingerBlocks; ++block_idx) {
    const double guide_x = (block_idx + 0.5) * kBlockLength;
    const Eigen::Vector3d guide_point(guide_x, kGuideY, kGuideZ);
    constraint_manager.AddCylindricalJoint(
        MakeElementRange(block_instances[block_idx]),
        MakeElementRange(tendon_instance), guide_point, guide_point,
        Eigen::Vector3d::UnitX(), kJointOffset, 1.0, 1.0);
  }

  const Eigen::Vector3d distal_attach_point(kDistalAttachX, kGuideY, kGuideZ);
  constraint_manager.AddFixedJoint(MakeElementRange(distal_instance),
                                   MakeElementRange(tendon_instance),
                                   distal_attach_point, kJointOffset, 1.0);

  const auto tip_reference = constraint_manager.LocateReferencePoint(
      Eigen::Vector3d(0.001, kGuideY, kBlockHeight - 0.001),
      MakeElementRange(distal_instance));
  const auto tail_reference = constraint_manager.LocateReferencePoint(
      Eigen::Vector3d(kTendonLength - 0.001, kGuideY, kGuideZ),
      MakeElementRange(tendon_instance));

  constraint_manager.Finalize();

  gpu_t10_data.CalcConstraintData();
  gpu_t10_data.ConvertToCSR_ConstraintJacT();
  gpu_t10_data.BuildConstraintJacobianCSR();
  gpu_t10_data.CalcP();
  gpu_t10_data.CalcInternalForce();

  SyncedNewtonParams params = {1e-4, 1e-4, 1e-7, 1e11, 8, 12, kDt, false};
  SyncedNewtonSolver solver(&gpu_t10_data, gpu_t10_data.get_n_constraint());
  solver.Setup();
  solver.SetParameters(&params);
  solver.AnalyzeHessianSparsity();
  solver.SetFixedSparsityPattern(true);

  std::cout << "constraints: " << gpu_t10_data.get_n_constraint() << "\n";
  std::cout << "writing initial frame to " << MakeOutputPath(0) << "\n";
  gpu_t10_data.WriteOutputVTU(MakeOutputPath(0));

  Eigen::VectorXd x_curr, y_curr, z_curr;
  gpu_t10_data.RetrievePositionToCPU(x_curr, y_curr, z_curr);
  const Eigen::Vector3d tip_initial =
      EvaluateCurrentPointPosition(tip_reference, all_elems, x_curr, y_curr,
                                   z_curr);
  const Eigen::Vector3d tail_initial =
      EvaluateCurrentPointPosition(tail_reference, all_elems, x_curr, y_curr,
                                   z_curr);
  const Eigen::Vector3d pull_face_initial =
      ComputeCurrentNodeCentroid(tendon_pull_nodes, x_curr, y_curr, z_curr);

  std::cout << "initial fingertip: [" << tip_initial.transpose() << "]\n";
  std::cout << "initial tendon tail point: [" << tail_initial.transpose()
            << "]\n";
  std::cout << "initial tendon pull-face centroid: ["
            << pull_face_initial.transpose() << "]\n";

  int output_frame = 1;
  for (int step = 1; step <= max_steps; ++step) {
    const double ramp =
        std::min(1.0, static_cast<double>(step) / kPullRampSteps);
    Eigen::VectorXd step_f_ext = Eigen::VectorXd::Zero(n_nodes * 3);
    AddDistributedForce(&step_f_ext, tendon_pull_nodes,
                        ramp * kTendonPullForce * Eigen::Vector3d::UnitX());
    gpu_t10_data.SetExternalForce(step_f_ext);

    solver.Solve();

    if (step % export_interval == 0 || step == max_steps) {
      gpu_t10_data.WriteOutputVTU(MakeOutputPath(output_frame));
      ++output_frame;

      gpu_t10_data.RetrievePositionToCPU(x_curr, y_curr, z_curr);
      const Eigen::Vector3d tip_current =
          EvaluateCurrentPointPosition(tip_reference, all_elems, x_curr, y_curr,
                                       z_curr);
      const Eigen::Vector3d tail_current =
          EvaluateCurrentPointPosition(tail_reference, all_elems, x_curr, y_curr,
                                       z_curr);
      const Eigen::Vector3d pull_face_current =
          ComputeCurrentNodeCentroid(tendon_pull_nodes, x_curr, y_curr, z_curr);

      std::cout << "step " << step << " ramp=" << ramp
                << " tip=[" << tip_current.transpose() << "]"
                << " tip_disp=[" << (tip_current - tip_initial).transpose()
                << "]"
                << " tail_disp=[" << (tail_current - tail_initial).transpose()
                << "]"
                << " pull_face_disp=["
                << (pull_face_current - pull_face_initial).transpose() << "]\n";
    }
  }

  gpu_t10_data.Destroy();
  std::cout << "Done. Output written to output/engineering_joint/\n";
  return 0;
}
