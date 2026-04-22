/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Ganesh Arivoli, Json Zhou
 * Email:   ganesh.arivoli@gmail.com, zzhou292@wisc.edu
 * File:    test_feat10_simple_hand_curl.cc
 * Brief:   FEAT10 Tendon-Driven Hand Curl Demo.
 *          Four fingers + thumb on a shared palm. Each finger: 3 block
 *          segments, revolute joints, tendon with cylindrical guides.
 *          Thumb at 45 deg from palm side. All tendons pulled to curl.
 *==============================================================
 *==============================================================*/

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
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

constexpr int kNumFingerBlocks = 3;

constexpr double kBlockLength = 0.025;
constexpr double kBlockWidth  = 0.020;
constexpr double kBlockHeight = 0.016;

constexpr double kFingerBaseX = -kNumFingerBlocks * kBlockLength;
constexpr double kPalmLength  = 3 * kBlockLength;
constexpr double kPalmBackX   = kPalmLength;
constexpr double kPalmGuideX  = 0.4 * kPalmLength;

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

struct FingerSpec {
  Eigen::Vector3d attach;
  double angle;
};

const FingerSpec kFingerSpecs[] = {
    {{0.0, 0.010, 0.0}, 0.0},
    {{0.0, 0.035, 0.0}, 0.0},
    {{0.0, 0.060, 0.0}, 0.0},
    {{0.0, 0.085, 0.0}, 0.0},
    {{0.05, 0.008, 0.0}, M_PI / 4},
};
constexpr int kNumFingers = sizeof(kFingerSpecs) / sizeof(kFingerSpecs[0]);

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

struct FingerData {
  std::vector<int> block_mesh_ids;
  int tendon_mesh_id;
  std::vector<ANCFCPUUtils::MeshInstance> block_instances;
  ANCFCPUUtils::MeshInstance tendon_instance;
  std::vector<int> tendon_pull_nodes;
  FEAT10ConstraintManager::ReferencePoint tip_reference;
  Eigen::Vector3d tip_initial;
};

Eigen::Matrix3d RotZ(double angle) {
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  Eigen::Matrix3d R;
  R << c, -s, 0,
       s,  c, 0,
       0,  0, 1;
  return R;
}

Eigen::Matrix4d MakeTransform(const Eigen::Matrix3d& R,
                               const Eigen::Vector3d& attach,
                               const Eigen::Vector3d& local_offset) {
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = R;
  T.block<3, 1>(0, 3) = attach + R * local_offset;
  return T;
}

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

std::vector<int> SelectNodesOnPlane(const Eigen::MatrixXd& all_nodes,
                                    const ANCFCPUUtils::MeshInstance& instance,
                                    const Eigen::Vector3d& point,
                                    const Eigen::Vector3d& normal,
                                    double tol) {
  std::vector<int> nodes;
  for (int local_node = 0; local_node < instance.num_nodes; ++local_node) {
    const int global_node = instance.node_offset + local_node;
    const Eigen::Vector3d pos(all_nodes(global_node, 0),
                              all_nodes(global_node, 1),
                              all_nodes(global_node, 2));
    if (std::abs((pos - point).dot(normal)) <= tol) {
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
  oss << "output/engineering_joint/simple_hand_curl_" << std::setw(6)
      << std::setfill('0') << frame << ".vtu";
  return oss.str();
}

void AddFinger(FEAT10ConstraintManager& cm,
               const FingerData& finger,
               const FEAT10ConstraintManager::ElementRange& palm_range,
               const Eigen::Matrix3d& R,
               const Eigen::Vector3d& attach) {
  const auto tendon_range = MakeElementRange(finger.tendon_instance);
  const Eigen::Vector3d hinge_axis = R * (-Eigen::Vector3d::UnitY());
  const Eigen::Vector3d guide_axis = R * Eigen::Vector3d::UnitX();

  auto to_world = [&](const Eigen::Vector3d& local) {
    return attach + R * local;
  };

  // Revolute joints between adjacent blocks
  for (int ji = 0; ji < kNumFingerBlocks - 1; ++ji) {
    const double hinge_x = kFingerBaseX + (ji + 1) * kBlockLength;
    cm.AddRevoluteJoint(
        MakeElementRange(finger.block_instances[ji]),
        MakeElementRange(finger.block_instances[ji + 1]),
        to_world({hinge_x, 0, kHingeZ}),
        hinge_axis, kJointOffset, 1.0);
  }

  // Revolute joint: proximal block to palm
  cm.AddRevoluteJoint(
      MakeElementRange(finger.block_instances.back()), palm_range,
      to_world({0, 0, kHingeZ}),
      hinge_axis, kJointOffset, 1.0);

  // Cylindrical guides for tendon through each block
  for (int bi = 0; bi < kNumFingerBlocks; ++bi) {
    const double guide_x = kFingerBaseX + (bi + 0.5) * kBlockLength;
    const Eigen::Vector3d gp = to_world({guide_x, 0, kGuideZ});
    cm.AddCylindricalJoint(
        MakeElementRange(finger.block_instances[bi]),
        tendon_range, gp, gp,
        guide_axis, kJointOffset, 1.0, 1.0);
  }

  // Cylindrical guide for tendon through palm
  const Eigen::Vector3d pg = to_world({kPalmGuideX, 0, kGuideZ});
  cm.AddCylindricalJoint(
      palm_range, tendon_range, pg, pg,
      guide_axis, kJointOffset, 1.0, 1.0);

  // Weld tendon to distal block
  cm.AddFixedJoint(
      MakeElementRange(finger.block_instances.front()),
      tendon_range,
      to_world({kFingerBaseX + kDistalAttachX, 0, kGuideZ}),
      kJointOffset, 1.0);
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
  std::cout << "FEAT10 Tendon-Driven Hand Curl Demo\n";
  std::cout << "========================================\n";
  std::cout << "fingers=" << kNumFingers << " blocks_per_finger="
            << kNumFingerBlocks << " + shared palm\n";
  std::cout << "steps=" << max_steps << " export_interval=" << export_interval
            << "\n";

  std::filesystem::create_directories("output/engineering_joint");

  const std::string mesh_dir = "data/meshes/T10/hand_simple";
  const std::string block_dir = "data/meshes/T10/finger";
  const std::string block_prefix = block_dir + "/finger_block.1";
  const std::string tendon_prefix = mesh_dir + "/tendon.1";
  const std::string palm_prefix = mesh_dir + "/palm.1";

  // --- Load shared palm mesh ---
  ANCFCPUUtils::MeshManager mesh_manager;

  const int palm_mesh_id = mesh_manager.LoadMesh(
      palm_prefix + ".node", palm_prefix + ".ele", "palm", kFingerMaterial);
  if (palm_mesh_id < 0) {
    std::cerr << "Failed to load palm mesh\n";
    return 1;
  }

  // --- Load meshes for all fingers ---
  std::vector<FingerData> fingers(kNumFingers);

  for (int fi = 0; fi < kNumFingers; ++fi) {
    const auto& spec = kFingerSpecs[fi];
    const Eigen::Matrix3d R = RotZ(spec.angle);
    FingerData& finger = fingers[fi];
    finger.block_mesh_ids.reserve(kNumFingerBlocks);

    for (int bi = 0; bi < kNumFingerBlocks; ++bi) {
      const std::string name =
          "f" + std::to_string(fi) + "_block_" + std::to_string(bi);
      const int mesh_id = mesh_manager.LoadMesh(
          block_prefix + ".node", block_prefix + ".ele", name, kFingerMaterial);
      if (mesh_id < 0) {
        std::cerr << "Failed to load finger block mesh\n";
        return 1;
      }
      const Eigen::Vector3d local_offset(kFingerBaseX + bi * kBlockLength,
                                         -kGuideY, 0.0);
      mesh_manager.TransformMesh(mesh_id, MakeTransform(R, spec.attach,
                                                         local_offset));
      finger.block_mesh_ids.push_back(mesh_id);
    }

    const std::string tendon_name = "f" + std::to_string(fi) + "_tendon";
    finger.tendon_mesh_id = mesh_manager.LoadMesh(
        tendon_prefix + ".node", tendon_prefix + ".ele", tendon_name,
        kTendonMaterial);
    if (finger.tendon_mesh_id < 0) {
      std::cerr << "Failed to load tendon mesh\n";
      return 1;
    }
    const Eigen::Vector3d tendon_local(kFingerBaseX, -0.5 * kTendonWidth,
                                       kGuideZ - 0.5 * kTendonThickness);
    mesh_manager.TransformMesh(finger.tendon_mesh_id,
                               MakeTransform(R, spec.attach, tendon_local));
  }

  // --- Cache mesh instances and select boundary nodes ---
  const Eigen::MatrixXd& all_nodes = mesh_manager.GetAllNodes();

  const auto palm_instance = mesh_manager.GetMeshInstance(palm_mesh_id);
  const auto palm_range = MakeElementRange(palm_instance);
  const auto palm_fixed_nodes = SelectNodesOnPlane(
      all_nodes, palm_instance, 0, kPalmBackX, kBasePatchTolerance);
  if (palm_fixed_nodes.empty()) {
    throw std::runtime_error("Failed to identify palm base-face nodes");
  }

  for (int fi = 0; fi < kNumFingers; ++fi) {
    const auto& spec = kFingerSpecs[fi];
    const Eigen::Matrix3d R = RotZ(spec.angle);
    FingerData& finger = fingers[fi];

    finger.block_instances.reserve(kNumFingerBlocks);
    for (int mesh_id : finger.block_mesh_ids) {
      finger.block_instances.push_back(mesh_manager.GetMeshInstance(mesh_id));
    }
    finger.tendon_instance = mesh_manager.GetMeshInstance(finger.tendon_mesh_id);

    // Tendon pull face: the far end in world space
    const Eigen::Vector3d pull_face_point =
        spec.attach + R * Eigen::Vector3d(kFingerBaseX + kTendonLength, 0, 0);
    const Eigen::Vector3d pull_face_normal = R * Eigen::Vector3d::UnitX();
    finger.tendon_pull_nodes = SelectNodesOnPlane(
        all_nodes, finger.tendon_instance, pull_face_point, pull_face_normal,
        kFaceTolerance);

    if (finger.tendon_pull_nodes.empty()) {
      throw std::runtime_error(
          "Failed to identify tendon pull-face nodes for finger " +
          std::to_string(fi));
    }
  }

  const Eigen::MatrixXi& all_elems = mesh_manager.GetAllElements();
  const int n_nodes                = mesh_manager.GetTotalNodes();
  const int n_elems                = mesh_manager.GetTotalElements();

  // --- Print geometry summary ---
  std::cout << "palm: " << palm_instance.num_nodes << " nodes, "
            << palm_instance.num_elements << " elems, fixed nodes: "
            << palm_fixed_nodes.size() << "\n";
  for (int fi = 0; fi < kNumFingers; ++fi) {
    const FingerData& finger = fingers[fi];
    const auto& spec = kFingerSpecs[fi];
    std::cout << "finger[" << fi << "] attach=["
              << spec.attach.transpose() << "] angle="
              << spec.angle * 180.0 / M_PI << "deg\n";
    for (int bi = 0; bi < kNumFingerBlocks; ++bi) {
      const auto& inst = finger.block_instances[bi];
      std::cout << "  block[" << bi << "]: " << inst.num_nodes << " nodes, "
                << inst.num_elements << " elems\n";
    }
    std::cout << "  tendon: " << finger.tendon_instance.num_nodes << " nodes, "
              << finger.tendon_instance.num_elements << " elems\n";
    std::cout << "  tendon pull nodes: " << finger.tendon_pull_nodes.size()
              << "\n";
  }
  std::cout << "total: " << n_nodes << " nodes, " << n_elems << " elements\n";

  // --- GPU setup ---
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

  // --- Constraints ---
  FEAT10ConstraintManager constraint_manager(&gpu_t10_data);

  // Ground palm back face
  for (int node : palm_fixed_nodes) {
    constraint_manager.AddNodeToWorldCD(node);
  }

  for (int fi = 0; fi < kNumFingers; ++fi) {
    const auto& spec = kFingerSpecs[fi];
    const Eigen::Matrix3d R = RotZ(spec.angle);
    FingerData& finger = fingers[fi];

    AddFinger(constraint_manager, finger, palm_range, R, spec.attach);

    finger.tip_reference = constraint_manager.LocateReferencePoint(
        spec.attach + R * Eigen::Vector3d(kFingerBaseX + 0.001, 0,
                                          kBlockHeight - 0.001),
        MakeElementRange(finger.block_instances.front()));
  }

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

  // --- Record initial positions ---
  Eigen::VectorXd x_curr, y_curr, z_curr;
  gpu_t10_data.RetrievePositionToCPU(x_curr, y_curr, z_curr);
  for (int fi = 0; fi < kNumFingers; ++fi) {
    fingers[fi].tip_initial = EvaluateCurrentPointPosition(
        fingers[fi].tip_reference, all_elems, x_curr, y_curr, z_curr);
    std::cout << "finger[" << fi << "] initial tip: ["
              << fingers[fi].tip_initial.transpose() << "]\n";
  }

  // --- Precompute per-finger pull directions ---
  std::vector<Eigen::Vector3d> pull_dirs(kNumFingers);
  for (int fi = 0; fi < kNumFingers; ++fi) {
    pull_dirs[fi] = RotZ(kFingerSpecs[fi].angle) * Eigen::Vector3d::UnitX();
  }

  // --- Time stepping ---
  int output_frame = 1;
  Eigen::VectorXd step_f_ext(n_nodes * 3);
  for (int step = 1; step <= max_steps; ++step) {
    const double ramp =
        std::min(1.0, static_cast<double>(step) / kPullRampSteps);
    step_f_ext.setZero();
    for (int fi = 0; fi < kNumFingers; ++fi) {
      AddDistributedForce(&step_f_ext, fingers[fi].tendon_pull_nodes,
                          ramp * kTendonPullForce * pull_dirs[fi]);
    }
    gpu_t10_data.SetExternalForce(step_f_ext);

    solver.Solve();

    if (step % export_interval == 0 || step == max_steps) {
      gpu_t10_data.WriteOutputVTU(MakeOutputPath(output_frame));
      ++output_frame;

      gpu_t10_data.RetrievePositionToCPU(x_curr, y_curr, z_curr);
      std::cout << "step " << step << " ramp=" << ramp;
      for (int fi = 0; fi < kNumFingers; ++fi) {
        const Eigen::Vector3d tip_current = EvaluateCurrentPointPosition(
            fingers[fi].tip_reference, all_elems, x_curr, y_curr, z_curr);
        const Eigen::Vector3d tip_disp = tip_current - fingers[fi].tip_initial;
        std::cout << " f" << fi << "_tip_disp=[" << tip_disp.transpose() << "]";
      }
      std::cout << "\n";
    }
  }

  gpu_t10_data.Destroy();
  std::cout << "Done. Output written to output/engineering_joint/\n";
  return 0;
}
