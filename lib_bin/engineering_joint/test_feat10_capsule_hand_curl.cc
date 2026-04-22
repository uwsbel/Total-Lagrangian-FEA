/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Ganesh Arivoli, Json Zhou
 * Email:   ganesh.arivoli@gmail.com, zzhou292@wisc.edu
 * File:    test_feat10_capsule_hand_curl.cc
 * Brief:   FEAT10 Tendon-Driven Hand Curl Demo (URDF Capsule Geometry).
 *          Palm + 5 fingers (index, middle, ring, pinky, thumb),
 *          each with 3 capsule phalanges and a tendon.
 *          Revolute joints, cylindrical guides, fixed welds.
 *          All tendons pulled to curl the hand.
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

constexpr double kJointOffset   = 0.001;
constexpr double kFaceTolerance = 1e-10;
constexpr double kPalmWristTol  = 1e-10;
constexpr double kGuideX        = -0.004;  // tendon X offset for moment arm
constexpr double kTendonLength  = 0.150;
constexpr double kTendonPullForce = 0.8;
constexpr double kPalmGuideLocalY = -0.030;  // 30mm back from attach into palm

const SolidMaterialProperties kFingerMaterial =
    SolidMaterialProperties::SVK(5.0e6, 0.32, 1200.0, 1.0e3, 1.0e3);

const SolidMaterialProperties kTendonMaterial =
    SolidMaterialProperties::SVK(5.0e7, 0.30, 1100.0, 5.0e2, 5.0e2);

// --- Finger specification from URDF ---

struct FingerSpec {
  const char* name;
  const char* proximal_mesh;
  const char* intermediate_mesh;
  const char* distal_mesh;
  Eigen::Vector3d attach;
  double rx_angle;
  double inter_y;
  double distal_y;
  double tip_y;
};

const FingerSpec kFingerSpecs[] = {
    {"index", "capsule_34.5x14", "capsule_23.5x14", "capsule_19.5x14",
     {0, -0.080, -0.029}, -M_PI, 0.0345, 0.0235, 0.0195},
    {"middle", "capsule_35.0x14", "capsule_25.0x14", "capsule_20.5x14",
     {0, -0.080, -0.010}, -M_PI, 0.035, 0.025, 0.0205},
    {"ring", "capsule_34.5x14", "capsule_23.5x14", "capsule_19.5x14",
     {0, -0.080, 0.010}, -M_PI, 0.0345, 0.0235, 0.0195},
    {"pinky", "capsule_30.5x14", "capsule_21.0x14", "capsule_18.0x14",
     {0, -0.080, 0.029}, -M_PI, 0.0305, 0.021, 0.018},
    {"thumb", "capsule_28.0x14", "capsule_21.0x14", "capsule_17.0x14",
     {0, -0.035, -0.036}, -2.3562, 0.028, 0.021, 0.017},
};
constexpr int kNumFingers = sizeof(kFingerSpecs) / sizeof(kFingerSpecs[0]);

struct FingerData {
  ANCFCPUUtils::MeshInstance prox_inst, inter_inst, dist_inst, tendon_inst;
  std::vector<int> tendon_pull_nodes;
  FEAT10ConstraintManager::ReferencePoint tip_reference;
  Eigen::Vector3d tip_initial;
};

// --- Helper functions ---

Eigen::Matrix3d Rx(double angle) {
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  Eigen::Matrix3d R;
  R << 1, 0, 0,
       0, c, -s,
       0, s, c;
  return R;
}

Eigen::Matrix4d MakeTransform(const Eigen::Matrix3d& R,
                               const Eigen::Vector3d& t) {
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = R;
  T.block<3, 1>(0, 3) = t;
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
  if (nodes.empty()) return;
  const Eigen::Vector3d fpn = total_force / static_cast<double>(nodes.size());
  for (int node : nodes) {
    (*h_f_ext)(3 * node + 0) += fpn.x();
    (*h_f_ext)(3 * node + 1) += fpn.y();
    (*h_f_ext)(3 * node + 2) += fpn.z();
  }
}

Eigen::Vector3d EvaluateCurrentPointPosition(
    const FEAT10ConstraintManager::ReferencePoint& point,
    const Eigen::MatrixXi& connectivity, const Eigen::VectorXd& x,
    const Eigen::VectorXd& y, const Eigen::VectorXd& z) {
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  for (int ln = 0; ln < Quadrature::N_NODE_T10_10; ++ln) {
    const double w = point.shape(ln);
    if (w == 0.0) continue;
    const int node = connectivity(point.element_idx, ln);
    position += w * Eigen::Vector3d(x(node), y(node), z(node));
  }
  return position;
}

std::string MakeOutputPath(int frame) {
  std::ostringstream oss;
  oss << "output/engineering_joint/capsule_hand_curl_" << std::setw(6)
      << std::setfill('0') << frame << ".vtu";
  return oss.str();
}

// --- Add all constraints for one finger ---

void AddFinger(FEAT10ConstraintManager& cm,
               FingerData& finger,
               const FEAT10ConstraintManager::ElementRange& palm_range,
               const FingerSpec& spec) {
  const Eigen::Matrix3d R = Rx(spec.rx_angle);
  const Eigen::Matrix3d R_tendon = Rx(spec.rx_angle + M_PI);

  const Eigen::Vector3d hinge_axis = R * Eigen::Vector3d::UnitZ();
  const Eigen::Vector3d guide_axis = R_tendon * Eigen::Vector3d::UnitY();

  // World-frame joint positions
  const Eigen::Vector3d prox_joint = spec.attach;
  const Eigen::Vector3d inter_joint =
      spec.attach + R * Eigen::Vector3d(0, spec.inter_y, 0);
  const Eigen::Vector3d dist_joint =
      spec.attach + R * Eigen::Vector3d(0, spec.inter_y + spec.distal_y, 0);
  const Eigen::Vector3d tip_pos =
      spec.attach +
      R * Eigen::Vector3d(0, spec.inter_y + spec.distal_y + spec.tip_y, 0);

  // Capsule midpoints (for cylindrical guides)
  const Eigen::Vector3d prox_mid = 0.5 * (prox_joint + inter_joint);
  const Eigen::Vector3d inter_mid = 0.5 * (inter_joint + dist_joint);
  const Eigen::Vector3d dist_mid = 0.5 * (dist_joint + tip_pos);

  // Palm guide: 30mm back from attachment into palm
  const Eigen::Vector3d palm_guide_pos =
      spec.attach + R * Eigen::Vector3d(0, kPalmGuideLocalY, 0) +
      Eigen::Vector3d(kGuideX, 0, 0);

  // Tendon weld point: near the distal midpoint
  const double weld_local_y =
      spec.inter_y + spec.distal_y + 0.4 * spec.tip_y;
  const Eigen::Vector3d weld_pos =
      spec.attach + R * Eigen::Vector3d(0, weld_local_y, 0) +
      Eigen::Vector3d(kGuideX, 0, 0);

  // Element ranges
  const auto prox_range = MakeElementRange(finger.prox_inst);
  const auto inter_range = MakeElementRange(finger.inter_inst);
  const auto dist_range = MakeElementRange(finger.dist_inst);
  const auto tendon_range = MakeElementRange(finger.tendon_inst);

  // Guide/weld positions with X offset for moment arm
  auto with_guide_x = [](const Eigen::Vector3d& p) {
    return Eigen::Vector3d(kGuideX, p.y(), p.z());
  };

  // Revolute joints
  cm.AddRevoluteJoint(palm_range, prox_range, prox_joint,
                      hinge_axis, kJointOffset, 1.0);
  cm.AddRevoluteJoint(prox_range, inter_range, inter_joint,
                      hinge_axis, kJointOffset, 1.0);
  cm.AddRevoluteJoint(inter_range, dist_range, dist_joint,
                      hinge_axis, kJointOffset, 1.0);

  // Cylindrical tendon guides
  cm.AddCylindricalJoint(palm_range, tendon_range,
                         palm_guide_pos, palm_guide_pos,
                         guide_axis, kJointOffset, 1.0, 1.0);
  cm.AddCylindricalJoint(prox_range, tendon_range,
                         with_guide_x(prox_mid), with_guide_x(prox_mid),
                         guide_axis, kJointOffset, 1.0, 1.0);
  cm.AddCylindricalJoint(inter_range, tendon_range,
                         with_guide_x(inter_mid), with_guide_x(inter_mid),
                         guide_axis, kJointOffset, 1.0, 1.0);
  cm.AddCylindricalJoint(dist_range, tendon_range,
                         with_guide_x(dist_mid), with_guide_x(dist_mid),
                         guide_axis, kJointOffset, 1.0, 1.0);

  // Fixed weld: tendon to distal capsule
  cm.AddFixedJoint(dist_range, tendon_range, weld_pos, kJointOffset, 1.0);

  // Tip reference point for tracking curl
  finger.tip_reference = cm.LocateReferencePoint(dist_mid, dist_range);
}

}  // namespace

int main(int argc, char** argv) {
  int max_steps       = kNumStepsDefault;
  int export_interval = kExportIntervalDef;
  if (argc > 1) {
    const int v = std::atoi(argv[1]);
    if (v > 0) max_steps = v;
  }
  if (argc > 2) {
    const int v = std::atoi(argv[2]);
    if (v > 0) export_interval = v;
  }

  std::cout << "========================================\n";
  std::cout << "FEAT10 Hand Curl Demo (URDF Capsules)\n";
  std::cout << "========================================\n";
  std::cout << "fingers=" << kNumFingers << " steps=" << max_steps
            << " export_interval=" << export_interval << "\n";

  std::filesystem::create_directories("output/engineering_joint");
  const std::string mesh_dir = "data/meshes/T10/hand_capsule";

  // --- Load palm ---
  ANCFCPUUtils::MeshManager mesh_manager;
  const int palm_id = mesh_manager.LoadMesh(
      mesh_dir + "/hand_base_link.1.node",
      mesh_dir + "/hand_base_link.1.ele", "palm", kFingerMaterial);
  if (palm_id < 0) {
    std::cerr << "Failed to load palm mesh\n";
    return 1;
  }

  // --- Load and transform finger meshes ---
  std::vector<FingerData> fingers(kNumFingers);

  for (int fi = 0; fi < kNumFingers; ++fi) {
    const auto& spec = kFingerSpecs[fi];
    const Eigen::Matrix3d R = Rx(spec.rx_angle);
    const Eigen::Matrix3d R_tendon = Rx(spec.rx_angle + M_PI);

    auto load_capsule = [&](const char* mesh_name, const std::string& label,
                            const Eigen::Vector3d& joint_pos) -> int {
      const std::string prefix =
          mesh_dir + "/" + std::string(mesh_name) + ".1";
      const int id = mesh_manager.LoadMesh(
          prefix + ".node", prefix + ".ele", label, kFingerMaterial);
      if (id < 0) {
        std::cerr << "Failed to load " << label << "\n";
        return -1;
      }
      mesh_manager.TransformMesh(id, MakeTransform(R, joint_pos));
      return id;
    };

    const std::string prefix = spec.name;

    // Capsule world positions
    const Eigen::Vector3d prox_pos = spec.attach;
    const Eigen::Vector3d inter_pos =
        spec.attach + R * Eigen::Vector3d(0, spec.inter_y, 0);
    const Eigen::Vector3d dist_pos =
        spec.attach +
        R * Eigen::Vector3d(0, spec.inter_y + spec.distal_y, 0);

    const int prox_id =
        load_capsule(spec.proximal_mesh, prefix + "_prox", prox_pos);
    const int inter_id =
        load_capsule(spec.intermediate_mesh, prefix + "_inter", inter_pos);
    const int dist_id =
        load_capsule(spec.distal_mesh, prefix + "_dist", dist_pos);
    if (prox_id < 0 || inter_id < 0 || dist_id < 0) return 1;

    // Tendon mesh is Y-oriented and centered: Y[0,0.150], X[-W/2,W/2], Z[-T/2,T/2].
    // R_tendon orients +Y toward the wrist. Translation places Y=0 at the distal tip.
    const double tendon_tip_local_y =
        spec.inter_y + spec.distal_y + 0.5 * spec.tip_y + 0.005;
    const Eigen::Vector3d tendon_translation =
        spec.attach + R * Eigen::Vector3d(0, tendon_tip_local_y, 0) +
        Eigen::Vector3d(kGuideX, 0, 0);

    const std::string tendon_prefix = mesh_dir + "/tendon.1";
    const int tendon_id = mesh_manager.LoadMesh(
        tendon_prefix + ".node", tendon_prefix + ".ele",
        prefix + "_tendon", kTendonMaterial);
    if (tendon_id < 0) {
      std::cerr << "Failed to load tendon for " << spec.name << "\n";
      return 1;
    }
    mesh_manager.TransformMesh(tendon_id,
                               MakeTransform(R_tendon, tendon_translation));

    fingers[fi].prox_inst = mesh_manager.GetMeshInstance(prox_id);
    fingers[fi].inter_inst = mesh_manager.GetMeshInstance(inter_id);
    fingers[fi].dist_inst = mesh_manager.GetMeshInstance(dist_id);
    fingers[fi].tendon_inst = mesh_manager.GetMeshInstance(tendon_id);
  }

  const auto palm_inst = mesh_manager.GetMeshInstance(palm_id);
  const auto palm_range = MakeElementRange(palm_inst);

  const Eigen::MatrixXd& all_nodes = mesh_manager.GetAllNodes();
  const Eigen::MatrixXi& all_elems = mesh_manager.GetAllElements();
  const int n_nodes = mesh_manager.GetTotalNodes();
  const int n_elems = mesh_manager.GetTotalElements();

  // --- Select boundary nodes and precompute pull directions ---
  const auto palm_fixed_nodes =
      SelectNodesOnPlane(all_nodes, palm_inst, 1, 0.0, kPalmWristTol);
  if (palm_fixed_nodes.empty()) {
    throw std::runtime_error("Failed to identify palm wrist-face nodes");
  }

  std::vector<Eigen::Vector3d> pull_dirs(kNumFingers);
  for (int fi = 0; fi < kNumFingers; ++fi) {
    const auto& spec = kFingerSpecs[fi];
    const Eigen::Matrix3d R_tendon = Rx(spec.rx_angle + M_PI);

    const double tendon_tip_local_y =
        spec.inter_y + spec.distal_y + 0.5 * spec.tip_y + 0.005;
    const Eigen::Vector3d tendon_translation =
        spec.attach + Rx(spec.rx_angle) * Eigen::Vector3d(0, tendon_tip_local_y, 0) +
        Eigen::Vector3d(kGuideX, 0, 0);

    const Eigen::Vector3d pull_pos =
        R_tendon * Eigen::Vector3d(0, kTendonLength, 0) + tendon_translation;
    pull_dirs[fi] = (R_tendon * Eigen::Vector3d::UnitY()).normalized();

    fingers[fi].tendon_pull_nodes = SelectNodesOnPlane(
        all_nodes, fingers[fi].tendon_inst, pull_pos, pull_dirs[fi],
        kFaceTolerance);

    if (fingers[fi].tendon_pull_nodes.empty()) {
      throw std::runtime_error(
          std::string("Failed to identify tendon pull-face nodes for ") +
          spec.name);
    }
  }

  // --- Print geometry summary ---
  std::cout << "palm: " << palm_inst.num_nodes << " nodes, "
            << palm_inst.num_elements << " elems, fixed: "
            << palm_fixed_nodes.size() << "\n";
  for (int fi = 0; fi < kNumFingers; ++fi) {
    const auto& f = fingers[fi];
    std::cout << kFingerSpecs[fi].name << ": prox=" << f.prox_inst.num_nodes
              << "n/" << f.prox_inst.num_elements << "e"
              << " inter=" << f.inter_inst.num_nodes << "n/"
              << f.inter_inst.num_elements << "e"
              << " dist=" << f.dist_inst.num_nodes << "n/"
              << f.dist_inst.num_elements << "e"
              << " tendon=" << f.tendon_inst.num_nodes << "n/"
              << f.tendon_inst.num_elements << "e"
              << " pull_nodes=" << f.tendon_pull_nodes.size() << "\n";
  }
  std::cout << "total: " << n_nodes << " nodes, " << n_elems << " elements\n";

  // --- GPU setup ---
  GPU_FEAT10_Data gpu_t10_data(n_elems, n_nodes);
  gpu_t10_data.Initialize();

  Eigen::VectorXd h_x(n_nodes), h_y(n_nodes), h_z(n_nodes);
  for (int i = 0; i < n_nodes; ++i) {
    h_x(i) = all_nodes(i, 0);
    h_y(i) = all_nodes(i, 1);
    h_z(i) = all_nodes(i, 2);
  }
  gpu_t10_data.Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y,
                     Quadrature::tet5pt_z, Quadrature::tet5pt_weights,
                     h_x, h_y, h_z, all_elems);
  gpu_t10_data.ApplyMaterialsFromMeshManager(mesh_manager);
  gpu_t10_data.CalcDnDuPre();
  gpu_t10_data.CalcMassMatrix();
  gpu_t10_data.SetExternalForce(Eigen::VectorXd::Zero(n_nodes * 3));

  // --- Constraints ---
  FEAT10ConstraintManager constraint_manager(&gpu_t10_data);

  for (int node : palm_fixed_nodes) {
    constraint_manager.AddNodeToWorldCD(node);
  }

  for (int fi = 0; fi < kNumFingers; ++fi) {
    AddFinger(constraint_manager, fingers[fi], palm_range, kFingerSpecs[fi]);
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

  // --- Record initial tip positions ---
  Eigen::VectorXd x_curr, y_curr, z_curr;
  gpu_t10_data.RetrievePositionToCPU(x_curr, y_curr, z_curr);
  for (int fi = 0; fi < kNumFingers; ++fi) {
    fingers[fi].tip_initial = EvaluateCurrentPointPosition(
        fingers[fi].tip_reference, all_elems, x_curr, y_curr, z_curr);
    std::cout << kFingerSpecs[fi].name << " initial tip: ["
              << fingers[fi].tip_initial.transpose() << "]\n";
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
        const Eigen::Vector3d tip = EvaluateCurrentPointPosition(
            fingers[fi].tip_reference, all_elems, x_curr, y_curr, z_curr);
        const Eigen::Vector3d disp = tip - fingers[fi].tip_initial;
        std::cout << " " << kFingerSpecs[fi].name << "_disp=["
                  << disp.transpose() << "]";
      }
      std::cout << "\n";
    }
  }

  gpu_t10_data.Destroy();
  std::cout << "Done. Output written to output/engineering_joint/\n";
  return 0;
}
