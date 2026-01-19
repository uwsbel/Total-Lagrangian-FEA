#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string_view>

#include "../../lib_src/collision/DemeMeshCollisionSystem.h"
#include "../../lib_src/elements/FEAT10Data.cuh"
#include "../../lib_src/solvers/SyncedNewton.cuh"
#include "../../lib_utils/cpu_utils.h"
#include "../../lib_utils/cli_utils.h"
#include "../../lib_utils/mesh_manager.h"
#include "../../lib_utils/quadrature_utils.h"
#include "../../lib_utils/surface_trimesh_extract.h"
#include "../../lib_utils/visualization_utils.h"

using ANCFCPUUtils::VisualizationUtils;

static void CheckCublas(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS error (" << what << "): status=" << int(status)
              << "\n";
    std::exit(1);
  }
}

static double TetVolume(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
                        const Eigen::Vector3d& p2, const Eigen::Vector3d& p3) {
  const Eigen::Vector3d a = p1 - p0;
  const Eigen::Vector3d b = p2 - p0;
  const Eigen::Vector3d c = p3 - p0;
  return std::abs(a.dot(b.cross(c))) / 6.0;
}

static double ComputeTetMeshVolume_T10_UsingCorners(
    const Eigen::MatrixXd& nodes, const Eigen::MatrixXi& elements,
    int elem_offset, int num_elems) {
  double V = 0.0;
  for (int e = 0; e < num_elems; ++e) {
    const int idx = elem_offset + e;
    const int n0  = elements(idx, 0);
    const int n1  = elements(idx, 1);
    const int n2  = elements(idx, 2);
    const int n3  = elements(idx, 3);
    const Eigen::Vector3d p0 = nodes.row(n0);
    const Eigen::Vector3d p1 = nodes.row(n1);
    const Eigen::Vector3d p2 = nodes.row(n2);
    const Eigen::Vector3d p3 = nodes.row(n3);
    V += TetVolume(p0, p1, p2, p3);
  }
  return V;
}

static double InferSlopeX0_FromZMin(const Eigen::MatrixXd& nodes,
                                    const ANCFCPUUtils::MeshInstance& inst,
                                    double z_eps = 1e-9) {
  double z_min = 1e100;
  for (int i = 0; i < inst.num_nodes; ++i) {
    const int idx = inst.node_offset + i;
    z_min         = std::min(z_min, nodes(idx, 2));
  }

  double sum_x = 0.0;
  int count    = 0;
  for (int i = 0; i < inst.num_nodes; ++i) {
    const int idx = inst.node_offset + i;
    if (std::abs(nodes(idx, 2) - z_min) <= z_eps) {
      sum_x += nodes(idx, 0);
      count++;
    }
  }
  if (count == 0) {
    return 0.0;
  }
  return sum_x / static_cast<double>(count);
}

int main(int argc, char** argv) {
  std::cout << "========================================" << std::endl;
  std::cout << "Brick Sliding Simulation" << std::endl;
  std::cout << "========================================" << std::endl;

  const double gravity = -9.81;
  const double dt      = 5e-4;

  const double contact_cor_default      = 0.0;
  const double contact_mu_s_default     = 0.25;
  const double contact_mu_k_default     = 0.2;
  const double contact_stiffness        = 1e8;
  const bool enable_self_collision      = false;
  const double init_clearance           = 1e-5;
  const double brick_mass_kg            = 1.0;

  std::string slope_key        = "02";
  double contact_cor          = contact_cor_default;
  double contact_mu_s         = contact_mu_s_default;
  double contact_mu_k         = contact_mu_k_default;
  int max_steps               = 2000;
  int export_interval         = 10;
  std::string out_suffix;

  const bool has_flag_args =
      (argc > 1 && argv[1] &&
       (std::string_view(argv[1]) == "--help" ||
        std::string_view(argv[1]) == "-h" ||
        std::string_view(argv[1]).rfind("--", 0) == 0));
  if (has_flag_args) {
    ANCFCPUUtils::Cli cli(argv[0] ? argv[0] : "brick_sliding");
    cli.SetDescription("Brick sliding on a fixed slope (DEME mesh-mesh contact coupled to FE)."
                       );
    cli.AddString("slope", "02",
                  "slope selection: 1|2|3|4|01|02|03|04|03p1|03p2|03p3|03p4");
    cli.AddDouble("cor", contact_cor_default,
                  "contact restitution (CoR), range [0, 1]");
    cli.AddDouble("mu_s", contact_mu_s_default, "contact static friction mu_s");
    cli.AddDouble("mu_k", contact_mu_k_default,
                  "contact kinetic friction mu_k");
    cli.AddInt("steps", max_steps, "max simulation steps");
    cli.AddInt("export_interval", export_interval,
               "VTK export interval (0 disables)");
    cli.AddString("out_suffix", "", "suffix appended to output folder name");

    std::string err;
    if (!cli.Parse(argc, argv, &err) || cli.HelpRequested()) {
      if (!err.empty()) std::cerr << "Error: " << err << "\n\n";
      cli.PrintUsage(std::cerr);
      return cli.HelpRequested() ? 0 : 1;
    }

    slope_key       = cli.GetString("slope");
    contact_cor     = cli.GetDouble("cor");
    contact_mu_s    = cli.GetDouble("mu_s");
    contact_mu_k    = cli.GetDouble("mu_k");
    max_steps       = cli.GetInt("steps");
    export_interval = cli.GetInt("export_interval");
    out_suffix      = cli.GetString("out_suffix");
  }

  if (contact_cor < 0.0 || contact_cor > 1.0) {
    std::cerr << "Invalid --cor: " << contact_cor << " (expected in [0,1])"
              << std::endl;
    return 1;
  }
  if (contact_mu_s < 0.0 || contact_mu_k < 0.0) {
    std::cerr << "Invalid friction: mu_s=" << contact_mu_s
              << ", mu_k=" << contact_mu_k << " (expected >= 0)"
              << std::endl;
    return 1;
  }
  if (out_suffix.find('/') != std::string::npos ||
      out_suffix.find('\\') != std::string::npos) {
    std::cerr << "Invalid --out_suffix: " << out_suffix
              << " (must not contain path separators)" << std::endl;
    return 1;
  }

  if (slope_key == "1") {
    slope_key = "01";
  } else if (slope_key == "2") {
    slope_key = "02";
  } else if (slope_key == "3") {
    slope_key = "03";
  } else if (slope_key == "4") {
    slope_key = "04";
  }

  double alpha = 0.0;
  if (slope_key == "01") {
    alpha = 0.18;
  } else if (slope_key == "02") {
    alpha = 0.197395560;
  } else if (slope_key == "03") {
    alpha = 0.244978663;
  } else if (slope_key == "03p1") {
    alpha = 0.2440372654;
  } else if (slope_key == "03p2") {
    alpha = 0.2459196179;
  } else if (slope_key == "03p3") {
    alpha = std::atan(0.247);
  } else if (slope_key == "03p4") {
    alpha = std::atan(0.248);
  } else if (slope_key == "04") {
    alpha = 0.25;
  } else {
    std::cerr << "Invalid --slope: " << slope_key
              << " (expected 1|2|3|4|01|02|03|04|03p1|03p2|03p3|03p4)" << std::endl;
    return 1;
  }

  std::string output_dir = "output/brick_sliding";
  if (!out_suffix.empty()) {
    output_dir += "_" + out_suffix;
  }
  output_dir += "_s" + slope_key;
  std::filesystem::create_directories(output_dir);

  ANCFCPUUtils::MeshManager mesh_manager;

  const std::string brick_node =
      "data/meshes/ptest/brick/brick_medium.1.node";
  const std::string brick_ele = "data/meshes/ptest/brick/brick_medium.1.ele";

  const std::string slope_prefix =
      std::string("data/meshes/ptest/slope/slope_") + slope_key + ".1";
  const std::string slope_node = slope_prefix + ".node";
  const std::string slope_ele  = slope_prefix + ".ele";

  const int mesh_slope =
      mesh_manager.LoadMesh(slope_node, slope_ele, "slope");
  const int mesh_brick =
      mesh_manager.LoadMesh(brick_node, brick_ele, "brick");

  if (mesh_slope < 0 || mesh_brick < 0) {
    std::cerr << "Failed to load meshes" << std::endl;
    return 1;
  }

  const auto& inst_slope = mesh_manager.GetMeshInstance(mesh_slope);
  const auto& inst_brick = mesh_manager.GetMeshInstance(mesh_brick);

  std::cout << "Loaded meshes:" << std::endl;
  std::cout << "  Slope: " << inst_slope.num_nodes << " nodes, "
            << inst_slope.num_elements << " elements" << std::endl;
  std::cout << "  Brick: " << inst_brick.num_nodes << " nodes, "
            << inst_brick.num_elements << " elements" << std::endl;

  std::cout << "Slope: " << slope_key << std::endl;
  std::cout << "Slope alpha (rad): " << alpha << std::endl;
  std::cout << "Contact CoR: " << contact_cor << std::endl;
  std::cout << "Contact mu_s: " << contact_mu_s << std::endl;
  std::cout << "Contact mu_k: " << contact_mu_k << std::endl;
  std::cout << "Max steps: " << max_steps << std::endl;
  std::cout << "Export interval: " << export_interval << std::endl;

  {
    mesh_manager.TransformMesh(mesh_brick, ANCFCPUUtils::rotationY(-alpha));

    const double x_target = 0.8;
    const double y_target = 0.0;

    mesh_manager.TranslateMesh(mesh_brick, x_target, y_target, 0.0);

    const Eigen::MatrixXd& nodes1 = mesh_manager.GetAllNodes();

    const Eigen::Vector3d n_raw(-std::tan(alpha), 0.0, 1.0);
    const Eigen::Vector3d n = n_raw.normalized();

    double brick_x_min = 1e100;
    double brick_x_max = -1e100;
    double brick_y_min = 1e100;
    double brick_y_max = -1e100;
    for (int i = 0; i < inst_brick.num_nodes; ++i) {
      const int idx = inst_brick.node_offset + i;
      brick_x_min   = std::min(brick_x_min, nodes1(idx, 0));
      brick_x_max   = std::max(brick_x_max, nodes1(idx, 0));
      brick_y_min   = std::min(brick_y_min, nodes1(idx, 1));
      brick_y_max   = std::max(brick_y_max, nodes1(idx, 1));
    }

    const double margin_xy = 0.02;
    const double x0        = brick_x_min - margin_xy;
    const double x1        = brick_x_max + margin_xy;
    const double y0        = brick_y_min - margin_xy;
    const double y1        = brick_y_max + margin_xy;

    double slope_b = -1e100;
    int slope_count = 0;
    for (int i = 0; i < inst_slope.num_nodes; ++i) {
      const int idx = inst_slope.node_offset + i;
      const double x = nodes1(idx, 0);
      const double y = nodes1(idx, 1);
      if (x < x0 || x > x1 || y < y0 || y > y1) {
        continue;
      }
      const Eigen::Vector3d p = nodes1.row(idx);
      slope_b                 = std::max(slope_b, n.dot(p));
      slope_count++;
    }
    if (slope_count == 0) {
      slope_b = -1e100;
      for (int i = 0; i < inst_slope.num_nodes; ++i) {
        const int idx           = inst_slope.node_offset + i;
        const Eigen::Vector3d p = nodes1.row(idx);
        slope_b                 = std::max(slope_b, n.dot(p));
      }
    }

    double brick_min = 1e100;
    for (int i = 0; i < inst_brick.num_nodes; ++i) {
      const int idx           = inst_brick.node_offset + i;
      const Eigen::Vector3d p = nodes1.row(idx);
      brick_min               = std::min(brick_min, n.dot(p));
    }

    const double shift = (slope_b + init_clearance) - brick_min;
    if (shift > 0.0) {
      const Eigen::Vector3d t = shift * n;
      mesh_manager.TranslateMesh(mesh_brick, t.x(), t.y(), t.z());
    }

    const double neighbor_r2 = 0.03 * 0.03;
    double lift_z_needed     = 0.0;
    {
      const Eigen::MatrixXd& nodes2 = mesh_manager.GetAllNodes();

      const auto PlaneZ = [&](double x, double y) {
        return (slope_b - n.x() * x - n.y() * y) / n.z();
      };

      double bx0 = 1e100, bx1 = -1e100, by0 = 1e100, by1 = -1e100;
      for (int i = 0; i < inst_brick.num_nodes; ++i) {
        const int idx = inst_brick.node_offset + i;
        bx0           = std::min(bx0, nodes2(idx, 0));
        bx1           = std::max(bx1, nodes2(idx, 0));
        by0           = std::min(by0, nodes2(idx, 1));
        by1           = std::max(by1, nodes2(idx, 1));
      }

      const double margin_xy2 = 0.05;
      bx0 -= margin_xy2;
      bx1 += margin_xy2;
      by0 -= margin_xy2;
      by1 += margin_xy2;

      std::vector<int> slope_candidates;
      slope_candidates.reserve(static_cast<size_t>(inst_slope.num_nodes));
      for (int i = 0; i < inst_slope.num_nodes; ++i) {
        const int idx = inst_slope.node_offset + i;
        const double x = nodes2(idx, 0);
        const double y = nodes2(idx, 1);
        if (x < bx0 || x > bx1 || y < by0 || y > by1) {
          continue;
        }
        slope_candidates.push_back(idx);
      }
      if (slope_candidates.empty()) {
        slope_candidates.reserve(static_cast<size_t>(inst_slope.num_nodes));
        for (int i = 0; i < inst_slope.num_nodes; ++i) {
          slope_candidates.push_back(inst_slope.node_offset + i);
        }
      }

      for (int i = 0; i < inst_brick.num_nodes; ++i) {
        const int bidx = inst_brick.node_offset + i;
        const double bx = nodes2(bidx, 0);
        const double by = nodes2(bidx, 1);
        const double bz = nodes2(bidx, 2);

        const double z_plane_b = PlaneZ(bx, by);
        double res_support      = -1e100;
        double nearest_d2 = 1e100;
        double nearest_res = -1e100;
        for (const int sidx : slope_candidates) {
          const double dx = nodes2(sidx, 0) - bx;
          const double dy = nodes2(sidx, 1) - by;
          const double d2 = dx * dx + dy * dy;
          const double z_plane_s = PlaneZ(nodes2(sidx, 0), nodes2(sidx, 1));
          const double res       = nodes2(sidx, 2) - z_plane_s;
          if (d2 < nearest_d2) {
            nearest_d2 = d2;
            nearest_res = res;
          }
          if (d2 <= neighbor_r2) {
            res_support = std::max(res_support, res);
          }
        }
        if (res_support < -1e50) {
          res_support = nearest_res;
        }

        const double z_support = z_plane_b + res_support;
        lift_z_needed =
            std::max(lift_z_needed, (z_support + init_clearance) - bz);
      }
    }

    if (lift_z_needed > 0.0) {
      mesh_manager.TranslateMesh(mesh_brick, 0.0, 0.0, lift_z_needed);
    }
  }

  const Eigen::MatrixXd& initial_nodes = mesh_manager.GetAllNodes();
  const Eigen::MatrixXi& elements      = mesh_manager.GetAllElements();

  const double brick_volume = ComputeTetMeshVolume_T10_UsingCorners(
      initial_nodes, elements, inst_brick.element_offset,
      inst_brick.num_elements);
  if (brick_volume <= 0.0) {
    std::cerr << "Invalid brick volume computed: " << brick_volume << std::endl;
    return 1;
  }
  const double brick_rho0 = brick_mass_kg / brick_volume;
  std::cout << "Brick volume: " << std::setprecision(15) << brick_volume
            << " m^3" << std::endl;
  std::cout << "Brick rho0: " << brick_rho0 << " kg/m^3" << std::endl;

  const int n_nodes = mesh_manager.GetTotalNodes();
  const int n_elems = mesh_manager.GetTotalElements();

  GPU_FEAT10_Data gpu_t10_data(n_elems, n_nodes);
  gpu_t10_data.Initialize();

  Eigen::VectorXd h_x12(n_nodes), h_y12(n_nodes), h_z12(n_nodes);
  for (int i = 0; i < n_nodes; i++) {
    h_x12(i) = initial_nodes(i, 0);
    h_y12(i) = initial_nodes(i, 1);
    h_z12(i) = initial_nodes(i, 2);
  }

  std::vector<int> fixed_node_indices;
  fixed_node_indices.reserve(static_cast<size_t>(inst_slope.num_nodes));
  for (int i = 0; i < inst_slope.num_nodes; ++i) {
    fixed_node_indices.push_back(inst_slope.node_offset + i);
  }
  Eigen::VectorXi h_fixed_nodes(fixed_node_indices.size());
  for (size_t i = 0; i < fixed_node_indices.size(); ++i) {
    h_fixed_nodes(i) = fixed_node_indices[i];
  }
  gpu_t10_data.SetNodalFixed(h_fixed_nodes);

  const Eigen::VectorXd& tet5pt_x       = Quadrature::tet5pt_x;
  const Eigen::VectorXd& tet5pt_y       = Quadrature::tet5pt_y;
  const Eigen::VectorXd& tet5pt_z       = Quadrature::tet5pt_z;
  const Eigen::VectorXd& tet5pt_weights = Quadrature::tet5pt_weights;

  gpu_t10_data.Setup(tet5pt_x, tet5pt_y, tet5pt_z, tet5pt_weights, h_x12, h_y12,
                     h_z12, elements);

  const SolidMaterialProperties mat_default =
      SolidMaterialProperties::SVK(1e7, 0.3, 1000.0, 2e4, 2e4);
  gpu_t10_data.ApplyMaterial(mat_default);

  gpu_t10_data.SetDensityForElementRange(inst_slope.element_offset,
                                        inst_slope.num_elements, 1000.0);
  gpu_t10_data.SetDensityForElementRange(inst_brick.element_offset,
                                        inst_brick.num_elements, brick_rho0);

  gpu_t10_data.CalcDnDuPre();
  gpu_t10_data.CalcMassMatrix();
  gpu_t10_data.CalcConstraintData();
  gpu_t10_data.ConvertToCSR_ConstraintJacT();
  gpu_t10_data.BuildConstraintJacobianCSR();

  Eigen::VectorXd lumped_mass(n_nodes);
  lumped_mass.setZero();
  {
    std::vector<int> offsets;
    std::vector<int> columns;
    std::vector<double> values;
    gpu_t10_data.RetrieveMassCSRToCPU(offsets, columns, values);
    if (static_cast<int>(offsets.size()) == n_nodes + 1) {
      for (int i = 0; i < n_nodes; ++i) {
        double sum = 0.0;
        for (int k = offsets[i]; k < offsets[i + 1]; ++k) {
          sum += values[k];
        }
        lumped_mass(i) = sum;
      }
    } else {
      lumped_mass.setOnes();
    }
  }

  SyncedNewtonParams params = {1e-6, 0.0, 1e-6, 1e12, 3, 10, dt};
  SyncedNewtonSolver solver(&gpu_t10_data, gpu_t10_data.get_n_constraint());
  solver.Setup();
  solver.SetParameters(&params);
  solver.AnalyzeHessianSparsity();

  double* d_vel_guess = solver.GetVelocityGuessDevicePtr();
  HANDLE_ERROR(cudaMemset(d_vel_guess, 0, n_nodes * 3 * sizeof(double)));

  ANCFCPUUtils::SurfaceTriMesh slope_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(initial_nodes, elements, inst_slope);
  ANCFCPUUtils::SurfaceTriMesh brick_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(initial_nodes, elements, inst_brick);

  std::vector<DemeMeshCollisionBody> bodies;
  {
    DemeMeshCollisionBody body;
    body.surface                 = std::move(slope_surface);
    body.family                  = 0;
    body.split_into_patches      = false;
    body.skip_self_contact_forces = true;
    body.mass                    = 1000.0f;
    bodies.push_back(std::move(body));
  }
  {
    DemeMeshCollisionBody body;
    body.surface            = std::move(brick_surface);
    body.family             = 1;
    body.split_into_patches = true;
    body.mass               = static_cast<float>(brick_mass_kg);
    bodies.push_back(std::move(body));
  }

  auto collision_system = std::make_unique<DemeMeshCollisionSystem>(
      std::move(bodies), contact_mu_s, contact_mu_k, contact_stiffness,
      contact_cor, enable_self_collision, dt);

  double* d_nodes_collision = nullptr;
  HANDLE_ERROR(cudaMalloc(&d_nodes_collision, n_nodes * 3 * sizeof(double)));
  HANDLE_ERROR(cudaMemcpy(d_nodes_collision, gpu_t10_data.GetX12DevicePtr(),
                          n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));
  HANDLE_ERROR(cudaMemcpy(d_nodes_collision + n_nodes,
                          gpu_t10_data.GetY12DevicePtr(),
                          n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));
  HANDLE_ERROR(cudaMemcpy(d_nodes_collision + 2 * n_nodes,
                          gpu_t10_data.GetZ12DevicePtr(),
                          n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));

  collision_system->BindNodesDevicePtr(d_nodes_collision, n_nodes);

  Eigen::VectorXd h_f_gravity = Eigen::VectorXd::Zero(n_nodes * 3);
  for (int i = 0; i < inst_brick.num_nodes; ++i) {
    const int idx = inst_brick.node_offset + i;
    h_f_gravity(3 * idx + 2) += lumped_mass(idx) * gravity;
  }

  double* d_f_gravity = nullptr;
  HANDLE_ERROR(cudaMalloc(&d_f_gravity, n_nodes * 3 * sizeof(double)));
  HANDLE_ERROR(cudaMemcpy(d_f_gravity, h_f_gravity.data(),
                          n_nodes * 3 * sizeof(double),
                          cudaMemcpyHostToDevice));

  cublasHandle_t cublas_handle = nullptr;
  CheckCublas(cublasCreate(&cublas_handle), "cublasCreate");

  Eigen::VectorXd displacement(n_nodes * 3);
  displacement.setZero();

  for (int step = 0; step < max_steps; ++step) {
    HANDLE_ERROR(cudaMemcpy(d_nodes_collision, gpu_t10_data.GetX12DevicePtr(),
                            n_nodes * sizeof(double),
                            cudaMemcpyDeviceToDevice));
    HANDLE_ERROR(cudaMemcpy(d_nodes_collision + n_nodes,
                            gpu_t10_data.GetY12DevicePtr(),
                            n_nodes * sizeof(double),
                            cudaMemcpyDeviceToDevice));
    HANDLE_ERROR(cudaMemcpy(d_nodes_collision + 2 * n_nodes,
                            gpu_t10_data.GetZ12DevicePtr(),
                            n_nodes * sizeof(double),
                            cudaMemcpyDeviceToDevice));

    CollisionSystemInput coll_in;
    coll_in.d_nodes_xyz = d_nodes_collision;
    coll_in.n_nodes     = n_nodes;
    coll_in.d_vel_xyz   = d_vel_guess;
    coll_in.dt          = dt;

    CollisionSystemParams coll_params;
    coll_params.damping  = 0.4;
    coll_params.friction = contact_mu_k;

    collision_system->Step(coll_in, coll_params);
    const int num_contacts = collision_system->GetNumContacts();

    HANDLE_ERROR(cudaMemcpy(gpu_t10_data.GetExternalForceDevicePtr(),
                            d_f_gravity, n_nodes * 3 * sizeof(double),
                            cudaMemcpyDeviceToDevice));

    if (num_contacts > 0) {
      const double alpha_axpy = 1.0;
      CheckCublas(cublasDaxpy(cublas_handle, n_nodes * 3, &alpha_axpy,
                              collision_system->GetExternalForcesDevicePtr(), 1,
                              gpu_t10_data.GetExternalForceDevicePtr(), 1),
                  "cublasDaxpy(contact + gravity)");
    }

    solver.Solve();

    if (export_interval > 0 && step % export_interval == 0) {
      Eigen::VectorXd x12_current, y12_current, z12_current;
      gpu_t10_data.RetrievePositionToCPU(x12_current, y12_current, z12_current);

      Eigen::MatrixXd current_nodes(n_nodes, 3);
      for (int i = 0; i < n_nodes; ++i) {
        current_nodes(i, 0)     = x12_current(i);
        current_nodes(i, 1)     = y12_current(i);
        current_nodes(i, 2)     = z12_current(i);
        displacement(3 * i)     = x12_current(i) - initial_nodes(i, 0);
        displacement(3 * i + 1) = y12_current(i) - initial_nodes(i, 1);
        displacement(3 * i + 2) = z12_current(i) - initial_nodes(i, 2);
      }

      std::ostringstream filename;
      filename << output_dir << "/mesh_" << std::setfill('0') << std::setw(4)
               << step << ".vtu";
      VisualizationUtils::ExportMeshWithDisplacement(current_nodes, elements,
                                                     displacement,
                                                     filename.str());
    }

    if (step % 50 == 0) {
      std::cout << "Step " << std::setw(4) << step
                << ": contacts=" << std::setw(5) << num_contacts << std::endl;
    }
  }

  cublasDestroy(cublas_handle);
  HANDLE_ERROR(cudaFree(d_nodes_collision));
  HANDLE_ERROR(cudaFree(d_f_gravity));
  gpu_t10_data.Destroy();

  std::cout << "Output files in: " << output_dir << "/" << std::endl;
  return 0;
}
