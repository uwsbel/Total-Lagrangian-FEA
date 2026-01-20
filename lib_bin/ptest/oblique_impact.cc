// ./bazel-bin/lib_bin/ptest/oblique_impact --theta=0.3 --v_i=2.0
#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "../../lib_src/collision/DemeMeshCollisionSystem.h"
#include "../../lib_src/elements/FEAT10Data.cuh"
#include "../../lib_src/solvers/SyncedNewton.cuh"
#include "../../lib_utils/cli_utils.h"
#include "../../lib_utils/mesh_manager.h"
#include "../../lib_utils/quadrature_utils.h"
#include "../../lib_utils/surface_trimesh_extract.h"
#include "../../lib_utils/visualization_utils.h"

using ANCFCPUUtils::VisualizationUtils;



static void AxisMinMaxForInstance(const Eigen::MatrixXd& nodes,
                                  const ANCFCPUUtils::MeshInstance& inst,
                                  int axis, double* out_min,
                                  double* out_max) {
  double vmin = 1e300;
  double vmax = -1e300;
  for (int i = 0; i < inst.num_nodes; ++i) {
    const int idx = inst.node_offset + i;
    const double v = nodes(idx, axis);
    vmin = std::min(vmin, v);
    vmax = std::max(vmax, v);
  }
  *out_min = vmin;
  *out_max = vmax;
}

int main(int argc, char** argv) {
  double theta = 0.0;
  double v_i   = 1.0;

  double contact_mu_s = 0.01;
  double contact_mu_k = 0.01;
  double contact_cor  = 0.4;
  const double contact_stiffness = 1e8;

  const double dt = 2e-4;
  int max_steps = 1000;
  int export_interval = 5;
  std::string out_suffix;

  const bool has_flag_args =
      (argc > 1 && argv[1] &&
       (std::string_view(argv[1]) == "--help" ||
        std::string_view(argv[1]) == "-h" ||
        std::string_view(argv[1]).rfind("--", 0) == 0));

  if (has_flag_args) {
    ANCFCPUUtils::Cli cli(argv[0] ? argv[0] : "oblique_impact");
    cli.SetDescription(
        "Oblique impact demo (sphere vs plate, no gravity). The plate is rotated "
        "into the x-z plane so motion is in the x-y plane.");
    cli.AddDouble("theta", theta, "impact angle in radians (0 = normal impact)");
    cli.AddDouble("v_i", v_i, "initial speed magnitude");
    cli.AddDouble("cor", contact_cor, "contact restitution (CoR), range [0, 1]");
    cli.AddDouble("mu_s", contact_mu_s, "contact static friction mu_s");
    cli.AddDouble("mu_k", contact_mu_k, "contact kinetic friction mu_k");
    cli.AddInt("steps", max_steps, "max simulation steps");
    cli.AddInt("export_interval", export_interval, "VTK export interval (0 disables)");
    cli.AddString("out_suffix", "", "suffix appended to output folder name");

    std::string err;
    if (!cli.Parse(argc, argv, &err) || cli.HelpRequested()) {
      if (!err.empty()) std::cerr << "Error: " << err << "\n\n";
      cli.PrintUsage(std::cerr);
      return cli.HelpRequested() ? 0 : 1;
    }

    theta = cli.GetDouble("theta");
    v_i = cli.GetDouble("v_i");
    contact_cor = cli.GetDouble("cor");
    contact_mu_s = cli.GetDouble("mu_s");
    contact_mu_k = cli.GetDouble("mu_k");
    max_steps = cli.GetInt("steps");
    export_interval = cli.GetInt("export_interval");
    out_suffix = cli.GetString("out_suffix");
  } else {
    if (argc > 1) theta = std::atof(argv[1]);
    if (argc > 2) v_i = std::atof(argv[2]);
  }

  if (v_i < 0.0) {
    std::cerr << "Invalid v_i: " << v_i << " (expected >= 0)" << std::endl;
    return 1;
  }
  if (out_suffix.find('/') != std::string::npos ||
      out_suffix.find('\\') != std::string::npos) {
    std::cerr << "Invalid --out_suffix: " << out_suffix
              << " (must not contain path separators)" << std::endl;
    return 1;
  }

  std::string output_dir = "output/oblique_impact";
  if (!out_suffix.empty()) {
    output_dir += "_" + out_suffix;
  }
  std::filesystem::create_directories(output_dir);

  const double v_n = v_i * std::cos(theta);
  const double v_t = v_i * std::sin(theta);

  std::cout << "theta: " << theta << " rad\n";
  std::cout << "v_i:   " << v_i << "\n";
  std::cout << "v_x:   " << v_t << "\n";
  std::cout << "v_y:   " << -v_n << "\n";

  ANCFCPUUtils::MeshManager mesh_manager;

  const std::string sphere_prefix =
      "data/meshes/ptest/sphere/sphere_r0p15_med.1";
  const std::string sphere_node = sphere_prefix + ".node";
  const std::string sphere_ele  = sphere_prefix + ".ele";
  const std::string plate_node  = "data/meshes/ptest/plate/1.1.001plate.1.node";
  const std::string plate_ele   = "data/meshes/ptest/plate/1.1.001plate.1.ele";

  int mesh_plate  = mesh_manager.LoadMesh(plate_node, plate_ele, "plate");
  int mesh_sphere = mesh_manager.LoadMesh(sphere_node, sphere_ele, "sphere");
  if (mesh_plate < 0 || mesh_sphere < 0) {
    std::cerr << "Failed to load meshes" << std::endl;
    return 1;
  }

  const double pi = std::acos(-1.0);
  mesh_manager.TransformMesh(mesh_plate, ANCFCPUUtils::rotationX(-pi / 2.0));

  const auto& inst_plate  = mesh_manager.GetMeshInstance(mesh_plate);
  const auto& inst_sphere = mesh_manager.GetMeshInstance(mesh_sphere);

  {
    const Eigen::MatrixXd& nodes = mesh_manager.GetAllNodes();
    double plate_y_min, plate_y_max;
    double sphere_y_min, sphere_y_max;
    AxisMinMaxForInstance(nodes, inst_plate, 1, &plate_y_min, &plate_y_max);
    AxisMinMaxForInstance(nodes, inst_sphere, 1, &sphere_y_min, &sphere_y_max);

    const double clearance = 2e-3;
    const double dy = (plate_y_max + clearance) - sphere_y_min;
    mesh_manager.TranslateMesh(mesh_sphere, 0.0, dy, 0.0);

    const double standoff = 0.10;
    const double v_norm = std::sqrt(v_t * v_t + v_n * v_n);
    if (v_norm > 0.0) {
      const double dir_x = v_t / v_norm;
      const double dir_y = (-v_n) / v_norm;
      mesh_manager.TranslateMesh(mesh_sphere, -dir_x * standoff,
                                -dir_y * standoff, 0.0);
    }
  }

  const Eigen::MatrixXd& initial_nodes = mesh_manager.GetAllNodes();
  const Eigen::MatrixXi& elements      = mesh_manager.GetAllElements();

  const int n_nodes = mesh_manager.GetTotalNodes();
  const int n_elems = mesh_manager.GetTotalElements();

  std::cout << "Total nodes: " << n_nodes << "\n";
  std::cout << "Total elements: " << n_elems << "\n";

  GPU_FEAT10_Data gpu_t10_data(n_elems, n_nodes);
  gpu_t10_data.Initialize();

  Eigen::VectorXd h_x12(n_nodes), h_y12(n_nodes), h_z12(n_nodes);
  for (int i = 0; i < n_nodes; ++i) {
    h_x12(i) = initial_nodes(i, 0);
    h_y12(i) = initial_nodes(i, 1);
    h_z12(i) = initial_nodes(i, 2);
  }

  {
    std::vector<int> fixed_node_indices;
    fixed_node_indices.reserve(inst_plate.num_nodes);
    for (int i = 0; i < inst_plate.num_nodes; ++i) {
      fixed_node_indices.push_back(inst_plate.node_offset + i);
    }
    Eigen::VectorXi h_fixed_nodes(fixed_node_indices.size());
    for (size_t i = 0; i < fixed_node_indices.size(); ++i) {
      h_fixed_nodes(i) = fixed_node_indices[i];
    }
    gpu_t10_data.SetNodalFixed(h_fixed_nodes);
  }

  const Eigen::VectorXd& tet5pt_x       = Quadrature::tet5pt_x;
  const Eigen::VectorXd& tet5pt_y       = Quadrature::tet5pt_y;
  const Eigen::VectorXd& tet5pt_z       = Quadrature::tet5pt_z;
  const Eigen::VectorXd& tet5pt_weights = Quadrature::tet5pt_weights;

  gpu_t10_data.Setup(tet5pt_x, tet5pt_y, tet5pt_z, tet5pt_weights, h_x12, h_y12,
                     h_z12, elements);

  const SolidMaterialProperties mat_sphere = SolidMaterialProperties::SVK(
      1e7, 0.3, 70.7355, 2e4, 2e4);
  const SolidMaterialProperties mat_plate = SolidMaterialProperties::SVK(
      1e7, 0.3, 70.7355, 2e4, 2e4);

  gpu_t10_data.ApplyMaterial(mat_sphere);
  gpu_t10_data.SetDensityForElementRange(inst_plate.element_offset,
                                        inst_plate.num_elements, mat_plate.rho0);
  gpu_t10_data.SetDensityForElementRange(inst_sphere.element_offset,
                                        inst_sphere.num_elements, mat_sphere.rho0);

  gpu_t10_data.CalcDnDuPre();
  gpu_t10_data.CalcMassMatrix();
  gpu_t10_data.CalcConstraintData();
  gpu_t10_data.ConvertToCSR_ConstraintJacT();
  gpu_t10_data.BuildConstraintJacobianCSR();

  SyncedNewtonParams params = {1e-6, 0.0, 1e-6, 1e12, 3, 10, dt};
  SyncedNewtonSolver solver(&gpu_t10_data, gpu_t10_data.get_n_constraint());
  solver.Setup();
  solver.SetParameters(&params);
  solver.AnalyzeHessianSparsity();

  Eigen::VectorXd h_v0 = Eigen::VectorXd::Zero(n_nodes * 3);
  for (int i = 0; i < inst_sphere.num_nodes; ++i) {
    const int idx = inst_sphere.node_offset + i;
    h_v0(3 * idx + 0) = v_t;
    h_v0(3 * idx + 1) = -v_n;
    h_v0(3 * idx + 2) = 0.0;
  }
  solver.SetInitialVelocity(h_v0);

  double* d_vel_guess = solver.GetVelocityGuessDevicePtr();

  ANCFCPUUtils::SurfaceTriMesh plate_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(initial_nodes, elements, inst_plate);
  ANCFCPUUtils::SurfaceTriMesh sphere_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(initial_nodes, elements, inst_sphere);

  const double sphere_radius = 0.15;
  const double sphere_volume =
      (4.0 / 3.0) * pi * sphere_radius * sphere_radius * sphere_radius;
  const float sphere_mass =
      static_cast<float>(mat_sphere.rho0 * sphere_volume);

  std::vector<DemeMeshCollisionBody> bodies;
  {
    DemeMeshCollisionBody body;
    body.surface = std::move(plate_surface);
    body.family = 0;
    body.split_into_patches = false;
    body.skip_self_contact_forces = true;
    body.mass = 1000.0f;
    bodies.push_back(std::move(body));
  }
  {
    DemeMeshCollisionBody body;
    body.surface = std::move(sphere_surface);
    body.family = 1;
    body.split_into_patches = true;
    body.mass = sphere_mass;
    bodies.push_back(std::move(body));
  }

  auto collision_system = std::make_unique<DemeMeshCollisionSystem>(
      std::move(bodies), contact_mu_s, contact_mu_k, contact_stiffness,
      contact_cor, false, dt);

  double* d_nodes_collision = nullptr;
  HANDLE_ERROR(cudaMalloc(&d_nodes_collision,
                          static_cast<size_t>(n_nodes) * 3 * sizeof(double)));
  HANDLE_ERROR(cudaMemcpy(d_nodes_collision, gpu_t10_data.GetX12DevicePtr(),
                          n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));
  HANDLE_ERROR(cudaMemcpy(d_nodes_collision + n_nodes, gpu_t10_data.GetY12DevicePtr(),
                          n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));
  HANDLE_ERROR(cudaMemcpy(d_nodes_collision + 2 * n_nodes, gpu_t10_data.GetZ12DevicePtr(),
                          n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));

  collision_system->BindNodesDevicePtr(d_nodes_collision, n_nodes);

  Eigen::VectorXd displacement(n_nodes * 3);
  displacement.setZero();

  for (int step = 0; step < max_steps; ++step) {
    HANDLE_ERROR(cudaMemcpy(d_nodes_collision, gpu_t10_data.GetX12DevicePtr(),
                            n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));
    HANDLE_ERROR(cudaMemcpy(d_nodes_collision + n_nodes,
                            gpu_t10_data.GetY12DevicePtr(),
                            n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));
    HANDLE_ERROR(cudaMemcpy(d_nodes_collision + 2 * n_nodes,
                            gpu_t10_data.GetZ12DevicePtr(),
                            n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));

    CollisionSystemInput coll_in;
    coll_in.d_nodes_xyz = d_nodes_collision;
    coll_in.n_nodes     = n_nodes;
    coll_in.d_vel_xyz   = d_vel_guess;
    coll_in.dt          = dt;

    CollisionSystemParams coll_params;
    coll_params.damping  = 0.0;
    coll_params.friction = contact_mu_k;

    collision_system->Step(coll_in, coll_params);

    HANDLE_ERROR(cudaMemcpy(gpu_t10_data.GetExternalForceDevicePtr(),
                            collision_system->GetExternalForcesDevicePtr(),
                            static_cast<size_t>(n_nodes) * 3 * sizeof(double),
                            cudaMemcpyDeviceToDevice));

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
  }

  HANDLE_ERROR(cudaFree(d_nodes_collision));
  gpu_t10_data.Destroy();

  std::cout << "Output files in: " << output_dir << "/" << std::endl;
  return 0;
}
