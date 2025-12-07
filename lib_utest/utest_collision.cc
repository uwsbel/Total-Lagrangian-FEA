#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <iostream>

#include "../lib_src/collision/Broadphase.cuh"
#include "../lib_src/collision/Narrowphase.cuh"
#include "../lib_utils/cpu_utils.h"
#include "../lib_utils/mesh_manager.h"
#include "../lib_utils/mesh_utils.h"
#include "../lib_utils/visualization_utils.h"

// Test collision class
class TestCollision : public ::testing::Test {
 protected:
  void SetUp() override {
    // Setup code that runs before each test
  }

  void TearDown() override {
    // Cleanup code that runs after each test
  }
};

TEST_F(TestCollision, BroadphaseInitialization) {
  // Initialize the Broadphase collision detection object
  Broadphase broadphase;

  // Read mesh data
  Eigen::MatrixXd nodes;
  Eigen::MatrixXi elements;

  int n_nodes =
      ANCFCPUUtils::FEAT10_read_nodes("data/meshes/T10/sphere.1.node", nodes);
  int n_elems = ANCFCPUUtils::FEAT10_read_elements(
      "data/meshes/T10/sphere.1.ele", elements);

  ASSERT_GT(n_nodes, 0);
  ASSERT_GT(n_elems, 0);

  // Initialize broadphase with mesh data
  broadphase.Initialize(nodes, elements);
  broadphase.CreateAABB();
  broadphase.BuildNeighborMap();  // NEW: Build neighbor connectivity
  broadphase.SortAABBs(0);
  broadphase.DetectCollisions();  // Now filters neighbors!
  broadphase.PrintCollisionPairs();
}

TEST_F(TestCollision, BroadphaseMultipleMeshes) {
  ANCFCPUUtils::MeshManager mesh_manager;
  Broadphase broadphase;

  // Load first sphere
  int mesh0 = mesh_manager.LoadMesh("data/meshes/T10/sphere.1.node",
                                    "data/meshes/T10/sphere.1.ele", "sphere_0");
  ASSERT_GE(mesh0, 0) << "Failed to load first sphere mesh";

  // Load second sphere
  int mesh1 = mesh_manager.LoadMesh("data/meshes/T10/sphere.1.node",
                                    "data/meshes/T10/sphere.1.ele", "sphere_1");
  ASSERT_GE(mesh1, 0) << "Failed to load second sphere mesh";

  // Translate second sphere by 0.1 in X direction
  mesh_manager.TranslateMesh(mesh1, 0.1, 0.0, 0.0);

  std::cout << "Loaded " << mesh_manager.GetNumMeshes() << " meshes"
            << std::endl;
  std::cout << "Total nodes: " << mesh_manager.GetTotalNodes() << std::endl;
  std::cout << "Total elements: " << mesh_manager.GetTotalElements()
            << std::endl;

  // Print mesh instance info
  for (int i = 0; i < mesh_manager.GetNumMeshes(); ++i) {
    const auto& instance = mesh_manager.GetMeshInstance(i);
    std::cout << "Mesh " << i << " (" << instance.name
              << "): node_offset=" << instance.node_offset
              << ", element_offset=" << instance.element_offset
              << ", num_nodes=" << instance.num_nodes
              << ", num_elements=" << instance.num_elements << std::endl;
  }

  // Initialize broadphase with unified mesh data
  broadphase.Initialize(mesh_manager.GetAllNodes(),
                        mesh_manager.GetAllElements());
  broadphase.CreateAABB();
  broadphase.BuildNeighborMap();
  broadphase.SortAABBs(0);
  broadphase.DetectCollisions();
  broadphase.PrintCollisionPairs();
}

TEST_F(TestCollision, NarrowphaseContactPatches) {
  // Full collision detection pipeline: Broadphase + Narrowphase
  // Using hydroelastic pressure fields computed based on mesh geometry
  ANCFCPUUtils::MeshManager mesh_manager;
  Broadphase broadphase;
  Narrowphase narrowphase;

  // Load first sphere
  int mesh0 = mesh_manager.LoadMesh("data/meshes/T10/sphere.1.node",
                                    "data/meshes/T10/sphere.1.ele", "sphere_0");
  ASSERT_GE(mesh0, 0) << "Failed to load first sphere mesh";

  // Load second sphere
  int mesh1 = mesh_manager.LoadMesh("data/meshes/T10/sphere.1.node",
                                    "data/meshes/T10/sphere.1.ele", "sphere_1");
  ASSERT_GE(mesh1, 0) << "Failed to load second sphere mesh";

  // Translate second sphere by 0.1 in X direction (for overlap)
  mesh_manager.TranslateMesh(mesh1, 0.2, 0.0, 0.1);

  std::cout << "\n========== Narrowphase Test ==========\n" << std::endl;
  std::cout << "Loaded " << mesh_manager.GetNumMeshes() << " meshes"
            << std::endl;
  std::cout << "Total nodes: " << mesh_manager.GetTotalNodes() << std::endl;
  std::cout << "Total elements: " << mesh_manager.GetTotalElements()
            << std::endl;

  // Get unified mesh data
  const Eigen::MatrixXd& nodes    = mesh_manager.GetAllNodes();
  const Eigen::MatrixXi& elements = mesh_manager.GetAllElements();

  // Compute hydroelastic pressure fields for each mesh separately
  // p(x) = Eh * max(0, R - ||x - center||)
  // Each mesh has its own center based on its current (transformed) position

  const auto& instance0 = mesh_manager.GetMeshInstance(mesh0);
  const auto& instance1 = mesh_manager.GetMeshInstance(mesh1);

  // Compute center of mesh 0 (in global coordinates)
  Eigen::Vector3d center0 = Eigen::Vector3d::Zero();
  for (int i = 0; i < instance0.num_nodes; ++i) {
    int global_idx = instance0.node_offset + i;
    center0 += nodes.row(global_idx).transpose();
  }
  center0 /= instance0.num_nodes;

  // Compute center of mesh 1 (in global coordinates, after translation)
  Eigen::Vector3d center1 = Eigen::Vector3d::Zero();
  for (int i = 0; i < instance1.num_nodes; ++i) {
    int global_idx = instance1.node_offset + i;
    center1 += nodes.row(global_idx).transpose();
  }
  center1 /= instance1.num_nodes;

  // Estimate radius from mesh 0
  double R = 0.0;
  for (int i = 0; i < instance0.num_nodes; ++i) {
    int global_idx = instance0.node_offset + i;
    double dist    = (nodes.row(global_idx).transpose() - center0).norm();
    R              = std::max(R, dist);
  }

  double Eh = 1.0;  // Hydroelastic modulus

  std::cout << "Mesh 0 center: (" << center0.x() << ", " << center0.y() << ", "
            << center0.z() << ")" << std::endl;
  std::cout << "Mesh 1 center: (" << center1.x() << ", " << center1.y() << ", "
            << center1.z() << ")" << std::endl;
  std::cout << "Estimated radius R: " << R << std::endl;

  // Compute pressure field for mesh 0: p(x) = Eh * max(0, R - ||x - center0||)
  Eigen::VectorXd pressure0(instance0.num_nodes);
  for (int i = 0; i < instance0.num_nodes; ++i) {
    int global_idx = instance0.node_offset + i;
    double dist    = (nodes.row(global_idx).transpose() - center0).norm();
    pressure0(i)   = Eh * std::max(0.0, R - dist);
  }
  mesh_manager.SetScalarField(mesh0, pressure0);

  // Compute pressure field for mesh 1: p(x) = Eh * max(0, R - ||x - center1||)
  // Note: This uses center1 (the translated center), not center0!
  Eigen::VectorXd pressure1(instance1.num_nodes);
  for (int i = 0; i < instance1.num_nodes; ++i) {
    int global_idx = instance1.node_offset + i;
    double dist    = (nodes.row(global_idx).transpose() - center1).norm();
    pressure1(i)   = Eh * std::max(0.0, R - dist);
  }
  mesh_manager.SetScalarField(mesh1, pressure1);

  // Get unified scalar field
  const Eigen::VectorXd& pressure = mesh_manager.GetAllScalarFields();
  std::cout << "Computed hydroelastic pressure fields" << std::endl;
  std::cout << "Scalar field size: " << pressure.size() << std::endl;
  std::cout << "Pressure 0 range: [" << pressure0.minCoeff() << ", "
            << pressure0.maxCoeff() << "]" << std::endl;
  std::cout << "Pressure 1 range: [" << pressure1.minCoeff() << ", "
            << pressure1.maxCoeff() << "]" << std::endl;

  // ===== Broadphase =====
  broadphase.Initialize(nodes, elements);
  broadphase.CreateAABB();
  broadphase.BuildNeighborMap();
  broadphase.SortAABBs(0);
  broadphase.DetectCollisions();

  std::cout << "\nBroadphase found " << broadphase.numCollisions
            << " potential collision pairs" << std::endl;

  // Convert broadphase results to pair vector for narrowphase
  std::vector<std::pair<int, int>> collisionPairs;
  for (const auto& cp : broadphase.h_collisionPairs) {
    collisionPairs.emplace_back(cp.idA, cp.idB);
  }

  // ===== Narrowphase =====
  // Build element-to-mesh ID mapping
  Eigen::VectorXi elementMeshIds(mesh_manager.GetTotalElements());
  for (int i = 0; i < mesh_manager.GetNumMeshes(); ++i) {
    const auto& instance = mesh_manager.GetMeshInstance(i);
    for (int e = 0; e < instance.num_elements; ++e) {
      elementMeshIds(instance.element_offset + e) = i;
    }
  }

  narrowphase.Initialize(nodes, elements, pressure, elementMeshIds);
  narrowphase.SetCollisionPairs(collisionPairs);
  narrowphase.ComputeContactPatches();
  narrowphase.RetrieveResults();

  // Print summary (not verbose for large meshes)
  narrowphase.PrintContactPatches(false);

  // Get valid patches
  auto validPatches = narrowphase.GetValidPatches();
  std::cout << "Retrieved " << validPatches.size() << " valid contact patches"
            << std::endl;

  // Export visualization files
  std::cout << "\n--- Exporting visualization files ---\n" << std::endl;

  // Export contact patches to VTP (ParaView)
  ANCFCPUUtils::VisualizationUtils::ExportContactPatchesToVTP(
      validPatches, "output/contact_patches.vtp");

  // Export contact patches to CSV (Python/MATLAB analysis)
  ANCFCPUUtils::VisualizationUtils::ExportContactPatchesToCSV(
      validPatches, "output/contact_patches.csv");

  // Export contact patches to JSON (debugging)
  ANCFCPUUtils::VisualizationUtils::ExportContactPatchesToJSON(
      validPatches, "output/contact_patches.json");

  // Export normal vectors as arrows
  ANCFCPUUtils::VisualizationUtils::ExportNormalsAsArrows(
      validPatches, 0.02, "output/contact_normals.vtp");

  // Export mesh with scalar field to VTU
  ANCFCPUUtils::VisualizationUtils::ExportMeshToVTU(
      nodes, elements, pressure, "output/mesh_with_pressure.vtu");

  // Print first few patches in detail
  int printCount = std::min(5, (int)validPatches.size());
  if (printCount > 0) {
    std::cout << "\n--- First " << printCount << " contact patches ---\n"
              << std::endl;
    for (int i = 0; i < printCount; ++i) {
      const auto& patch = validPatches[i];
      std::cout << "Patch " << i << " (Tet " << patch.tetA_idx << " <-> Tet "
                << patch.tetB_idx << "):" << std::endl;
      std::cout << "  Vertices: " << patch.numVertices << std::endl;
      std::cout << "  Normal: (" << patch.normal.x << ", " << patch.normal.y
                << ", " << patch.normal.z << ")" << std::endl;
      std::cout << "  Centroid: (" << patch.centroid.x << ", "
                << patch.centroid.y << ", " << patch.centroid.z << ")"
                << std::endl;
      std::cout << "  Area: " << patch.area << std::endl;
      std::cout << "  g_A: " << patch.g_A << ", g_B: " << patch.g_B
                << std::endl;
      std::cout << "  p_equilibrium: " << patch.p_equilibrium << std::endl;
      std::cout << "  Valid orientation: "
                << (patch.validOrientation ? "Yes" : "No") << std::endl;
      std::cout << std::endl;
    }
  }

  // Basic sanity checks
  EXPECT_GT(validPatches.size(), 0) << "Expected some valid contact patches";

  // Check that normals are unit vectors
  for (const auto& patch : validPatches) {
    double len =
        sqrt(patch.normal.x * patch.normal.x + patch.normal.y * patch.normal.y +
             patch.normal.z * patch.normal.z);
    EXPECT_NEAR(len, 1.0, 1e-6) << "Normal should be unit length";
  }
}

TEST_F(TestCollision, ThreeSpheresContactPatches) {
  // Three spheres collision test
  // Sphere 0: at origin
  // Sphere 1: translated by [0.2, 0.0, 0.1]
  // Sphere 2: translated by [0.1, 0.2, 0.1]
  ANCFCPUUtils::MeshManager mesh_manager;
  Broadphase broadphase;
  Narrowphase narrowphase;

  // Load first sphere (at origin)
  int mesh0 = mesh_manager.LoadMesh("data/meshes/T10/sphere.1.node",
                                    "data/meshes/T10/sphere.1.ele", "sphere_0");
  ASSERT_GE(mesh0, 0) << "Failed to load first sphere mesh";

  // Load second sphere
  int mesh1 = mesh_manager.LoadMesh("data/meshes/T10/sphere.1.node",
                                    "data/meshes/T10/sphere.1.ele", "sphere_1");
  ASSERT_GE(mesh1, 0) << "Failed to load second sphere mesh";

  // Load third sphere
  int mesh2 = mesh_manager.LoadMesh("data/meshes/T10/sphere.1.node",
                                    "data/meshes/T10/sphere.1.ele", "sphere_2");
  ASSERT_GE(mesh2, 0) << "Failed to load third sphere mesh";

  // Translate spheres
  mesh_manager.TranslateMesh(mesh1, 0.2, 0.0, 0.1);  // Same as two-sphere test
  mesh_manager.TranslateMesh(mesh2, 0.1, 0.2, 0.1);  // New third sphere

  std::cout << "\n========== Three Spheres Test ==========\n" << std::endl;
  std::cout << "Loaded " << mesh_manager.GetNumMeshes() << " meshes"
            << std::endl;
  std::cout << "Total nodes: " << mesh_manager.GetTotalNodes() << std::endl;
  std::cout << "Total elements: " << mesh_manager.GetTotalElements()
            << std::endl;

  // Get unified mesh data
  const Eigen::MatrixXd& nodes    = mesh_manager.GetAllNodes();
  const Eigen::MatrixXi& elements = mesh_manager.GetAllElements();

  // Compute hydroelastic pressure fields for each mesh
  double Eh = 1.0;  // Hydroelastic modulus

  // First, compute centers and estimate radius
  std::vector<Eigen::Vector3d> centers(3);
  for (int m = 0; m < 3; ++m) {
    const auto& instance = mesh_manager.GetMeshInstance(m);
    centers[m]           = Eigen::Vector3d::Zero();
    for (int i = 0; i < instance.num_nodes; ++i) {
      int global_idx = instance.node_offset + i;
      centers[m] += nodes.row(global_idx).transpose();
    }
    centers[m] /= instance.num_nodes;
    std::cout << "Mesh " << m << " center: (" << centers[m].x() << ", "
              << centers[m].y() << ", " << centers[m].z() << ")" << std::endl;
  }

  // Estimate radius from mesh 0
  const auto& instance0 = mesh_manager.GetMeshInstance(mesh0);
  double R              = 0.0;
  for (int i = 0; i < instance0.num_nodes; ++i) {
    int global_idx = instance0.node_offset + i;
    double dist    = (nodes.row(global_idx).transpose() - centers[0]).norm();
    R              = std::max(R, dist);
  }
  std::cout << "Estimated radius R: " << R << std::endl;

  // Compute pressure fields for all three meshes
  for (int m = 0; m < 3; ++m) {
    const auto& instance = mesh_manager.GetMeshInstance(m);
    Eigen::VectorXd pressure_m(instance.num_nodes);
    for (int i = 0; i < instance.num_nodes; ++i) {
      int global_idx = instance.node_offset + i;
      double dist    = (nodes.row(global_idx).transpose() - centers[m]).norm();
      pressure_m(i)  = Eh * std::max(0.0, R - dist);
    }
    mesh_manager.SetScalarField(m, pressure_m);
    std::cout << "Pressure " << m << " range: [" << pressure_m.minCoeff()
              << ", " << pressure_m.maxCoeff() << "]" << std::endl;
  }

  // Get unified scalar field
  const Eigen::VectorXd& pressure = mesh_manager.GetAllScalarFields();
  std::cout << "Computed hydroelastic pressure fields" << std::endl;
  std::cout << "Scalar field size: " << pressure.size() << std::endl;

  // ===== Broadphase =====
  broadphase.Initialize(nodes, elements);
  broadphase.CreateAABB();
  broadphase.BuildNeighborMap();
  broadphase.SortAABBs(0);
  broadphase.DetectCollisions();

  std::cout << "\nBroadphase found " << broadphase.numCollisions
            << " potential collision pairs" << std::endl;

  // Convert broadphase results to pair vector for narrowphase
  std::vector<std::pair<int, int>> collisionPairs;
  for (const auto& cp : broadphase.h_collisionPairs) {
    collisionPairs.emplace_back(cp.idA, cp.idB);
  }

  // ===== Narrowphase =====
  // Build element-to-mesh ID mapping
  Eigen::VectorXi elementMeshIds(mesh_manager.GetTotalElements());
  for (int i = 0; i < mesh_manager.GetNumMeshes(); ++i) {
    const auto& instance = mesh_manager.GetMeshInstance(i);
    for (int e = 0; e < instance.num_elements; ++e) {
      elementMeshIds(instance.element_offset + e) = i;
    }
  }

  narrowphase.Initialize(nodes, elements, pressure, elementMeshIds);
  narrowphase.SetCollisionPairs(collisionPairs);
  narrowphase.ComputeContactPatches();
  narrowphase.RetrieveResults();

  // Print summary
  narrowphase.PrintContactPatches(false);

  // Get valid patches
  auto validPatches = narrowphase.GetValidPatches();
  std::cout << "Retrieved " << validPatches.size() << " valid contact patches"
            << std::endl;

  // Analyze patches by mesh pair
  std::map<std::pair<int, int>, int> patchCountByMeshPair;
  for (const auto& patch : validPatches) {
    int meshA = elementMeshIds(patch.tetA_idx);
    int meshB = elementMeshIds(patch.tetB_idx);
    auto key  = std::make_pair(std::min(meshA, meshB), std::max(meshA, meshB));
    patchCountByMeshPair[key]++;
  }

  std::cout << "\n--- Patches by mesh pair ---" << std::endl;
  for (const auto& kv : patchCountByMeshPair) {
    std::cout << "  Mesh " << kv.first.first << " <-> Mesh " << kv.first.second
              << ": " << kv.second << " patches" << std::endl;
  }

  // Export visualization files
  std::cout << "\n--- Exporting visualization files ---\n" << std::endl;

  ANCFCPUUtils::VisualizationUtils::ExportContactPatchesToVTP(
      validPatches, "output/three_spheres_patches.vtp");
  ANCFCPUUtils::VisualizationUtils::ExportContactPatchesToJSON(
      validPatches, "output/three_spheres_patches.json");
  ANCFCPUUtils::VisualizationUtils::ExportNormalsAsArrows(
      validPatches, 0.02, "output/three_spheres_normals.vtp");
  ANCFCPUUtils::VisualizationUtils::ExportMeshToVTU(
      nodes, elements, pressure, "output/three_spheres_mesh.vtu");

  // Print first few patches
  int printCount = std::min(5, (int)validPatches.size());
  if (printCount > 0) {
    std::cout << "\n--- First " << printCount << " contact patches ---\n"
              << std::endl;
    for (int i = 0; i < printCount; ++i) {
      const auto& patch = validPatches[i];
      int meshA         = elementMeshIds(patch.tetA_idx);
      int meshB         = elementMeshIds(patch.tetB_idx);
      std::cout << "Patch " << i << " (Mesh " << meshA << " Tet "
                << patch.tetA_idx << " <-> Mesh " << meshB << " Tet "
                << patch.tetB_idx << "):" << std::endl;
      std::cout << "  Normal: (" << patch.normal.x << ", " << patch.normal.y
                << ", " << patch.normal.z << ")" << std::endl;
      std::cout << "  Area: " << patch.area << std::endl;
      std::cout << "  Valid orientation: "
                << (patch.validOrientation ? "Yes" : "No") << std::endl;
      std::cout << std::endl;
    }
  }

  // Sanity checks
  EXPECT_GT(validPatches.size(), 0) << "Expected some valid contact patches";

  // Check that normals are unit vectors
  for (const auto& patch : validPatches) {
    double len =
        sqrt(patch.normal.x * patch.normal.x + patch.normal.y * patch.normal.y +
             patch.normal.z * patch.normal.z);
    EXPECT_NEAR(len, 1.0, 1e-6) << "Normal should be unit length";
  }
}