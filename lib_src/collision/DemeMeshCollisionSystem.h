/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    DemeMeshCollisionSystem.h
 * Brief:   Declares a collision backend that wraps DEM-Engine (DEME) mesh-mesh
 *          contact and produces per-node external forces for FE solvers.
 *==============================================================
 *==============================================================*/

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "lib_src/collision/CollisionSystemBase.h"
#include "lib_utils/surface_trimesh.h"

namespace deme {
class DEMSolver;
class DEMMaterial;
class DEMMesh;
}  // namespace deme

struct DemeMeshCollisionBody {
  ANCFCPUUtils::SurfaceTriMesh surface;

  // DEME family code for contact filtering / prescribing.
  unsigned int family = 0;

  // If true, split this body's surface mesh into convex patches (more patch
  // pairs -> more mesh-mesh contact points).
  bool split_into_patches = false;

  // Patch split angle in degrees. If < 0, use the default patch angle
  // (`DEME_PATCH_ANGLE_DEG`, back-compat: `DEME_ITEM_PATCH_ANGLE_DEG`).
  float patch_angle_deg = -1.0f;
};

// Collision system that couples DEM-Engine (DEME) mesh-mesh contact to this
// repo's FE solvers via per-node external forces.
//
// This class is element-agnostic: it operates on a surface triangle mesh with
// global node IDs. Extract surface meshes from specific discretizations (T10,
// ANCF, ...) outside of this class.
//
// Useful knobs (env vars):
// - Patch splitting: `DEME_PATCH_ANGLE_DEG` (smaller -> more patches)
// - Contact material: `DEME_CONTACT_E`, `DEME_CONTACT_NU`, `DEME_CONTACT_COR`
// - FE coupling: `DEME_FORCE_DISTRIB_K`, `DEME_FORCE_SCALE`, `DEME_FORCE_CLAMP`
class DemeMeshCollisionSystem final : public CollisionSystem {
 public:
  DemeMeshCollisionSystem(std::vector<DemeMeshCollisionBody> bodies,
                          double friction, bool enable_self_collision);

  ~DemeMeshCollisionSystem() override;

  void BindNodesDevicePtr(double* d_nodes_xyz, int n_nodes) override;

  void Step(const CollisionSystemInput& in,
            const CollisionSystemParams& params) override;

  const double* GetExternalForcesDevicePtr() const override;
  int GetNumContacts() const override;

 private:
  void BuildSolver(double friction);

  struct RuntimeBody {
    DemeMeshCollisionBody body;
    // Handle returned by DEME during setup; its `owner` ID is populated during
    // `solver_->Initialize()`.
    std::shared_ptr<deme::DEMMesh> mesh_handle;
    unsigned int owner = 0;
    size_t tri_start   = 0;
  };

  std::vector<RuntimeBody> bodies_;
  bool enable_self_collision_ = false;

  // DEM-Engine objects
  std::unique_ptr<deme::DEMSolver> solver_;

  // FE node buffer (device)
  double* d_nodes_xyz_ = nullptr;
  int n_nodes_         = 0;

  // Output buffer (device): length 3*n_nodes_
  double* d_f_contact_ = nullptr;

  // Host scratch
  std::vector<double> h_nodes_xyz_;
  std::vector<double> h_f_contact_;

  int num_contacts_ = 0;
};
