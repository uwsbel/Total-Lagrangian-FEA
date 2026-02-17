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

#include <cstdint>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <cuda_runtime.h>

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

  // If true, skip processing contact forces for this body (body still
  // participates in collision detection for other bodies to collide with it).
  bool skip_self_contact_forces = false;

  // Mass of the body in kg. Used by DEME for contact/friction calculations.
  // Default 1.0 for backward compatibility; should be set to actual mass.
  float mass = 1.0f;
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
  // Backward-compatible constructor: uses restitution (CoR) = 0.5 by default
  // (overridable via `DEME_CONTACT_COR`).
  DemeMeshCollisionSystem(std::vector<DemeMeshCollisionBody> bodies,
                          double friction, double stiffness,
                          bool enable_self_collision, double time_step)
      : DemeMeshCollisionSystem(std::move(bodies), friction, friction, stiffness,
                                0.5, enable_self_collision, time_step) {}

  DemeMeshCollisionSystem(std::vector<DemeMeshCollisionBody> bodies,
                          double friction, double stiffness,
                          double restitution,
                          bool enable_self_collision, double time_step)
      : DemeMeshCollisionSystem(std::move(bodies), friction, friction, stiffness,
                                restitution, enable_self_collision, time_step) {}

  DemeMeshCollisionSystem(std::vector<DemeMeshCollisionBody> bodies,
                          double mu_s, double mu_k, double stiffness,
                          double restitution,
                          bool enable_self_collision, double time_step);

  ~DemeMeshCollisionSystem() override;

  void BindNodesDevicePtr(double* d_nodes_xyz, int n_nodes) override;

  void Step(const CollisionSystemInput& in,
            const CollisionSystemParams& params) override;

  const double* GetExternalForcesDevicePtr() const override;
  int GetNumContacts() const override;

 private:
  void BuildSolver(double mu_s, double mu_k, double stiffness, double restitution);

 struct RuntimeBody {
    DemeMeshCollisionBody body;
    // Handle returned by DEME during setup; its `owner` ID is populated during
    // `solver_->Initialize()`.
    std::shared_ptr<deme::DEMMesh> mesh_handle;
    unsigned int owner = 0;
    size_t tri_start   = 0;

    // Reference vertex positions in the owner's local frame at t=0, used to
    // recover a best-fit rigid pose (pos+orientation) from current nodal
    // positions. Keeping pose separate from deformation is important for DEME's
    // patch-level collision and friction computations.
    std::vector<Eigen::Vector3d> ref_vertices_local;

    // Previous pose for velocity/omega estimation.
    Eigen::Vector3d prev_pos = Eigen::Vector3d::Zero();
    Eigen::Quaterniond prev_quat = Eigen::Quaterniond::Identity();

    int surf_vert_start = 0;
    int surf_vert_count = 0;
    int tri_count       = 0;
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

  // Triangle velocity coupling (device): per DEME triangle primitive.
  int n_tris_               = 0;
  int3* d_tri_global_nodes_ = nullptr;   // length n_tris_
  float3* d_tri_vel_center_ = nullptr;   // length n_tris_

  int n_surf_verts_            = 0;
  int* d_surf_global_node_id_  = nullptr;
  int* d_surf_body_id_         = nullptr;
  double3* d_surf_ref_local_   = nullptr;
  float3* d_surf_pos_global_   = nullptr;
  float3* d_surf_pos_local_    = nullptr;

  int n_bodies_                = 0;
  uint32_t* d_body_owner_id_   = nullptr;
  int* d_owner_to_body_        = nullptr;
  int owner_to_body_size_      = 0;
  int* d_body_vert_start_      = nullptr;
  int* d_body_vert_count_      = nullptr;
  double* d_body_inv_vert_count_ = nullptr;
  uint8_t* d_body_skip_forces_ = nullptr;
  float3* d_body_pos_f_        = nullptr;
  float4* d_body_quat_f_       = nullptr;
  float3* d_body_lin_vel_f_    = nullptr;
  float3* d_body_omega_f_      = nullptr;

  double3* d_body_com_         = nullptr;
  double* d_body_H_            = nullptr;

  int3* d_tri_surf_vert_ids_   = nullptr;
  float3* d_tri_relpos_n1_     = nullptr;
  float3* d_tri_relpos_n2_     = nullptr;
  float3* d_tri_relpos_n3_     = nullptr;

  size_t contact_capacity_     = 0;
  float3* d_contact_points_    = nullptr;
  float3* d_contact_forces_    = nullptr;
  uint32_t* d_contact_owner_ = nullptr;

  // Host scratch
  std::vector<double> h_nodes_xyz_;
  std::vector<double> h_f_contact_;
  std::vector<float3> h_tri_vel_center_;

  std::vector<double3> h_body_com_;
  std::vector<double> h_body_H_;

  std::vector<float3> h_body_pos_f_;
  std::vector<float4> h_body_quat_f_;
  std::vector<float3> h_body_lin_vel_f_;
  std::vector<float3> h_body_omega_f_;

  std::vector<float3> h_owner_pos_;
  std::vector<float4> h_owner_quat_;
  std::vector<float3> h_owner_lin_vel_;
  std::vector<float3> h_owner_omega_;
  bool owners_dense_ = false;

  int num_contacts_ = 0;

  // Pose tracking for friction computation: DEME needs (lin/ang) velocity to
  // compute friction directions. We estimate owner velocities from pose changes.
  bool first_step_ = true;

  double dt_ = 0.0;
 };
