/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    HydroelasticPatchCollisionSystem.h
 * Brief:   Declares a collision system that combines hydroelastic broadphase
 *          and narrowphase to produce per-node external contact forces.
 *==============================================================
 *==============================================================*/

#pragma once

#include <Eigen/Dense>
#include <vector>

#include "lib_src/collision/CollisionSystemBase.h"
#include "lib_utils/mesh_manager.h"

#include "HydroelasticBroadphase.cuh"
#include "HydroelasticNarrowphase.cuh"

class HydroelasticPatchCollisionSystem final : public CollisionSystem {
 public:
  HydroelasticPatchCollisionSystem(const ANCFCPUUtils::MeshManager& mesh_manager,
                                   const Eigen::MatrixXd& initial_nodes,
                                   const Eigen::MatrixXi& elements,
                                   const Eigen::VectorXd& pressure,
                                   const Eigen::VectorXi& elementMeshIds,
                                   bool enable_self_collision);

  void BindNodesDevicePtr(double* d_nodes_xyz, int n_nodes) override;

  void Step(const CollisionSystemInput& in,
            const CollisionSystemParams& params) override;

  const double* GetExternalForcesDevicePtr() const override;
  int GetNumContacts() const override;

  // For visualization/debugging.
  void RetrieveResults();
  std::vector<ContactPatch> GetValidPatches() const;

 private:
  Broadphase broadphase_;
  Narrowphase narrowphase_;
  int num_collision_pairs_ = 0;
};
