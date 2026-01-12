/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    HydroelasticPatchCollisionSystem.cc
 * Brief:   Implements the HydroelasticPatchCollisionSystem wrapper that
 *          orchestrates broadphase/narrowphase contact and writes per-node
 *          external forces for the FE solver.
 *==============================================================
 *==============================================================*/

#include "HydroelasticPatchCollisionSystem.h"

#include <stdexcept>

HydroelasticPatchCollisionSystem::HydroelasticPatchCollisionSystem(
    const ANCFCPUUtils::MeshManager& mesh_manager,
    const Eigen::MatrixXd& initial_nodes,
    const Eigen::MatrixXi& elements,
    const Eigen::VectorXd& pressure,
    const Eigen::VectorXi& elementMeshIds,
    bool enable_self_collision) {
  broadphase_.Initialize(mesh_manager);
  broadphase_.EnableSelfCollision(enable_self_collision);
  broadphase_.BuildNeighborMap();

  narrowphase_.Initialize(initial_nodes, elements, pressure, elementMeshIds);
  narrowphase_.EnableSelfCollision(enable_self_collision);
}

void HydroelasticPatchCollisionSystem::BindNodesDevicePtr(double* d_nodes_xyz,
                                                         int n_nodes) {
  if (d_nodes_xyz == nullptr) {
    throw std::invalid_argument("BindNodesDevicePtr: d_nodes_xyz is null");
  }
  broadphase_.BindNodesDevicePtr(d_nodes_xyz);
  narrowphase_.BindNodesDevicePtr(d_nodes_xyz);
  (void)n_nodes;

  broadphase_.CreateAABB();
  broadphase_.SortAABBs(0);
}

void HydroelasticPatchCollisionSystem::Step(const CollisionSystemInput& in,
                                           const CollisionSystemParams& params) {
  if (in.d_vel_xyz == nullptr) {
    throw std::invalid_argument("HydroelasticPatchCollisionSystem::Step: d_vel_xyz is null");
  }

  broadphase_.CreateAABB();
  broadphase_.SortAABBs(0);
  broadphase_.DetectCollisions(false);
  num_collision_pairs_ = broadphase_.numCollisions;

  narrowphase_.SetCollisionPairsDevice(broadphase_.GetCollisionPairsDevicePtr(),
                                       num_collision_pairs_);
  narrowphase_.ComputeContactPatches();

  // This writes the per-node external force buffer on the device.
  narrowphase_.ComputeExternalForcesGPUDevice(in.d_vel_xyz, params.damping,
                                              params.friction);
}

const double* HydroelasticPatchCollisionSystem::GetExternalForcesDevicePtr()
    const {
  return narrowphase_.GetExternalForcesDevicePtr();
}

int HydroelasticPatchCollisionSystem::GetNumContacts() const {
  return num_collision_pairs_;
}

void HydroelasticPatchCollisionSystem::RetrieveResults() {
  narrowphase_.RetrieveResults();
}

std::vector<ContactPatch> HydroelasticPatchCollisionSystem::GetValidPatches() const {
  return narrowphase_.GetValidPatches();
}
