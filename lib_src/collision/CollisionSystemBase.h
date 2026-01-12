/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    CollisionSystemBase.h
 * Brief:   Defines the collision backend interface used by FE solvers,
 *          including input/parameter structs and the external-force API.
 *==============================================================
 *==============================================================*/

#pragma once

// A collision backend produces per-node external forces given the current FE
// nodal state. Multiple backends can coexist (hydroelastic, DEME, ...).

struct CollisionSystemInput {
  // Column-major node buffer on device:
  //   [x(0..n-1), y(0..n-1), z(0..n-1)]
  double* d_nodes_xyz = nullptr;
  int n_nodes = 0;

  // Optional velocity buffer (3*n) on device:
  //   [vx0, vy0, vz0, vx1, ...]
  double* d_vel_xyz = nullptr;

  // The integrator time step (seconds).
  double dt = 0.0;
};

struct CollisionSystemParams {
  double damping = 0.0;
  double friction = 0.0;
};

class CollisionSystem {
 public:
  virtual ~CollisionSystem() = default;

  virtual void BindNodesDevicePtr(double* d_nodes_xyz, int n_nodes) = 0;

  virtual void Step(const CollisionSystemInput& in,
                    const CollisionSystemParams& params) = 0;

  virtual const double* GetExternalForcesDevicePtr() const = 0;
  virtual int GetNumContacts() const = 0;
};
