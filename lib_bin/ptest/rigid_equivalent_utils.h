#pragma once

#include <Eigen/Dense>

#include <iosfwd>
#include <memory>
#include <string>

#include "../../lib_utils/mesh_manager.h"

namespace ANCFPtest {

struct RigidEquivalentResult {
  double total_mass = 0.0;
  Eigen::Vector3d com = Eigen::Vector3d::Zero();
  Eigen::Vector3d v_com = Eigen::Vector3d::Zero();

  // Rigid-equivalent angular velocity in the *world frame* (right-hand rule).
  // Units: 1/time (e.g. rad/s if positions are meters and velocities are m/s).
  Eigen::Vector3d omega = Eigen::Vector3d::Zero();

  // Angular momentum and inertia about the CoM, both expressed in the world
  // frame. Units: L ~ mass * length^2 / time, I ~ mass * length^2.
  Eigen::Vector3d angular_momentum = Eigen::Vector3d::Zero();
  Eigen::Matrix3d inertia = Eigen::Matrix3d::Zero();

  // Inertia principal values (ascending). Useful to diagnose near-singular
  // cases where omega becomes ill-conditioned.
  Eigen::Vector3d inertia_eigenvalues = Eigen::Vector3d::Zero();
  double inertia_condition_number = 0.0;  // max/min over nonzero eigenvalues

  // How well omega satisfies I*omega ≈ L (0 if exact or if both are ~0).
  double angular_momentum_fit_rel_error = 0.0;

  // Residual motion after subtracting rigid-equivalent twist:
  // v_i_res = v_i - (v_com + omega × (r_i - com)).
  double residual_rms_speed = 0.0;
  double residual_kinetic_energy = 0.0;
};

// Computes a rigid-equivalent twist (v_com, omega) for the given mesh instance,
// using lumped nodal masses and the current nodal positions/velocities.
//
// - Positions are given as separate x/y/z vectors of length n_nodes.
// - Velocities are given as a packed vector [vx0, vy0, vz0, vx1, ...] of length
//   3*n_nodes.
// - `lumped_mass` is length n_nodes.
RigidEquivalentResult ComputeRigidEquivalentForInstance(
    const Eigen::VectorXd& x, const Eigen::VectorXd& y, const Eigen::VectorXd& z,
    const Eigen::VectorXd& v_xyz, const Eigen::VectorXd& lumped_mass,
    const ANCFCPUUtils::MeshInstance& inst);

// Simple CSV logger for rigid-equivalent motion.
//
// Row format:
//   time,step,body,total_mass,com_x,com_y,com_z,vcom_x,vcom_y,vcom_z,omega_x,omega_y,omega_z,inertia_eig0,inertia_eig1,inertia_eig2,inertia_cond,L_fit_rel,residual_rms,residual_ke
class RigidEquivalentCsvLogger {
 public:
  explicit RigidEquivalentCsvLogger(std::string path);
  ~RigidEquivalentCsvLogger();

  bool ok() const;
  const std::string& path() const;

  void WriteHeader();
  void WriteRow(double time, int step, const std::string& body,
                const RigidEquivalentResult& r);

 private:
  std::string path_;
  std::ostream* out_ = nullptr;
  std::unique_ptr<std::ostream> owned_;
  bool wrote_header_ = false;
};

}  // namespace ANCFPtest
