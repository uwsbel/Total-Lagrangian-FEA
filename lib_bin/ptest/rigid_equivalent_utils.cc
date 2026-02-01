#include "rigid_equivalent_utils.h"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <utility>

namespace ANCFPtest {
namespace {

Eigen::Vector3d PseudoInverseSolveSymmetric(const Eigen::Matrix3d& A,
                                            const Eigen::Vector3d& b) {
  // A is expected to be symmetric positive (semi-)definite (inertia tensor).
  // Use an eigen-based pseudo-inverse to handle near-singular cases (e.g. tiny
  // bodies / degenerate point distributions).
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(A);
  if (es.info() != Eigen::Success) {
    // Fallback: damped solve (still robust for most cases).
    const double eps = 1e-12;
    return (A + eps * Eigen::Matrix3d::Identity()).ldlt().solve(b);
  }

  const Eigen::Vector3d evals = es.eigenvalues();
  const Eigen::Matrix3d evecs = es.eigenvectors();

  const double max_eval = std::max({evals.x(), evals.y(), evals.z(), 0.0});
  const double tol = std::max(1e-12, 1e-10 * max_eval);

  Eigen::Vector3d inv;
  for (int i = 0; i < 3; ++i) {
    const double lam = evals(i);
    inv(i) = (lam > tol) ? (1.0 / lam) : 0.0;
  }

  const Eigen::Matrix3d A_pinv = evecs * inv.asDiagonal() * evecs.transpose();
  return A_pinv * b;
}

}  // namespace

RigidEquivalentResult ComputeRigidEquivalentForInstance(
    const Eigen::VectorXd& x, const Eigen::VectorXd& y, const Eigen::VectorXd& z,
    const Eigen::VectorXd& v_xyz, const Eigen::VectorXd& lumped_mass,
    const ANCFCPUUtils::MeshInstance& inst) {
  RigidEquivalentResult out;

  if (inst.num_nodes <= 0) {
    return out;
  }

  const int n_nodes = static_cast<int>(x.size());
  if (y.size() != n_nodes || z.size() != n_nodes || lumped_mass.size() != n_nodes ||
      v_xyz.size() != 3 * n_nodes) {
    std::cerr << "ComputeRigidEquivalentForInstance: size mismatch\n";
    return out;
  }
  if (inst.node_offset < 0 || inst.node_offset + inst.num_nodes > n_nodes) {
    std::cerr << "ComputeRigidEquivalentForInstance: invalid MeshInstance range\n";
    return out;
  }

  // Mass + CoM position and velocity.
  double M = 0.0;
  Eigen::Vector3d sum_mr = Eigen::Vector3d::Zero();
  Eigen::Vector3d sum_mv = Eigen::Vector3d::Zero();
  for (int i = 0; i < inst.num_nodes; ++i) {
    const int idx = inst.node_offset + i;
    const double m = lumped_mass(idx);
    if (!(m > 0.0)) continue;
    const Eigen::Vector3d r(x(idx), y(idx), z(idx));
    const Eigen::Vector3d v(v_xyz(3 * idx + 0), v_xyz(3 * idx + 1),
                            v_xyz(3 * idx + 2));
    M += m;
    sum_mr += m * r;
    sum_mv += m * v;
  }

  out.total_mass = M;
  if (!(M > 0.0)) {
    return out;
  }

  out.com = sum_mr / M;
  out.v_com = sum_mv / M;

  // Angular momentum about CoM and inertia about CoM.
  Eigen::Vector3d L = Eigen::Vector3d::Zero();
  Eigen::Matrix3d I = Eigen::Matrix3d::Zero();
  for (int i = 0; i < inst.num_nodes; ++i) {
    const int idx = inst.node_offset + i;
    const double m = lumped_mass(idx);
    if (!(m > 0.0)) continue;
    const Eigen::Vector3d r(x(idx), y(idx), z(idx));
    const Eigen::Vector3d v(v_xyz(3 * idx + 0), v_xyz(3 * idx + 1),
                            v_xyz(3 * idx + 2));
    const Eigen::Vector3d d = r - out.com;
    const Eigen::Vector3d u = v - out.v_com;
    L += m * d.cross(u);

    const double d2 = d.squaredNorm();
    I += m * (d2 * Eigen::Matrix3d::Identity() - d * d.transpose());
  }

  out.angular_momentum = L;
  out.inertia = I;
  out.omega = PseudoInverseSolveSymmetric(I, L);

  // Residual motion.
  double sum_m_vres2 = 0.0;
  for (int i = 0; i < inst.num_nodes; ++i) {
    const int idx = inst.node_offset + i;
    const double m = lumped_mass(idx);
    if (!(m > 0.0)) continue;
    const Eigen::Vector3d r(x(idx), y(idx), z(idx));
    const Eigen::Vector3d v(v_xyz(3 * idx + 0), v_xyz(3 * idx + 1),
                            v_xyz(3 * idx + 2));
    const Eigen::Vector3d v_hat = out.v_com + out.omega.cross(r - out.com);
    const Eigen::Vector3d v_res = v - v_hat;
    sum_m_vres2 += m * v_res.squaredNorm();
  }
  out.residual_kinetic_energy = 0.5 * sum_m_vres2;
  out.residual_rms_speed = std::sqrt(sum_m_vres2 / M);

  return out;
}

RigidEquivalentCsvLogger::RigidEquivalentCsvLogger(std::string path)
    : path_(std::move(path)) {
  auto file = std::make_unique<std::ofstream>(path_, std::ios::out);
  if (!file->is_open()) {
    std::cerr << "Failed to open rigid-equivalent CSV: " << path_ << "\n";
    return;
  }
  file->setf(std::ios::fixed);
  file->precision(17);
  owned_ = std::move(file);
  out_ = owned_.get();
}

RigidEquivalentCsvLogger::~RigidEquivalentCsvLogger() = default;

bool RigidEquivalentCsvLogger::ok() const {
  return out_ != nullptr;
}

const std::string& RigidEquivalentCsvLogger::path() const {
  return path_;
}

void RigidEquivalentCsvLogger::WriteHeader() {
  if (!ok() || wrote_header_) return;
  (*out_) << "time,step,body,total_mass,"
             "com_x,com_y,com_z,"
             "vcom_x,vcom_y,vcom_z,"
             "omega_x,omega_y,omega_z,"
             "residual_rms,residual_ke\n";
  wrote_header_ = true;
}

void RigidEquivalentCsvLogger::WriteRow(double time, int step,
                                        const std::string& body,
                                        const RigidEquivalentResult& r) {
  if (!ok()) return;
  if (!wrote_header_) WriteHeader();

  (*out_) << time << "," << step << "," << body << "," << r.total_mass << ","
          << r.com.x() << "," << r.com.y() << "," << r.com.z() << ","
          << r.v_com.x() << "," << r.v_com.y() << "," << r.v_com.z() << ","
          << r.omega.x() << "," << r.omega.y() << "," << r.omega.z() << ","
          << r.residual_rms_speed << "," << r.residual_kinetic_energy << "\n";
}

}  // namespace ANCFPtest
