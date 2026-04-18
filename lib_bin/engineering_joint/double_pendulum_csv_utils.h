#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "../../lib_src/elements/FEAT10Data.cuh"
#include "../../lib_src/materials/SolidMaterialProperties.h"
#include "../../lib_src/solvers/FEAT10ConstraintManager.h"
#include "../../lib_utils/quadrature_utils.h"

namespace engineering_joint {

struct HingeWrench {
  Eigen::Vector3d force  = Eigen::Vector3d::Zero();
  Eigen::Vector3d moment = Eigen::Vector3d::Zero();
};

inline HingeWrench ScaleHingeWrench(const HingeWrench& wrench, double scale) {
  HingeWrench scaled = wrench;
  scaled.force *= scale;
  scaled.moment *= scale;
  return scaled;
}

inline double ComputeInfinityNorm(const Eigen::VectorXd& values) {
  if (values.size() == 0) {
    return 0.0;
  }
  return values.cwiseAbs().maxCoeff();
}

inline double ComputeIndexedL2Norm(const Eigen::VectorXd& values,
                                   const std::vector<int>& indices) {
  double sum_sq = 0.0;
  for (int index : indices) {
    if (index < 0 || index >= values.size()) {
      continue;
    }
    const double value = values(index);
    sum_sq += value * value;
  }
  return std::sqrt(sum_sq);
}

inline double ComputeIndexedInfinityNorm(const Eigen::VectorXd& values,
                                         const std::vector<int>& indices) {
  double max_abs = 0.0;
  for (int index : indices) {
    if (index < 0 || index >= values.size()) {
      continue;
    }
    max_abs = std::max(max_abs, std::abs(values(index)));
  }
  return max_abs;
}

inline double ComputeSegmentL2Norm(const Eigen::VectorXd& values, int offset,
                                   int count) {
  if (offset < 0 || count <= 0 || offset + count > values.size()) {
    return 0.0;
  }
  return values.segment(offset, count).norm();
}

inline Eigen::Vector3d AveragePoint(const Eigen::Vector3d& a,
                                    const Eigen::Vector3d& b) {
  return 0.5 * (a + b);
}

inline Eigen::Vector3d EvaluateCurrentPointPosition(
    const FEAT10ConstraintManager::ReferencePoint& point,
    const Eigen::MatrixXi& connectivity, const Eigen::VectorXd& x,
    const Eigen::VectorXd& y, const Eigen::VectorXd& z) {
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  for (int local_node = 0; local_node < Quadrature::N_NODE_T10_10;
       ++local_node) {
    const double weight = point.shape(local_node);
    if (weight == 0.0) {
      continue;
    }
    const int node = connectivity(point.element_idx, local_node);
    position += weight * Eigen::Vector3d(x(node), y(node), z(node));
  }
  return position;
}

inline Eigen::Vector3d SafeNormalized(const Eigen::Vector3d& value) {
  const double norm = value.norm();
  if (norm < 1e-12) {
    return Eigen::Vector3d::Zero();
  }
  return value / norm;
}

inline double ClampUnit(double value) {
  return std::max(-1.0, std::min(1.0, value));
}

inline double ComputeAngleBetween(const Eigen::Vector3d& a,
                                  const Eigen::Vector3d& b) {
  const double a_norm = a.norm();
  const double b_norm = b.norm();
  if (a_norm < 1e-12 || b_norm < 1e-12) {
    return 0.0;
  }
  return std::acos(ClampUnit(a.dot(b) / (a_norm * b_norm)));
}

inline double ComputePointLineDistance(const Eigen::Vector3d& point,
                                       const Eigen::Vector3d& line_point,
                                       const Eigen::Vector3d& line_direction) {
  const Eigen::Vector3d line_dir_unit = SafeNormalized(line_direction);
  if (line_dir_unit.squaredNorm() < 1e-24) {
    return (point - line_point).norm();
  }
  const Eigen::Vector3d delta = point - line_point;
  const Eigen::Vector3d radial =
      delta - delta.dot(line_dir_unit) * line_dir_unit;
  return radial.norm();
}

inline double ComputeSwingAngleFromNegativeZ(const Eigen::Vector3d& direction) {
  return ComputeAngleBetween(direction, -Eigen::Vector3d::UnitZ());
}

inline double ComputeAzimuth(const Eigen::Vector3d& direction) {
  return std::atan2(direction.y(), direction.x());
}

inline double SignedAngleAboutAxis(const Eigen::Vector3d& from,
                                   const Eigen::Vector3d& to,
                                   const Eigen::Vector3d& axis) {
  const Eigen::Vector3d axis_unit = axis.normalized();
  const Eigen::Vector3d from_proj = from - from.dot(axis_unit) * axis_unit;
  const Eigen::Vector3d to_proj   = to - to.dot(axis_unit) * axis_unit;
  const double from_norm          = from_proj.norm();
  const double to_norm            = to_proj.norm();
  if (from_norm < 1e-12 || to_norm < 1e-12) {
    return 0.0;
  }

  const Eigen::Vector3d from_unit = from_proj / from_norm;
  const Eigen::Vector3d to_unit   = to_proj / to_norm;
  const double sin_theta          = axis_unit.dot(from_unit.cross(to_unit));
  const double cos_theta          = from_unit.dot(to_unit);
  return std::atan2(sin_theta, cos_theta);
}

inline double ComputePotentialEnergy(const std::vector<double>& lumped_mass,
                                     const Eigen::VectorXd& z, double gravity) {
  double potential_energy = 0.0;
  const int count =
      std::min<int>(static_cast<int>(lumped_mass.size()), z.size());
  for (int node = 0; node < count; ++node) {
    potential_energy +=
        -lumped_mass[static_cast<size_t>(node)] * gravity * z(node);
  }
  return potential_energy;
}

inline double ComputeKineticEnergy(const std::vector<double>& lumped_mass,
                                   const Eigen::VectorXd& velocity_xyz) {
  double kinetic_energy = 0.0;
  const int n_nodes     = std::min<int>(static_cast<int>(lumped_mass.size()),
                                        velocity_xyz.size() / 3);
  for (int node = 0; node < n_nodes; ++node) {
    const double vx = velocity_xyz(3 * node + 0);
    const double vy = velocity_xyz(3 * node + 1);
    const double vz = velocity_xyz(3 * node + 2);
    kinetic_energy += 0.5 * lumped_mass[static_cast<size_t>(node)] *
                      (vx * vx + vy * vy + vz * vz);
  }
  return kinetic_energy;
}

inline double ComputeSVKElasticStrainEnergyDensity(
    const Eigen::Matrix3d& deformation_gradient,
    const SolidMaterialProperties& material) {
  const Eigen::Matrix3d green_lagrange_strain =
      0.5 * (deformation_gradient.transpose() * deformation_gradient -
             Eigen::Matrix3d::Identity());
  const double trace_strain = green_lagrange_strain.trace();
  return material.mu() * green_lagrange_strain.squaredNorm() +
         0.5 * material.lambda() * trace_strain * trace_strain;
}

inline double ComputeMooneyRivlinElasticStrainEnergyDensity(
    const Eigen::Matrix3d& deformation_gradient,
    const SolidMaterialProperties& material) {
  const Eigen::Matrix3d cauchy_green =
      deformation_gradient.transpose() * deformation_gradient;
  const double first_invariant  = cauchy_green.trace();
  const double second_invariant = 0.5 * (first_invariant * first_invariant -
                                         (cauchy_green * cauchy_green).trace());

  double jacobian = deformation_gradient.determinant();
  if (std::abs(jacobian) < 1e-12) {
    jacobian = jacobian >= 0.0 ? 1e-12 : -1e-12;
  }
  const double j_to_one_third        = std::cbrt(jacobian);
  const double j_to_minus_two_thirds = 1.0 / (j_to_one_third * j_to_one_third);
  const double j_to_minus_four_thirds =
      j_to_minus_two_thirds * j_to_minus_two_thirds;

  const double first_isochoric  = j_to_minus_two_thirds * first_invariant;
  const double second_isochoric = j_to_minus_four_thirds * second_invariant;
  return material.mu10 * (first_isochoric - 3.0) +
         material.mu01 * (second_isochoric - 3.0) +
         0.5 * material.kappa * (jacobian - 1.0) * (jacobian - 1.0);
}

inline double ComputeElasticStrainEnergyDensity(
    const Eigen::Matrix3d& deformation_gradient,
    const SolidMaterialProperties& material) {
  if (material.material_model == MATERIAL_MODEL_MOONEY_RIVLIN) {
    return ComputeMooneyRivlinElasticStrainEnergyDensity(deformation_gradient,
                                                         material);
  }
  return ComputeSVKElasticStrainEnergyDensity(deformation_gradient, material);
}

inline double ComputeElasticStrainEnergy(
    const std::vector<std::vector<Eigen::MatrixXd>>& deformation_gradient,
    const std::vector<std::vector<double>>& det_j_ref,
    const SolidMaterialProperties& material) {
  double elastic_strain_energy = 0.0;
  const int n_elem =
      std::min<int>(deformation_gradient.size(), det_j_ref.size());
  for (int elem_idx = 0; elem_idx < n_elem; ++elem_idx) {
    const int n_qp = std::min<int>(
        deformation_gradient[static_cast<size_t>(elem_idx)].size(),
        det_j_ref[static_cast<size_t>(elem_idx)].size());
    for (int qp_idx = 0; qp_idx < n_qp; ++qp_idx) {
      const Eigen::Matrix3d f =
          deformation_gradient[static_cast<size_t>(elem_idx)]
                              [static_cast<size_t>(qp_idx)];
      const double dV = det_j_ref[static_cast<size_t>(elem_idx)]
                                 [static_cast<size_t>(qp_idx)] *
                        Quadrature::tet5pt_weights(qp_idx);
      elastic_strain_energy +=
          ComputeElasticStrainEnergyDensity(f, material) * dV;
    }
  }
  return elastic_strain_energy;
}

inline double ComputeElasticStrainEnergy(
    GPU_FEAT10_Data& data, const SolidMaterialProperties& material) {
  std::vector<std::vector<Eigen::MatrixXd>> deformation_gradient;
  std::vector<std::vector<double>> det_j_ref;
  data.CalcP();
  data.RetrieveDeformationGradientToCPU(deformation_gradient);
  data.RetrieveDetJToCPU(det_j_ref);
  return ComputeElasticStrainEnergy(deformation_gradient, det_j_ref, material);
}

inline Eigen::VectorXd ComputeGeneralizedReactionFromCSR(
    int n_dofs, const std::vector<int>& offsets,
    const std::vector<int>& columns, const std::vector<double>& values,
    const Eigen::VectorXd& lambda, int row_begin, int row_end) {
  Eigen::VectorXd generalized_reaction = Eigen::VectorXd::Zero(n_dofs);
  if (row_begin < 0 || row_end < row_begin ||
      static_cast<size_t>(row_end + 1) > offsets.size()) {
    return generalized_reaction;
  }

  for (int row = row_begin; row < row_end; ++row) {
    if (row < 0 || row >= lambda.size()) {
      continue;
    }
    const double lambda_row = lambda(row);
    const int start         = offsets[static_cast<size_t>(row)];
    const int end           = offsets[static_cast<size_t>(row + 1)];
    for (int idx = start; idx < end; ++idx) {
      const int dof = columns[static_cast<size_t>(idx)];
      if (dof < 0 || dof >= n_dofs) {
        continue;
      }
      generalized_reaction(dof) +=
          values[static_cast<size_t>(idx)] * lambda_row;
    }
  }
  return generalized_reaction;
}

inline HingeWrench EstimateHingeWrench(
    const Eigen::VectorXd& generalized_reaction, int node_offset, int num_nodes,
    const Eigen::VectorXd& x, const Eigen::VectorXd& y,
    const Eigen::VectorXd& z, const Eigen::Vector3d& hinge_point) {
  HingeWrench wrench;
  const int node_end = node_offset + num_nodes;
  for (int node = node_offset; node < node_end; ++node) {
    if (3 * node + 2 >= generalized_reaction.size() || node >= x.size() ||
        node >= y.size() || node >= z.size()) {
      continue;
    }

    const Eigen::Vector3d nodal_force(generalized_reaction(3 * node + 0),
                                      generalized_reaction(3 * node + 1),
                                      generalized_reaction(3 * node + 2));
    const Eigen::Vector3d position(x(node), y(node), z(node));
    wrench.force += nodal_force;
    wrench.moment += (position - hinge_point).cross(nodal_force);
  }
  return wrench;
}

inline HingeWrench RecoverSeparatedRevoluteJointWrenchFromCSR(
    int n_dofs, const std::vector<int>& offsets,
    const std::vector<int>& columns, const std::vector<double>& values,
    const Eigen::VectorXd& augmented_dual, int row_begin,
    int position_row_count, int orientation_row_count, int node_offset,
    int num_nodes, const Eigen::VectorXd& x, const Eigen::VectorXd& y,
    const Eigen::VectorXd& z, const Eigen::Vector3d& hinge_point) {
  HingeWrench wrench;

  const Eigen::VectorXd position_reaction = ComputeGeneralizedReactionFromCSR(
      n_dofs, offsets, columns, values, augmented_dual, row_begin,
      row_begin + position_row_count);
  const Eigen::VectorXd orientation_reaction =
      ComputeGeneralizedReactionFromCSR(
          n_dofs, offsets, columns, values, augmented_dual,
          row_begin + position_row_count,
          row_begin + position_row_count + orientation_row_count);

  const HingeWrench position_wrench = EstimateHingeWrench(
      position_reaction, node_offset, num_nodes, x, y, z, hinge_point);
  const HingeWrench orientation_wrench = EstimateHingeWrench(
      orientation_reaction, node_offset, num_nodes, x, y, z, hinge_point);

  wrench.force  = position_wrench.force;
  wrench.moment = orientation_wrench.moment;
  return wrench;
}

class DoublePendulumCsvWriter {
 public:
  DoublePendulumCsvWriter() = default;

  explicit DoublePendulumCsvWriter(const std::string& path) {
    Open(path);
  }

  void Open(const std::string& path) {
    if (stream_.is_open()) {
      stream_.close();
    }
    stream_.clear();
    std::filesystem::remove(path);
    stream_.open(path, std::ios::out | std::ios::trunc);
    stream_ << std::setprecision(16);
    stream_ << "step,time"
            << ",upper_joint_angle_rad"
            << ",lower_joint_absolute_angle_rad"
            << ",lower_joint_relative_angle_rad"
            << ",constraint_violation_l2"
            << ",constraint_violation_linf"
            << ",position_constraint_violation_l2"
            << ",position_constraint_violation_linf"
            << ",orientation_constraint_violation_l2"
            << ",orientation_constraint_violation_linf"
            << ",total_energy"
            << ",kinetic_energy"
            << ",potential_energy"
            << ",elastic_strain_energy"
            << ",upper_joint_reaction_force_l2"
            << ",upper_joint_reaction_torque_l2"
            << ",upper_joint_reaction_force_x"
            << ",upper_joint_reaction_force_y"
            << ",upper_joint_reaction_force_z"
            << ",upper_joint_reaction_torque_x"
            << ",upper_joint_reaction_torque_y"
            << ",upper_joint_reaction_torque_z"
            << ",lower_joint_reaction_force_l2"
            << ",lower_joint_reaction_torque_l2"
            << ",lower_joint_reaction_force_x"
            << ",lower_joint_reaction_force_y"
            << ",lower_joint_reaction_force_z"
            << ",lower_joint_reaction_torque_x"
            << ",lower_joint_reaction_torque_y"
            << ",lower_joint_reaction_torque_z"
            << ",lower_hinge_mismatch_norm"
            << ",tip_x"
            << ",tip_y"
            << ",tip_z\n";
  }

  bool is_open() const {
    return stream_.is_open();
  }

  void WriteRow(
      int step, double time, double upper_joint_angle_rad,
      double lower_joint_absolute_angle_rad,
      double lower_joint_relative_angle_rad, double constraint_violation_l2,
      double constraint_violation_linf, double position_constraint_violation_l2,
      double position_constraint_violation_linf,
      double orientation_constraint_violation_l2,
      double orientation_constraint_violation_linf, double total_energy,
      double kinetic_energy, double potential_energy,
      double elastic_strain_energy, const HingeWrench& upper_joint_reaction,
      const HingeWrench& lower_joint_reaction, double lower_hinge_mismatch_norm,
      const Eigen::Vector3d& tip_position) {
    const double upper_force_l2  = upper_joint_reaction.force.norm();
    const double upper_torque_l2 = upper_joint_reaction.moment.norm();
    const double lower_force_l2  = lower_joint_reaction.force.norm();
    const double lower_torque_l2 = lower_joint_reaction.moment.norm();
    std::ostringstream row;
    row << std::setprecision(16) << step << ',' << time << ','
        << upper_joint_angle_rad << ',' << lower_joint_absolute_angle_rad << ','
        << lower_joint_relative_angle_rad << ',' << constraint_violation_l2
        << ',' << constraint_violation_linf << ','
        << position_constraint_violation_l2 << ','
        << position_constraint_violation_linf << ','
        << orientation_constraint_violation_l2 << ','
        << orientation_constraint_violation_linf << ',' << total_energy << ','
        << kinetic_energy << ',' << potential_energy << ','
        << elastic_strain_energy << ',' << upper_force_l2 << ','
        << upper_torque_l2 << ',' << upper_joint_reaction.force.x() << ','
        << upper_joint_reaction.force.y() << ','
        << upper_joint_reaction.force.z() << ','
        << upper_joint_reaction.moment.x() << ','
        << upper_joint_reaction.moment.y() << ','
        << upper_joint_reaction.moment.z() << ',' << lower_force_l2 << ','
        << lower_torque_l2 << ',' << lower_joint_reaction.force.x() << ','
        << lower_joint_reaction.force.y() << ','
        << lower_joint_reaction.force.z() << ','
        << lower_joint_reaction.moment.x() << ','
        << lower_joint_reaction.moment.y() << ','
        << lower_joint_reaction.moment.z() << ',' << lower_hinge_mismatch_norm
        << ',' << tip_position.x() << ',' << tip_position.y() << ','
        << tip_position.z() << '\n';
    stream_ << row.str();
    stream_.flush();
  }

 private:
  std::ofstream stream_;
};

class DoublePendulumSphericalCsvWriter {
 public:
  DoublePendulumSphericalCsvWriter() = default;

  explicit DoublePendulumSphericalCsvWriter(const std::string& path) {
    Open(path);
  }

  void Open(const std::string& path) {
    if (stream_.is_open()) {
      stream_.close();
    }
    stream_.clear();
    std::filesystem::remove(path);
    stream_.open(path, std::ios::out | std::ios::trunc);
    stream_ << std::setprecision(16);
    stream_ << "step,time"
            << ",upper_swing_angle_rad"
            << ",upper_azimuth_rad"
            << ",lower_swing_angle_rad"
            << ",lower_azimuth_rad"
            << ",inter_link_angle_rad"
            << ",constraint_violation_l2"
            << ",constraint_violation_linf"
            << ",upper_joint_position_residual_l2"
            << ",upper_joint_position_residual_linf"
            << ",lower_joint_position_residual_l2"
            << ",lower_joint_position_residual_linf"
            << ",total_energy"
            << ",kinetic_energy"
            << ",potential_energy"
            << ",elastic_strain_energy"
            << ",upper_joint_reaction_force_l2"
            << ",upper_joint_resultant_moment_about_hinge_l2"
            << ",upper_joint_reaction_force_x"
            << ",upper_joint_reaction_force_y"
            << ",upper_joint_reaction_force_z"
            << ",upper_joint_resultant_moment_about_hinge_x"
            << ",upper_joint_resultant_moment_about_hinge_y"
            << ",upper_joint_resultant_moment_about_hinge_z"
            << ",lower_joint_reaction_force_l2"
            << ",lower_joint_resultant_moment_about_hinge_l2"
            << ",lower_joint_reaction_force_x"
            << ",lower_joint_reaction_force_y"
            << ",lower_joint_reaction_force_z"
            << ",lower_joint_resultant_moment_about_hinge_x"
            << ",lower_joint_resultant_moment_about_hinge_y"
            << ",lower_joint_resultant_moment_about_hinge_z"
            << ",lower_hinge_mismatch_norm"
            << ",upper_dir_x"
            << ",upper_dir_y"
            << ",upper_dir_z"
            << ",lower_dir_x"
            << ",lower_dir_y"
            << ",lower_dir_z"
            << ",tip_x"
            << ",tip_y"
            << ",tip_z\n";
  }

  bool is_open() const {
    return stream_.is_open();
  }

  void WriteRow(int step, double time, double upper_swing_angle_rad,
                double upper_azimuth_rad, double lower_swing_angle_rad,
                double lower_azimuth_rad, double inter_link_angle_rad,
                double constraint_violation_l2,
                double constraint_violation_linf,
                double upper_joint_position_residual_l2,
                double upper_joint_position_residual_linf,
                double lower_joint_position_residual_l2,
                double lower_joint_position_residual_linf, double total_energy,
                double kinetic_energy, double potential_energy,
                double elastic_strain_energy,
                const HingeWrench& upper_joint_reaction,
                const HingeWrench& lower_joint_reaction,
                double lower_hinge_mismatch_norm,
                const Eigen::Vector3d& upper_direction,
                const Eigen::Vector3d& lower_direction,
                const Eigen::Vector3d& tip_position) {
    const double upper_force_l2          = upper_joint_reaction.force.norm();
    const double upper_torque_l2         = upper_joint_reaction.moment.norm();
    const double lower_force_l2          = lower_joint_reaction.force.norm();
    const double lower_torque_l2         = lower_joint_reaction.moment.norm();
    const Eigen::Vector3d upper_dir_unit = SafeNormalized(upper_direction);
    const Eigen::Vector3d lower_dir_unit = SafeNormalized(lower_direction);

    std::ostringstream row;
    row << std::setprecision(16) << step << ',' << time << ','
        << upper_swing_angle_rad << ',' << upper_azimuth_rad << ','
        << lower_swing_angle_rad << ',' << lower_azimuth_rad << ','
        << inter_link_angle_rad << ',' << constraint_violation_l2 << ','
        << constraint_violation_linf << ',' << upper_joint_position_residual_l2
        << ',' << upper_joint_position_residual_linf << ','
        << lower_joint_position_residual_l2 << ','
        << lower_joint_position_residual_linf << ',' << total_energy << ','
        << kinetic_energy << ',' << potential_energy << ','
        << elastic_strain_energy << ',' << upper_force_l2 << ','
        << upper_torque_l2 << ',' << upper_joint_reaction.force.x() << ','
        << upper_joint_reaction.force.y() << ','
        << upper_joint_reaction.force.z() << ','
        << upper_joint_reaction.moment.x() << ','
        << upper_joint_reaction.moment.y() << ','
        << upper_joint_reaction.moment.z() << ',' << lower_force_l2 << ','
        << lower_torque_l2 << ',' << lower_joint_reaction.force.x() << ','
        << lower_joint_reaction.force.y() << ','
        << lower_joint_reaction.force.z() << ','
        << lower_joint_reaction.moment.x() << ','
        << lower_joint_reaction.moment.y() << ','
        << lower_joint_reaction.moment.z() << ',' << lower_hinge_mismatch_norm
        << ',' << upper_dir_unit.x() << ',' << upper_dir_unit.y() << ','
        << upper_dir_unit.z() << ',' << lower_dir_unit.x() << ','
        << lower_dir_unit.y() << ',' << lower_dir_unit.z() << ','
        << tip_position.x() << ',' << tip_position.y() << ','
        << tip_position.z() << '\n';
    stream_ << row.str();
    stream_.flush();
  }

 private:
  std::ofstream stream_;
};

class PistonPumpCsvWriter {
 public:
  PistonPumpCsvWriter() = default;

  explicit PistonPumpCsvWriter(const std::string& path) {
    Open(path);
  }

  void Open(const std::string& path) {
    if (stream_.is_open()) {
      stream_.close();
    }
    stream_.clear();
    std::filesystem::remove(path);
    stream_.open(path, std::ios::out | std::ios::trunc);
    stream_ << std::setprecision(16);
    stream_ << "step,time"
            << ",cylindrical_constraint_violation_l2"
            << ",cylindrical_constraint_violation_linf"
            << ",parallel_constraint_violation_l2"
            << ",parallel_constraint_violation_linf"
            << ",collocation_constraint_violation_l2"
            << ",collocation_constraint_violation_linf"
            << ",joint_reaction_force_l2"
            << ",joint_reaction_force_x"
            << ",joint_reaction_force_y"
            << ",joint_reaction_force_z"
            << ",joint_reaction_moment_x"
            << ",joint_reaction_moment_y"
            << ",joint_reaction_moment_z"
            << ",piston_axis_x"
            << ",piston_axis_y"
            << ",piston_axis_z"
            << ",piston_axial_disp"
            << ",piston_axis_relative_radial_drift"
            << ",lower_axis_radial_drift"
            << ",upper_axis_radial_drift"
            << ",world_axis_radial_drift"
            << ",applied_bend_force_y"
            << ",cylindrical_row_reconstruction_l2"
            << ",cylindrical_row_reconstruction_linf"
            << ",lower_parallel_residual_0"
            << ",lower_parallel_residual_1"
            << ",lower_collocation_residual_0"
            << ",lower_collocation_residual_1"
            << ",upper_parallel_residual_0"
            << ",upper_parallel_residual_1"
            << ",upper_collocation_residual_0"
            << ",upper_collocation_residual_1"
            << ",handle_tip_x"
            << ",handle_tip_y"
            << ",handle_tip_z\n";
  }

  bool is_open() const {
    return stream_.is_open();
  }

  void WriteRow(
      int step, double time, double cylindrical_constraint_violation_l2,
      double cylindrical_constraint_violation_linf,
      double parallel_constraint_violation_l2,
      double parallel_constraint_violation_linf,
      double collocation_constraint_violation_l2,
      double collocation_constraint_violation_linf,
      const HingeWrench& joint_reaction,
      const Eigen::Vector3d& piston_axis_position, double piston_axial_disp,
      double piston_axis_radial_drift, double lower_axis_radial_drift,
      double upper_axis_radial_drift, double world_axis_radial_drift,
      double applied_bend_force_y, double cylindrical_row_reconstruction_l2,
      double cylindrical_row_reconstruction_linf,
      double lower_parallel_residual_0, double lower_parallel_residual_1,
      double lower_collocation_residual_0, double lower_collocation_residual_1,
      double upper_parallel_residual_0, double upper_parallel_residual_1,
      double upper_collocation_residual_0, double upper_collocation_residual_1,
      const Eigen::Vector3d& handle_tip_position) {
    std::ostringstream row;
    row << std::setprecision(16) << step << ',' << time << ','
        << cylindrical_constraint_violation_l2 << ','
        << cylindrical_constraint_violation_linf << ','
        << parallel_constraint_violation_l2 << ','
        << parallel_constraint_violation_linf << ','
        << collocation_constraint_violation_l2 << ','
        << collocation_constraint_violation_linf << ','
        << joint_reaction.force.norm() << ',' << joint_reaction.force.x() << ','
        << joint_reaction.force.y() << ',' << joint_reaction.force.z() << ','
        << joint_reaction.moment.x() << ',' << joint_reaction.moment.y() << ','
        << joint_reaction.moment.z() << ',' << piston_axis_position.x() << ','
        << piston_axis_position.y() << ',' << piston_axis_position.z() << ','
        << piston_axial_disp << ',' << piston_axis_radial_drift << ','
        << lower_axis_radial_drift << ',' << upper_axis_radial_drift << ','
        << world_axis_radial_drift << ',' << applied_bend_force_y << ','
        << cylindrical_row_reconstruction_l2 << ','
        << cylindrical_row_reconstruction_linf << ','
        << lower_parallel_residual_0 << ',' << lower_parallel_residual_1 << ','
        << lower_collocation_residual_0 << ',' << lower_collocation_residual_1
        << ',' << upper_parallel_residual_0 << ',' << upper_parallel_residual_1
        << ',' << upper_collocation_residual_0 << ','
        << upper_collocation_residual_1 << ',' << handle_tip_position.x() << ','
        << handle_tip_position.y() << ',' << handle_tip_position.z() << '\n';
    stream_ << row.str();
    stream_.flush();
  }

 private:
  std::ofstream stream_;
};

}  // namespace engineering_joint
