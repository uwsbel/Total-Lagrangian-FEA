#pragma once

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "../lib_src/collision/DemeMeshCollisionSystem.h"
#include "mesh_manager.h"

namespace CollisionMassProperties {

struct RigidMassProperties {
  double mass = 1.0;
  Eigen::Vector3d com = Eigen::Vector3d::Zero();
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d principal_moi = Eigen::Vector3d::Ones();
};

inline RigidMassProperties ComputeFromLumpedNodes(
    const Eigen::MatrixXd& nodes, int node_offset, int num_nodes,
    const std::vector<double>& lump_masses, int lump_offset = 0) {
  RigidMassProperties props;
  const int n_node_rows = static_cast<int>(nodes.rows());
  if (node_offset < 0 || num_nodes <= 0 || node_offset + num_nodes > n_node_rows ||
      lump_offset < 0 ||
      lump_offset + num_nodes > static_cast<int>(lump_masses.size())) {
    return props;
  }

  props.mass = 0.0;
  props.com.setZero();
  for (int i = 0; i < num_nodes; ++i) {
    const double m = std::max(0.0, lump_masses[static_cast<size_t>(lump_offset + i)]);
    const Eigen::Vector3d x = nodes.row(node_offset + i).transpose();
    props.mass += m;
    props.com += m * x;
  }

  if (!(props.mass > 0.0)) {
    props.mass = 1.0;
    props.com.setZero();
    for (int i = 0; i < num_nodes; ++i) {
      props.com += nodes.row(node_offset + i).transpose();
    }
    props.com /= static_cast<double>(num_nodes);
    return props;
  }
  props.com /= props.mass;

  Eigen::Matrix3d inertia = Eigen::Matrix3d::Zero();
  for (int i = 0; i < num_nodes; ++i) {
    const double m = std::max(0.0, lump_masses[static_cast<size_t>(lump_offset + i)]);
    if (!(m > 0.0)) {
      continue;
    }
    const Eigen::Vector3d r = nodes.row(node_offset + i).transpose() - props.com;
    inertia += m * (r.squaredNorm() * Eigen::Matrix3d::Identity() - r * r.transpose());
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(inertia);
  if (eig.info() == Eigen::Success) {
    Eigen::Matrix3d axes = eig.eigenvectors();
    if (axes.determinant() < 0.0) {
      axes.col(2) *= -1.0;
    }
    props.orientation = Eigen::Quaterniond(axes).normalized();
    props.principal_moi = eig.eigenvalues().cwiseMax(0.0);
  } else {
    props.orientation = Eigen::Quaterniond::Identity();
    props.principal_moi = inertia.diagonal().cwiseMax(0.0);
  }

  constexpr double kMoiFloor = 1e-12;
  props.principal_moi =
      props.principal_moi.cwiseMax(Eigen::Vector3d::Constant(kMoiFloor));
  return props;
}

inline RigidMassProperties ComputeFromLumpedNodes(
    const Eigen::MatrixXd& nodes, const ANCFCPUUtils::MeshInstance& inst,
    const std::vector<double>& lump_masses, int lump_offset = 0) {
  return ComputeFromLumpedNodes(nodes, inst.node_offset, inst.num_nodes, lump_masses,
                                lump_offset);
}

inline RigidMassProperties ComputeRectangularPrism(
    double mass, const Eigen::Vector3d& center, const Eigen::Vector3d& size,
    const Eigen::Quaterniond& orientation = Eigen::Quaterniond::Identity()) {
  RigidMassProperties props;
  props.mass = std::max(mass, 1e-12);
  props.com = center;
  props.orientation = orientation.normalized();
  props.principal_moi(0) =
      props.mass * (size.y() * size.y() + size.z() * size.z()) / 12.0;
  props.principal_moi(1) =
      props.mass * (size.x() * size.x() + size.z() * size.z()) / 12.0;
  props.principal_moi(2) =
      props.mass * (size.x() * size.x() + size.y() * size.y()) / 12.0;
  constexpr double kMoiFloor = 1e-12;
  props.principal_moi =
      props.principal_moi.cwiseMax(Eigen::Vector3d::Constant(kMoiFloor));
  return props;
}

inline void AssignToCollisionBody(DemeMeshCollisionBody& body,
                                  const RigidMassProperties& props) {
  body.mass = static_cast<float>(std::max(props.mass, 1e-12));
  body.moi = make_float3(static_cast<float>(props.principal_moi.x()),
                         static_cast<float>(props.principal_moi.y()),
                         static_cast<float>(props.principal_moi.z()));
  body.reference_point = props.com;
  body.reference_orientation = props.orientation.normalized();
  body.has_rigid_mass_properties = true;
}

}  // namespace CollisionMassProperties
