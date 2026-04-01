/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  OpenAI Codex
 * File:    MixedConstraintSystem.h
 * Brief:   Global mixed-element constraint builder for holistic solves.
 *          Supports point-based scalar constraints spanning ANCF3243 beam
 *          blocks and FEAT10 tetrahedral blocks inside one
 *          FEMultiElementProblem.
 *==============================================================
 *==============================================================*/

#pragma once

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <vector>

#include "../elements/ElementBase.h"
#include "../solvers/FEMultiElementProblem.h"

enum MixedConstraintType {
  kMixedConstraintPointWorldCD = 0,
  kMixedConstraintPointPointCD,
  kMixedConstraintWorldDP1,
  kMixedConstraintDP1,
  kMixedConstraintDP2,
};

enum MixedConstraintTermKind {
  kMixedConstraintTermConstant = 0,
  kMixedConstraintTermUsesA,
  kMixedConstraintTermUsesD,
};

struct MixedConstraintPointBinding {
  static constexpr int kMaxCoefficients = 10;

  int count = 0;
  int coef_indices[kMaxCoefficients] = {-1, -1, -1, -1, -1,
                                        -1, -1, -1, -1, -1};
  double weights[kMaxCoefficients] = {0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0};
};

struct MixedConstraintScalar {
  int type = kMixedConstraintPointWorldCD;
  int axis = 0;
  double target = 0.0;
  double row_scale = 1.0;
  Eigen::Vector3d world_direction = Eigen::Vector3d::Zero();
  MixedConstraintPointBinding points[4];
};

struct MixedConstraintLayout {
  std::vector<MixedConstraintScalar> scalars;
  std::vector<int> point_counts;
  std::vector<int> point_coef_indices;
  std::vector<double> point_weights;
  std::vector<int> term_offsets;
  std::vector<int> term_kinds;
  std::vector<double> term_scales;
  std::vector<int> term_j_indices;
  std::vector<int> term_jt_indices;
  std::vector<int> j_offsets;
  std::vector<int> j_columns;
  std::vector<int> jt_offsets;
  std::vector<int> jt_columns;
};

class MixedConstraintSystem {
 public:
  explicit MixedConstraintSystem(FEMultiElementProblem* problem);
  ~MixedConstraintSystem();

  MixedConstraintSystem(const MixedConstraintSystem&) = delete;
  MixedConstraintSystem& operator=(const MixedConstraintSystem&) = delete;
  MixedConstraintSystem(MixedConstraintSystem&&) = delete;
  MixedConstraintSystem& operator=(MixedConstraintSystem&&) = delete;

  MixedConstraintPointBinding MakeCoefficientBinding(int global_coef) const;

  MixedConstraintPointBinding LocateReferencePoint(
      int block_idx, const Eigen::Vector3d& reference_point) const;

  void AddPointToPointCDAxis(const MixedConstraintPointBinding& p,
                             const MixedConstraintPointBinding& r, int axis,
                             double target = 0.0);
  void AddPointToWorldCDAxis(const MixedConstraintPointBinding& p, int axis,
                             double target);
  void AddWorldDP1Constraint(const MixedConstraintPointBinding& p,
                             const MixedConstraintPointBinding& q,
                             const Eigen::Vector3d& world_direction,
                             double target = 0.0, double weight = 1.0);
  void AddDP1Constraint(const MixedConstraintPointBinding& p,
                        const MixedConstraintPointBinding& q,
                        const MixedConstraintPointBinding& r,
                        const MixedConstraintPointBinding& s,
                        double target = 0.0, double weight = 1.0);
  void AddDP2Constraint(const MixedConstraintPointBinding& p,
                        const MixedConstraintPointBinding& q,
                        const MixedConstraintPointBinding& r,
                        const MixedConstraintPointBinding& s,
                        double target = 0.0, double weight = 1.0);

  void AddSphericalJoint(const MixedConstraintPointBinding& p,
                         const MixedConstraintPointBinding& r);
  void AddRevoluteJoint(const MixedConstraintPointBinding& p,
                        const MixedConstraintPointBinding& q,
                        const MixedConstraintPointBinding& r,
                        const MixedConstraintPointBinding& s,
                        const MixedConstraintPointBinding& t, double f1 = 0.0,
                        double f2 = 0.0, double dp1_weight = 1.0);
  void AddFixedJoint(const MixedConstraintPointBinding& p,
                     const MixedConstraintPointBinding& q,
                     const MixedConstraintPointBinding& w,
                     const MixedConstraintPointBinding& r,
                     const MixedConstraintPointBinding& s,
                     const MixedConstraintPointBinding& t, double f1 = 0.0,
                     double f2 = 0.0, double f3 = 0.0,
                     double dp1_weight = 1.0);
  void AddCylindricalJoint(const MixedConstraintPointBinding& p,
                           const MixedConstraintPointBinding& q,
                           const MixedConstraintPointBinding& r,
                           const MixedConstraintPointBinding& s,
                           const MixedConstraintPointBinding& u,
                           const MixedConstraintPointBinding& v,
                           const MixedConstraintPointBinding& w,
                           double f_par1 = 0.0, double f_par2 = 0.0,
                           double f_col1 = 0.0, double f_col2 = 0.0,
                           double dp1_weight = 1.0,
                           double dp2_weight = 1.0);

  void AddSphericalJoint(int block_idx_a, int block_idx_b,
                         const Eigen::Vector3d& hinge_point);
  void AddRevoluteJoint(int block_idx_a, int block_idx_b,
                        const Eigen::Vector3d& hinge_point,
                        const Eigen::Vector3d& hinge_axis,
                        double offset = -1.0, double dp1_weight = 1.0);
  void AddCylindricalJoint(int block_idx_a, int block_idx_b,
                           const Eigen::Vector3d& axis_point_a,
                           const Eigen::Vector3d& axis_point_b,
                           const Eigen::Vector3d& axis_direction,
                           double offset = -1.0, double dp1_weight = 1.0,
                           double dp2_weight = 1.0);

  void Finalize();
  void Evaluate(double* d_x, double* d_y, double* d_z);

  bool IsFinalized() const { return finalized_; }
  int num_constraints() const { return n_constraints_; }

  const MixedConstraintLayout& host_layout() const { return host_layout_; }

  double* GetConstraintDevicePtr() const { return d_constraint_; }
  int* GetConstraintJacobianOffsetsDevicePtr() const { return d_j_csr_offsets_; }
  int* GetConstraintJacobianColumnsDevicePtr() const { return d_j_csr_columns_; }
  double* GetConstraintJacobianValuesDevicePtr() const {
    return d_j_csr_values_;
  }
  int* GetConstraintTypesDevicePtr() const { return d_constraint_types_; }
  double* GetConstraintRowScalesDevicePtr() const {
    return d_constraint_row_scales_;
  }
  int* GetPointCountsDevicePtr() const { return d_point_counts_; }
  int* GetPointCoefficientIndicesDevicePtr() const {
    return d_point_coef_indices_;
  }
  double* GetPointWeightsDevicePtr() const { return d_point_weights_; }
  int* GetConstraintJacobianTransposeOffsetsDevicePtr() const {
    return d_cj_csr_offsets_;
  }
  int* GetConstraintJacobianTransposeColumnsDevicePtr() const {
    return d_cj_csr_columns_;
  }
  double* GetConstraintJacobianTransposeValuesDevicePtr() const {
    return d_cj_csr_values_;
  }
  bool HasNonlinearDotConstraints() const;

 private:
  struct ScalarConstraintTerm {
    int dof = -1;
    int kind = kMixedConstraintTermConstant;
    double scale = 0.0;
  };

  struct BlockCache {
    ElementType type = TYPE_T10;
    int coef_offset = 0;
    int coef_count = 0;

    Eigen::VectorXd x_ref;
    Eigen::VectorXd y_ref;
    Eigen::VectorXd z_ref;
    Eigen::MatrixXi connectivity;

    Eigen::VectorXd beam_length;
    Eigen::VectorXd beam_width;
    Eigen::VectorXd beam_height;
    std::vector<Eigen::Matrix<double, 8, 8>> beam_B_inv;
  };

  FEMultiElementProblem* problem_ = nullptr;
  std::vector<BlockCache> block_cache_;
  Eigen::VectorXd reference_x_;
  Eigen::VectorXd reference_y_;
  Eigen::VectorXd reference_z_;
  std::vector<MixedConstraintScalar> scalar_constraints_;
  MixedConstraintLayout host_layout_;
  bool finalized_ = false;
  int n_constraints_ = 0;

  double* d_constraint_ = nullptr;
  int* d_constraint_types_ = nullptr;
  int* d_constraint_axes_ = nullptr;
  double* d_constraint_targets_ = nullptr;
  double* d_constraint_row_scales_ = nullptr;
  double* d_constraint_world_directions_ = nullptr;
  int* d_point_counts_ = nullptr;
  int* d_point_coef_indices_ = nullptr;
  double* d_point_weights_ = nullptr;
  int* d_term_offsets_ = nullptr;
  int* d_term_kinds_ = nullptr;
  double* d_term_scales_ = nullptr;
  int* d_term_j_indices_ = nullptr;
  int* d_term_jt_indices_ = nullptr;
  int* d_j_csr_offsets_ = nullptr;
  int* d_j_csr_columns_ = nullptr;
  double* d_j_csr_values_ = nullptr;
  int* d_cj_csr_offsets_ = nullptr;
  int* d_cj_csr_columns_ = nullptr;
  double* d_cj_csr_values_ = nullptr;

  void BuildBlockCaches();
  Eigen::Vector3d EvaluateReferencePoint(
      const MixedConstraintPointBinding& point) const;
  double ComputeDefaultOffset(const MixedConstraintPointBinding& a,
                              const MixedConstraintPointBinding& b) const;
  MixedConstraintPointBinding LocateWithAdaptiveOffset(
      int block_idx, const Eigen::Vector3d& base_point,
      const Eigen::Vector3d& direction, double initial_offset) const;
  void AddDotProductConstraint(int type, const MixedConstraintPointBinding& p,
                               const MixedConstraintPointBinding& q,
                               const MixedConstraintPointBinding& r,
                               const MixedConstraintPointBinding& s,
                               double target, double weight);
  MixedConstraintLayout BuildLayout() const;
  MixedConstraintPointBinding LocateReferencePointT10(
      const BlockCache& cache, const Eigen::Vector3d& reference_point) const;
  MixedConstraintPointBinding LocateReferencePointANCF3243(
      const BlockCache& cache, const Eigen::Vector3d& reference_point) const;
};
