#pragma once

#include <Eigen/Dense>
#include <vector>

#include "FEAT10Data.cuh"

// FEAT10-only constraint manager that preserves the existing node-to-world
// fixed/moving workflow while adding general DP1 and revolute-joint setup for
// T10 meshes.
class FEAT10ConstraintManager {
 public:
  struct ElementRange {
    int begin = 0;
    int end   = 0;
  };

  using ReferencePoint = FEAT10ConstraintPoint;

  explicit FEAT10ConstraintManager(GPU_FEAT10_Data* data);
  ~FEAT10ConstraintManager();

  FEAT10ConstraintManager(const FEAT10ConstraintManager&)            = delete;
  FEAT10ConstraintManager& operator=(const FEAT10ConstraintManager&) = delete;
  FEAT10ConstraintManager(FEAT10ConstraintManager&&)                 = delete;
  FEAT10ConstraintManager& operator=(FEAT10ConstraintManager&&)      = delete;

  // Adds a node-to-world CD primitive. When no general constraints are added,
  // these primitives still compile to the original fixed-node FEAT10 path.
  void AddNodeToWorldCD(int node_id);
  void AddNodesToWorldCD(const Eigen::VectorXi& node_ids);

  // Locates a reference-space point inside the FEAT10 mesh and returns the
  // hosting element together with its T10 shape-function vector.
  ReferencePoint LocateReferencePoint(
      const Eigen::Vector3d& reference_point) const;
  ReferencePoint LocateReferencePoint(const Eigen::Vector3d& reference_point,
                                      const ElementRange& range) const;
  ReferencePoint LocateNodeReferencePoint(
      const Eigen::Vector3d& reference_point) const;
  ReferencePoint LocateNodeReferencePoint(const Eigen::Vector3d& reference_point,
                                          const ElementRange& range) const;

  // Adds a primitive DP1 constraint of the form (r_Q - r_P) dot (r_S - r_R) =
  // target.
  void AddDP1Constraint(const ReferencePoint& p, const ReferencePoint& q,
                        const ReferencePoint& r, const ReferencePoint& s,
                        double target = 0.0, double weight = 1.0);

  // Adds a revolute joint from the five-point definition in the engineering
  // joint document: three CD constraints for P/R coincidence and two DP1
  // constraints for axis collinearity.
  void AddRevoluteJoint(const ReferencePoint& p, const ReferencePoint& q,
                        const ReferencePoint& r, const ReferencePoint& s,
                        const ReferencePoint& t, double f1 = 0.0,
                        double f2 = 0.0, double dp1_weight = 1.0);

  // Geometry-based revolute construction following Appendix A of the joint
  // document. The two element ranges identify the connected FEAT10 bodies
  // within a single aggregated T10 mesh.
  void AddRevoluteJoint(const ElementRange& body_b, const ElementRange& body_c,
                        const Eigen::Vector3d& hinge_point,
                        const Eigen::Vector3d& hinge_axis, double offset = -1.0,
                        double dp1_weight = 1.0);

  // Geometry-based revolute construction against the world. The hinge point
  // is fixed in world space, while the hinge axis remains free to rotate.
  void AddRevoluteJointToWorld(const ElementRange& body,
                               const Eigen::Vector3d& hinge_point,
                               const Eigen::Vector3d& hinge_axis,
                               double offset = -1.0, double dp1_weight = 1.0);

  // Installs the accumulated primitives into GPU_FEAT10_Data. Pure fixed-node
  // usage stays on the original FEAT10 fixed-node path; any DP1/revolute work
  // switches to the FEAT10 general-constraint path.
  void Finalize();

  bool IsFinalized() const {
    return finalized_;
  }
  bool uses_general_constraints() const {
    return uses_general_constraints_;
  }
  int num_primitives() const {
    return static_cast<int>(node_ids_.size() + scalar_constraints_.size());
  }
  int num_constrained_nodes() const {
    return constrained_nodes_.size();
  }

  const Eigen::VectorXi& constrained_nodes() const {
    return constrained_nodes_;
  }

  // Applies a rigid offset to the constrained nodes and their target positions
  // on one Cartesian axis (0:x, 1:y, 2:z). This remains available for the
  // original fixed-node FEAT10 path used by demos such as test_vase.
  void OffsetConstrainedNodesAndTargets(int axis, double delta);

 private:
  struct ScalarConstraint {
    int type                        = kFEAT10ConstraintNodeWorldCD;
    int axis                        = 0;
    int node_id                     = -1;
    double target                   = 0.0;
    double row_scale                = 1.0;
    Eigen::Vector3d world_direction = Eigen::Vector3d::Zero();
    ReferencePoint points[4];
  };

  struct TermDescriptor {
    int dof      = -1;
    int kind     = kFEAT10ConstraintTermConstant;
    double scale = 0.0;
  };

  void EnsureReferenceCache() const;
  ElementRange FullElementRange() const;
  void AddPointToPointCDAxis(const ReferencePoint& p, const ReferencePoint& r,
                             int axis, double target = 0.0);
  void AddPointToWorldCDAxis(const ReferencePoint& p, int axis, double target);
  void AddWorldDP1Constraint(const ReferencePoint& p, const ReferencePoint& q,
                             const Eigen::Vector3d& world_direction,
                             double target = 0.0, double weight = 1.0);
  Eigen::Vector3d EvaluateReferencePoint(const ReferencePoint& point) const;
  double ComputeElementCharacteristicLength(int elem_idx) const;
  double DefaultJointOffset(const ReferencePoint& p,
                            const ReferencePoint& r) const;
  ReferencePoint BuildNodalReferencePoint(int elem_idx, int local_node) const;
  ReferencePoint LocateWithAdaptiveOffset(const Eigen::Vector3d& base_point,
                                          const Eigen::Vector3d& direction,
                                          const ElementRange& range,
                                          double initial_offset) const;
  FEAT10GeneralConstraintLayout BuildGeneralConstraintLayout() const;

  GPU_FEAT10_Data* data_ = nullptr;

  std::vector<int> node_ids_;
  std::vector<ScalarConstraint> scalar_constraints_;
  Eigen::VectorXi constrained_nodes_;

  mutable bool reference_cache_ready_ = false;
  mutable Eigen::MatrixXd reference_nodes_;
  mutable Eigen::MatrixXi connectivity_;

  int* d_constrained_nodes_      = nullptr;
  bool finalized_                = false;
  bool uses_general_constraints_ = false;
};
