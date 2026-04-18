/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * File:    FEAT10ConstraintManager.h
 * Brief:   FEAT10-specific constraint/joint builder that translates
 *          user-facing fixed-node and engineering-joint requests into the
 *          general constraint layout consumed by GPU_FEAT10_Data.
 *==============================================================
 *==============================================================*/

#pragma once

#include <Eigen/Dense>
#include <vector>

#include "../elements/FEAT10Data.cuh"

// FEAT10-only constraint manager that owns the host-side setup phase for
// constraints on a single aggregated T10 mesh.
//
// Conceptually, this class sits between demo/application code and
// GPU_FEAT10_Data:
//   1. Callers describe constraints in geometric terms such as "fix this
//      node", "coincide these two points", or "add a revolute joint".
//   2. The manager resolves those requests into FEAT10 reference-space points
//      and scalar primitive constraints.
//   3. Finalize() lowers the accumulated primitives into the sparse
//      FEAT10GeneralConstraintLayout expected by GPU_FEAT10_Data.
//
// The manager is intentionally FEAT10-specific because it relies on T10
// element shape functions, FEAT10 connectivity, and FEAT10's runtime
// constraint representation. The implementation is split across the
// generalized manager, primitive builders, and engineering-joint builders
// under lib_src/constraints/.
class FEAT10ConstraintManager {
 public:
  // Half-open element interval [begin, end) used to identify one body inside
  // an aggregated FEAT10 mesh. Engineering-joint demos commonly merge several
  // FE meshes into one GPU_FEAT10_Data instance, then pass body-local ranges
  // here so point-location queries stay on the intended body.
  struct ElementRange {
    int begin = 0;
    int end   = 0;
  };

  // A material point stored in the form FEAT10Data expects:
  //   - the hosting element index
  //   - the T10 shape-function weights evaluated at that material location
  //
  // At runtime, GPU_FEAT10_Data reconstructs the current world-space position
  // by applying these weights to the element's nodal positions.
  using ReferencePoint = FEAT10ConstraintPoint;

  explicit FEAT10ConstraintManager(GPU_FEAT10_Data* data);
  ~FEAT10ConstraintManager();

  FEAT10ConstraintManager(const FEAT10ConstraintManager&)            = delete;
  FEAT10ConstraintManager& operator=(const FEAT10ConstraintManager&) = delete;
  FEAT10ConstraintManager(FEAT10ConstraintManager&&)                 = delete;
  FEAT10ConstraintManager& operator=(FEAT10ConstraintManager&&)      = delete;

  // Adds a node-to-world coordinate-difference primitive.
  //
  // This is the legacy "fixed node" workflow used by many FEAT10 demos.
  // If the manager only receives these nodal constraints, Finalize() keeps the
  // original FEAT10 fixed-node code path instead of switching to the more
  // general joint constraint machinery.
  void AddNodeToWorldCD(int node_id);
  void AddNodesToWorldCD(const Eigen::VectorXi& node_ids);

  // Locate an arbitrary material point in the FEAT10 reference mesh.
  //
  // The returned ReferencePoint contains everything needed to evaluate that
  // material point later, even after the mesh deforms.
  ReferencePoint LocateReferencePoint(
      const Eigen::Vector3d& reference_point) const;
  ReferencePoint LocateReferencePoint(const Eigen::Vector3d& reference_point,
                                      const ElementRange& range) const;
  ReferencePoint LocateReferencePointStrict(
      const Eigen::Vector3d& reference_point) const;
  ReferencePoint LocateReferencePointStrict(
      const Eigen::Vector3d& reference_point,
      const ElementRange& range) const;

  // Locate a material point that must coincide with an exact FEAT10 node.
  //
  // This stricter lookup remains available for demos that intentionally anchor
  // a constraint to a mesh vertex, but the geometry-based joint builders use
  // the general material-point lookup above so they match the appendix
  // construction in the engineering-joint document.
  ReferencePoint LocateNodeReferencePoint(
      const Eigen::Vector3d& reference_point) const;
  ReferencePoint LocateNodeReferencePoint(const Eigen::Vector3d& reference_point,
                                          const ElementRange& range) const;

  // Add a DP1 primitive of the form
  //   (r_Q - r_P) dot (r_S - r_R) = target.
  //
  // In joint language, this constrains the relative orientation of two
  // body-attached directions. The optional weight is folded into the row scale
  // stored in FEAT10GeneralConstraintLayout.
  void AddDP1Constraint(const ReferencePoint& p, const ReferencePoint& q,
                        const ReferencePoint& r, const ReferencePoint& s,
                        double target = 0.0, double weight = 1.0);

  // Add a DP2 primitive with the same scalar equation as DP1:
  //   (r_Q - r_P) dot (r_S - r_R) = target.
  //
  // The intended ownership pattern is that P, Q, and R lie on one body while
  // S lies on the other. The builder also accepts repeated points, which the
  // cylindrical-joint construction uses for the offset-collinearity rows.
  void AddDP2Constraint(const ReferencePoint& p, const ReferencePoint& q,
                        const ReferencePoint& r, const ReferencePoint& s,
                        double target = 0.0, double weight = 1.0);

  // Add a spherical joint directly from its two attachment points.
  //
  // A spherical joint is just the three Cartesian coincidence constraints
  // r_P = r_R, leaving all three relative rotations unconstrained.
  void AddSphericalJoint(const ReferencePoint& p, const ReferencePoint& r);

  // Geometry-based spherical construction from one hinge point shared by two
  // FEAT10 bodies represented as element subranges of the aggregated mesh.
  void AddSphericalJoint(const ElementRange& body_b, const ElementRange& body_c,
                         const Eigen::Vector3d& hinge_point);

  // Geometry-based spherical construction against the world frame.
  //
  // This fixes a material point to its current reference-space world location
  // while leaving all rotations free.
  void AddSphericalJointToWorld(const ElementRange& body,
                                const Eigen::Vector3d& hinge_point);

  // Add a revolute joint from the five-point construction described in the
  // engineering-joint notes:
  //   - P/R define the coincident hinge location
  //   - Q defines the hinge axis on body b
  //   - S/T define two off-axis directions on body c
  //
  // The resulting joint is assembled from three CD rows and two DP1 rows.
  void AddRevoluteJoint(const ReferencePoint& p, const ReferencePoint& q,
                        const ReferencePoint& r, const ReferencePoint& s,
                        const ReferencePoint& t, double f1 = 0.0,
                        double f2 = 0.0, double dp1_weight = 1.0);

  // Geometry-based revolute construction following Appendix A of the
  // engineering-joint document. The two ranges identify the connected FEAT10
  // bodies inside one aggregated mesh.
  //
  // The hinge point and offset points are all resolved as material points via
  // reference-space point location, so the helper follows the document's
  // element-agnostic construction instead of snapping the hinge to a node.
  void AddRevoluteJoint(const ElementRange& body_b, const ElementRange& body_c,
                        const Eigen::Vector3d& hinge_point,
                        const Eigen::Vector3d& hinge_axis, double offset = -1.0,
                        double dp1_weight = 1.0);

  // Geometry-based revolute construction against the world frame.
  //
  // The hinge location is fixed in world space, but rotation about the hinge
  // axis remains free.
  void AddRevoluteJointToWorld(const ElementRange& body,
                               const Eigen::Vector3d& hinge_point,
                               const Eigen::Vector3d& hinge_axis,
                               double offset = -1.0, double dp1_weight = 1.0);

  // Add a fixed (weld) joint from the six-point construction in
  // Appendix A.8 of the engineering-joint notes:
  //   - P/R define the coincident weld location
  //   - Q/W define two independent directions on body b
  //   - S/T define two independent directions on body c
  //
  // The resulting joint is assembled from three CD rows and three DP1 rows.
  void AddFixedJoint(const ReferencePoint& p, const ReferencePoint& q,
                     const ReferencePoint& w, const ReferencePoint& r,
                     const ReferencePoint& s, const ReferencePoint& t,
                     double f1 = 0.0, double f2 = 0.0, double f3 = 0.0,
                     double dp1_weight = 1.0);

  // Geometry-based fixed-joint construction following Appendix A.8 of the
  // engineering-joint document. The weld location is shared by the two bodies,
  // while the orientation-lock directions are generated from one deterministic
  // reference axis and two perpendicular directions in the reference frame.
  void AddFixedJoint(const ElementRange& body_b, const ElementRange& body_c,
                     const Eigen::Vector3d& joint_point,
                     double offset = -1.0, double dp1_weight = 1.0);

  // Add a cylindrical joint from the seven-point construction in Appendix A.7
  // of the engineering-joint notes:
  //   - P/R are attachment points on the shared axis
  //   - Q defines the axis direction on body b
  //   - S/U define the two axis-parallelism directions on body c
  //   - V/W define the two offset-collinearity directions on body b
  //
  // The resulting joint is assembled from two DP1 rows and two DP2 rows.
  void AddCylindricalJoint(const ReferencePoint& p, const ReferencePoint& q,
                           const ReferencePoint& r, const ReferencePoint& s,
                           const ReferencePoint& u, const ReferencePoint& v,
                           const ReferencePoint& w, double f_par1 = 0.0,
                           double f_par2 = 0.0, double f_col1 = 0.0,
                           double f_col2 = 0.0, double dp1_weight = 1.0,
                           double dp2_weight = 1.0);

  // Shared cup-side frame used by the cylindrical-joint construction.
  //
  // P is the guide-axis base point, Q defines the guide-axis direction, and
  // V/W define two perpendicular guide directions used by the DP2
  // collinearity rows.
  struct CylindricalGuideFrame {
    ReferencePoint p;
    ReferencePoint q;
    ReferencePoint v;
    ReferencePoint w;
  };

  // Build the cup-side guide frame once so callers can reuse the exact same
  // material points for world-fixing, cylindrical-joint construction, and
  // diagnostics.
  CylindricalGuideFrame BuildCylindricalGuideFrame(
      const ElementRange& body,
      const Eigen::Vector3d& axis_point,
      const Eigen::Vector3d& axis_direction,
      double offset = -1.0) const;

  // Freeze a previously-built cylindrical guide frame to the world frame.
  //
  // This keeps P fixed in world space, keeps Q - P aligned with the frame's
  // reference axis, and keeps V - P / W - P aligned with the frame's
  // reference perpendicular directions.
  void AddCylindricalGuideFrameToWorld(const CylindricalGuideFrame& frame,
                                       double dp1_weight = 1.0);

  // Convenience overload that builds the guide frame internally before fixing
  // it to world.
  void AddCylindricalGuideFrameToWorld(const ElementRange& body,
                                       const Eigen::Vector3d& axis_point,
                                       const Eigen::Vector3d& axis_direction,
                                       double offset = -1.0,
                                       double dp1_weight = 1.0);

  // Fully-resolved seven-point cylindrical-joint construction, including the
  // reference-space targets derived from those material points.
  struct CylindricalJointGeometry {
    ReferencePoint p;
    ReferencePoint q;
    ReferencePoint r;
    ReferencePoint s;
    ReferencePoint u;
    ReferencePoint v;
    ReferencePoint w;
    double f_par1 = 0.0;
    double f_par2 = 0.0;
    double f_col1 = 0.0;
    double f_col2 = 0.0;
  };

  CylindricalJointGeometry BuildCylindricalJointGeometry(
      const CylindricalGuideFrame& guide_frame,
      const ElementRange& body_c,
      const Eigen::Vector3d& axis_point_c,
      double offset = -1.0) const;
  CylindricalJointGeometry BuildCylindricalJointGeometry(
      const ElementRange& body_b, const ElementRange& body_c,
      const Eigen::Vector3d& axis_point_b,
      const Eigen::Vector3d& axis_point_c,
      const Eigen::Vector3d& axis_direction,
      double offset = -1.0) const;
  void AddCylindricalJoint(const CylindricalJointGeometry& geometry,
                           double dp1_weight = 1.0,
                           double dp2_weight = 1.0);

  // Geometry-based cylindrical construction following Appendix A.7 of the
  // engineering-joint document.
  void AddCylindricalJoint(const ElementRange& body_b,
                           const ElementRange& body_c,
                           const Eigen::Vector3d& axis_point_b,
                           const Eigen::Vector3d& axis_point_c,
                           const Eigen::Vector3d& axis_direction,
                           double offset = -1.0,
                           double dp1_weight = 1.0,
                           double dp2_weight = 1.0);

  // Freeze the host-side description and install it into GPU_FEAT10_Data.
  //
  // After Finalize():
  //   - duplicate nodal constraints are deduplicated
  //   - device-side node lists are allocated
  //   - GPU_FEAT10_Data is configured either for the legacy fixed-node mode or
  //     for the general sparse constraint mode used by joints
  //
  // No new primitives may be added after this point.
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
  // on one Cartesian axis (0:x, 1:y, 2:z).
  //
  // This helper only exists for the original fixed-node FEAT10 workflow.
  // General joint constraints currently store full primitive descriptions
  // rather than a simple movable target vector, so they are intentionally not
  // supported here.
  void OffsetConstrainedNodesAndTargets(int axis, double delta);

 private:
  // Host-side description of one scalar primitive constraint before it is
  // lowered into FEAT10GeneralConstraintLayout.
  struct ScalarConstraint {
    int type                        = kFEAT10ConstraintNodeWorldCD;
    int axis                        = 0;
    int node_id                     = -1;
    double target                   = 0.0;
    double row_scale                = 1.0;
    Eigen::Vector3d world_direction = Eigen::Vector3d::Zero();
    ReferencePoint points[4];
  };

  // One contribution term in the sparse Jacobian/Jacobian-transpose pattern.
  // BuildGeneralConstraintLayout() uses these descriptors to assemble the row
  // and transpose sparsity patterns expected by GPU_FEAT10_Data.
  struct TermDescriptor {
    int dof      = -1;
    int kind     = kFEAT10ConstraintTermConstant;
    double scale = 0.0;
  };

  // Lazily pull FEAT10 reference positions and connectivity to the CPU so
  // point-location and joint construction can run with simple Eigen code.
  void EnsureReferenceCache() const;

  // Convenience wrapper for the whole aggregated mesh.
  ElementRange FullElementRange() const;

  // Low-level CD primitive builders used by the higher-level joint helpers.
  void AddPointToPointCDAxis(const ReferencePoint& p, const ReferencePoint& r,
                             int axis, double target = 0.0);
  void AddPointToWorldCDAxis(const ReferencePoint& p, int axis, double target);

  // DP1 helper against a fixed world direction rather than another material
  // fiber.
  void AddWorldDP1Constraint(const ReferencePoint& p, const ReferencePoint& q,
                             const Eigen::Vector3d& world_direction,
                             double target = 0.0, double weight = 1.0);

  // Shared host-side helper for FEAT10's bilinear dot-product primitives.
  void AddDotProductConstraint(int type, const ReferencePoint& p,
                               const ReferencePoint& q,
                               const ReferencePoint& r,
                               const ReferencePoint& s,
                               double target = 0.0, double weight = 1.0);

  // Reconstruct the reference-space position of a stored material point.
  Eigen::Vector3d EvaluateReferencePoint(const ReferencePoint& point) const;

  // Mesh-scale helpers used to choose robust geometry-construction offsets.
  double ComputeElementCharacteristicLength(int elem_idx) const;
  double DefaultJointOffset(const ReferencePoint& p,
                            const ReferencePoint& r) const;

  // Build a ReferencePoint that selects one exact local node.
  ReferencePoint BuildNodalReferencePoint(int elem_idx, int local_node) const;

  bool TryLocateReferencePoint(const Eigen::Vector3d& reference_point,
                               const ElementRange& range, bool require_inside,
                               ReferencePoint* point,
                               double* best_residual = nullptr) const;

  // Retry offset point-location with progressively smaller offsets. This keeps
  // the geometry-based joint builders robust near element boundaries.
  ReferencePoint LocateWithAdaptiveOffset(const Eigen::Vector3d& base_point,
                                          const Eigen::Vector3d& direction,
                                          const ElementRange& range,
                                          double initial_offset) const;
  ReferencePoint LocateWithAdaptiveOffsetStrict(
      const Eigen::Vector3d& base_point, const Eigen::Vector3d& direction,
      const ElementRange& range, double initial_offset) const;

  // Lower the accumulated primitive constraints into the sparse scalar layout
  // consumed by GPU_FEAT10_Data::SetGeneralConstraints().
  FEAT10GeneralConstraintLayout BuildGeneralConstraintLayout() const;

  // Non-owning FEAT10 object whose constraint state is being configured.
  GPU_FEAT10_Data* data_ = nullptr;

  // Legacy fixed-node requests gathered before Finalize().
  std::vector<int> node_ids_;

  // General scalar primitives gathered before Finalize().
  std::vector<ScalarConstraint> scalar_constraints_;

  // Deduplicated node list installed into the legacy fixed-node path.
  Eigen::VectorXi constrained_nodes_;

  // CPU-side reference mesh cache used for point-location and geometric joint
  // construction. Marked mutable because callers conceptually perform read-only
  // geometric queries while the cache is materialized lazily.
  mutable bool reference_cache_ready_ = false;
  mutable Eigen::MatrixXd reference_nodes_;
  mutable Eigen::MatrixXi connectivity_;

  // Device mirror of constrained_nodes_ for the legacy fixed-node update path.
  int* d_constrained_nodes_      = nullptr;

  // Lifecycle flags.
  bool finalized_                = false;
  bool uses_general_constraints_ = false;
};
