#pragma once

namespace PrescribedShake {

// Offsets a set of nodes by delta along one axis, updating both the position
// array and the constraint-target array (Jacobian target) so constraints remain
// satisfied for those nodes.
void OffsetNodesAndTargets(double* d_pos_axis, double* d_target_axis,
                           const int* d_node_ids, int n_node_ids,
                           double delta);

}  // namespace PrescribedShake

