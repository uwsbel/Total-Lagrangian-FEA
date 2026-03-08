#pragma once

// Demo-local CUDA helpers for collision coupling:
// - Gather FE coefficient positions into DEME's column-major node buffer
// - Scatter DEME per-vertex forces into the unified external-force buffer

void GatherCollisionNodesColumnMajor(double* d_nodes_col_major, int n_nodes,
                                     const double* d_x12, const double* d_y12,
                                     const double* d_z12,
                                     const int* d_coef_idx,
                                     const double* d_z_offset);

// d_f_coll_interleaved: [fx0,fy0,fz0, fx1,fy1,fz1, ...] length 3*n_nodes
// d_f_ext_interleaved:  [fx0,fy0,fz0, fx1,fy1,fz1, ...] length 3*total_coef
void ScatterCollisionForcesToExternal(const double* d_f_coll_interleaved,
                                      int n_nodes, const int* d_coef_idx,
                                      double* d_f_ext_interleaved);

