/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    DemeMeshCollisionSystem.cu
 * Brief:   Implements a DEM-Engine (DEME) based mesh collision backend,
 *          including runtime path configuration, mesh deformation updates, and
 *          per-node contact force accumulation for FE coupling.
 *==============================================================
 *==============================================================*/

#include <DEM/API.h>
#include <core/utils/JitHelper.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "DemeMeshCollisionSystem.h"
#include "lib_utils/cuda_utils.h"

namespace {

static void ConfigureDemeRuntimePathsFromBazelBin() {
  namespace fs = std::filesystem;

  // DEM-Engine bakes an absolute data/include path into `DEMERuntimeDataHelper`
  // at CMake configure time. Under Bazel + sandboxing, that path points into a
  // sandbox that does not exist at runtime, so we override it here.

  auto try_set_from_install = [&](const fs::path& dem_install) {
    const fs::path data_path    = dem_install / "share" / "DEME";
    const fs::path include_path = dem_install / "include";

    if (fs::exists(data_path)) {
      DEMERuntimeDataHelper::data_path = data_path;
      JitHelper::KERNEL_DIR = DEMERuntimeDataHelper::data_path / "kernel";
    }
    if (fs::exists(include_path)) {
      DEMERuntimeDataHelper::include_path = include_path;
      JitHelper::KERNEL_INCLUDE_DIR       = DEMERuntimeDataHelper::include_path;
    }

    return fs::exists(DEMERuntimeDataHelper::data_path / "kernel") &&
           fs::exists(DEMERuntimeDataHelper::include_path);
  };

  // Explicit overrides (useful when running outside Bazel output trees).
  if (const char* data = std::getenv("DEME_DATA_PATH"); data && *data) {
    const fs::path data_path(data);
    if (fs::exists(data_path)) {
      DEMERuntimeDataHelper::data_path = data_path;
      JitHelper::KERNEL_DIR = DEMERuntimeDataHelper::data_path / "kernel";
    }
  }
  if (const char* inc = std::getenv("DEME_INCLUDE_PATH"); inc && *inc) {
    const fs::path include_path(inc);
    if (fs::exists(include_path)) {
      DEMERuntimeDataHelper::include_path = include_path;
      JitHelper::KERNEL_INCLUDE_DIR       = DEMERuntimeDataHelper::include_path;
    }
  }
  if (fs::exists(DEMERuntimeDataHelper::data_path / "kernel") &&
      fs::exists(DEMERuntimeDataHelper::include_path)) {
    return;
  }

  std::error_code ec;
  fs::path exe = fs::read_symlink("/proc/self/exe", ec);
  if (ec)
    return;
  exe = fs::weakly_canonical(exe, ec);
  if (ec)
    return;

  // When the binary is executed from `bazel-bin/...`, `/proc/self/exe` usually
  // resolves into the output tree:
  //   .../execroot/_main/bazel-out/.../bin/<pkg>/<name>
  //
  // The `external/` dir lives under the `bin/` root, not next to the
  // executable, so search upward from the executable directory for a matching
  // install tree.
  const fs::path install_rel =
      fs::path("external") / "+_repo_rules+dem_engine" / "dem_engine";
  for (fs::path p = exe.parent_path(); !p.empty(); p = p.parent_path()) {
    if (try_set_from_install(p / install_rel))
      return;
  }

  // Fallback: common symlink location in the workspace when launched as
  // `./bazel-bin/...`.
  if (try_set_from_install(fs::path("bazel-bin") / install_rel))
    return;
}

static float EnvFloatOr(const char* name, float default_value) {
  const char* s = std::getenv(name);
  if (!s || !*s)
    return default_value;
  char* end      = nullptr;
  const double v = std::strtod(s, &end);
  if (end == s)
    return default_value;
  return static_cast<float>(v);
}

static float EnvFloatOrFallback(const char* primary, const char* fallback,
                                float default_value) {
  const float nan       = std::numeric_limits<float>::quiet_NaN();
  const float v_primary = EnvFloatOr(primary, nan);
  if (std::isfinite(v_primary))
    return v_primary;
  const float v_fallback = EnvFloatOr(fallback, nan);
  if (std::isfinite(v_fallback))
    return v_fallback;
  return default_value;
}

static double EnvDoubleOr(const char* name, double default_value) {
  const char* s = std::getenv(name);
  if (!s || !*s)
    return default_value;
  char* end      = nullptr;
  const double v = std::strtod(s, &end);
  if (end == s)
    return default_value;
  return v;
}

static float3 ToFloat3(const Eigen::Vector3d& v) {
  return make_float3(static_cast<float>(v.x()), static_cast<float>(v.y()),
                     static_cast<float>(v.z()));
}

static Eigen::Vector3d ComputeCentroid(
    const std::vector<Eigen::Vector3d>& vertices) {
  if (vertices.empty())
    return Eigen::Vector3d::Zero();
  Eigen::Vector3d c = Eigen::Vector3d::Zero();
  for (const auto& v : vertices)
    c += v;
  c /= static_cast<double>(vertices.size());
  return c;
}

[[maybe_unused]] static float3 MeanOf(const std::vector<float3>& nodes) {
  if (nodes.empty())
    return make_float3(0.f, 0.f, 0.f);
  double sx = 0.0, sy = 0.0, sz = 0.0;
  for (const auto& p : nodes) {
    sx += static_cast<double>(p.x);
    sy += static_cast<double>(p.y);
    sz += static_cast<double>(p.z);
  }
  const double inv = 1.0 / static_cast<double>(nodes.size());
  return make_float3(static_cast<float>(sx * inv), static_cast<float>(sy * inv),
                     static_cast<float>(sz * inv));
}

[[maybe_unused]] static std::vector<float3> Subtract(const std::vector<float3>& nodes,
                                    const float3& t) {
  std::vector<float3> out(nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    out[i] = make_float3(nodes[i].x - t.x, nodes[i].y - t.y, nodes[i].z - t.z);
  }
  return out;
}

static float4 IdentityQuat() {
  return make_float4(0.f, 0.f, 0.f, 1.f);
}

static float4 ToQuatXYZW(const Eigen::Quaterniond& q) {
  Eigen::Quaterniond qq = q.normalized();
  return make_float4(static_cast<float>(qq.x()), static_cast<float>(qq.y()),
                     static_cast<float>(qq.z()), static_cast<float>(qq.w()));
}

[[maybe_unused]] static float3 Vec3FromNodesXYZ(const std::vector<double>& h_nodes_xyz,
                               int n_nodes, int global_node_id) {
  const double x = h_nodes_xyz[global_node_id];
  const double y = h_nodes_xyz[n_nodes + global_node_id];
  const double z = h_nodes_xyz[2 * n_nodes + global_node_id];
  return make_float3(static_cast<float>(x), static_cast<float>(y),
                     static_cast<float>(z));
}

[[maybe_unused]] static Eigen::Matrix3d BestFitRotationKabsch(
    const std::vector<Eigen::Vector3d>& ref_local,
    const std::vector<float3>& cur_global,
    const Eigen::Vector3d& cur_centroid) {
  if (ref_local.size() != cur_global.size() || ref_local.size() < 3) {
    return Eigen::Matrix3d::Identity();
  }

  Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
  for (size_t i = 0; i < ref_local.size(); ++i) {
    const Eigen::Vector3d p = ref_local[i];
    const Eigen::Vector3d q(static_cast<double>(cur_global[i].x) - cur_centroid.x(),
                            static_cast<double>(cur_global[i].y) - cur_centroid.y(),
                            static_cast<double>(cur_global[i].z) - cur_centroid.z());
    H += p * q.transpose();
  }

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d V = svd.matrixV();
  const Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d R = V * U.transpose();

  // Fix reflection if needed.
  if (R.determinant() < 0.0) {
    V.col(2) *= -1.0;
    R = V * U.transpose();
  }
  return R;
}

static Eigen::Matrix3d BestFitRotationFromH(const Eigen::Matrix3d& H) {
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d V = svd.matrixV();
  const Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d R = V * U.transpose();
  if (R.determinant() < 0.0) {
    V.col(2) *= -1.0;
    R = V * U.transpose();
  }
  return R;
}

[[maybe_unused]] static void AccumulatePointForcesToKNearestNodes(
    const std::vector<float3>& mesh_vertices,
    const std::vector<int>& global_node_ids, const std::vector<float3>& points,
    const std::vector<float3>& forces, int k, double force_scale,
    double clamp_force_norm, double damping_scale,
    std::vector<double>& h_f_contact) {
  if (k <= 0)
    return;
  const double combined_scale = force_scale * damping_scale;
  for (size_t i = 0; i < points.size() && i < forces.size(); ++i) {
    float3 f = forces[i];
    f.x      = static_cast<float>(static_cast<double>(f.x) * combined_scale);
    f.y      = static_cast<float>(static_cast<double>(f.y) * combined_scale);
    f.z      = static_cast<float>(static_cast<double>(f.z) * combined_scale);

    if (clamp_force_norm > 0.0) {
      const double fn = std::sqrt(static_cast<double>(f.x) * f.x +
                                  static_cast<double>(f.y) * f.y +
                                  static_cast<double>(f.z) * f.z);
      if (fn > clamp_force_norm && fn > 0.0) {
        const double s = clamp_force_norm / fn;
        f.x            = static_cast<float>(static_cast<double>(f.x) * s);
        f.y            = static_cast<float>(static_cast<double>(f.y) * s);
        f.z            = static_cast<float>(static_cast<double>(f.z) * s);
      }
    }

    // Find k nearest vertices (small k, brute force).
    constexpr int kMax = 8;
    const int kk       = std::min(k, kMax);
    int best_idx[kMax];
    double best_d2[kMax];
    for (int j = 0; j < kk; ++j) {
      best_idx[j] = -1;
      best_d2[j]  = std::numeric_limits<double>::infinity();
    }

    for (int vi = 0; vi < static_cast<int>(mesh_vertices.size()); ++vi) {
      const double dx = static_cast<double>(mesh_vertices[vi].x) -
                        static_cast<double>(points[i].x);
      const double dy = static_cast<double>(mesh_vertices[vi].y) -
                        static_cast<double>(points[i].y);
      const double dz = static_cast<double>(mesh_vertices[vi].z) -
                        static_cast<double>(points[i].z);
      const double d2 = dx * dx + dy * dy + dz * dz;

      // Insert into sorted best list (descending slot replacement).
      int worst = 0;
      for (int j = 1; j < kk; ++j) {
        if (best_d2[j] > best_d2[worst])
          worst = j;
      }
      if (d2 >= best_d2[worst])
        continue;
      best_d2[worst]  = d2;
      best_idx[worst] = vi;
    }

    // Inverse-distance weights.
    double wsum = 0.0;
    double w[kMax];
    for (int j = 0; j < kk; ++j) {
      if (best_idx[j] < 0) {
        w[j] = 0.0;
        continue;
      }
      w[j] = 1.0 / (best_d2[j] + 1e-18);
      wsum += w[j];
    }
    if (wsum <= 0.0)
      continue;

    for (int j = 0; j < kk; ++j) {
      const int vi = best_idx[j];
      if (vi < 0)
        continue;
      const int global = global_node_ids[static_cast<size_t>(vi)];
      const double a   = w[j] / wsum;
      h_f_contact[3 * global + 0] += a * static_cast<double>(f.x);
      h_f_contact[3 * global + 1] += a * static_cast<double>(f.y);
      h_f_contact[3 * global + 2] += a * static_cast<double>(f.z);
    }
  }
}

static void ExtendBBox(Eigen::Vector3d& mn, Eigen::Vector3d& mx,
                       const Eigen::Vector3d& p) {
  mn = mn.cwiseMin(p);
  mx = mx.cwiseMax(p);
}

__device__ inline double AtomicAddDouble(double* address, double val) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, val);
#else
  unsigned long long int* address_as_ull =
      reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
#endif
}

__device__ inline void AtomicAddDouble3(double3* addr, const double3& v) {
  AtomicAddDouble(&addr->x, v.x);
  AtomicAddDouble(&addr->y, v.y);
  AtomicAddDouble(&addr->z, v.z);
}

__device__ inline float3 QuatRotate(const float4& q, const float3& v) {
  const float3 qv = make_float3(q.x, q.y, q.z);
  const float3 c1 = cross(qv, v);
  const float3 t = make_float3(2.f * c1.x, 2.f * c1.y, 2.f * c1.z);
  const float3 c2 = cross(qv, t);
  return make_float3(v.x + q.w * t.x + c2.x,
                     v.y + q.w * t.y + c2.y,
                     v.z + q.w * t.z + c2.z);
}

__global__ void GatherSurfPosGlobalKernel(const double* d_nodes_xyz, int n_nodes,
                                         const int* surf_global_node_id,
                                         float3* surf_pos_global,
                                         int n_surf_verts) {
  const int pid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (pid >= n_surf_verts)
    return;
  const int gid = surf_global_node_id[pid];
  const double x = d_nodes_xyz[gid];
  const double y = d_nodes_xyz[n_nodes + gid];
  const double z = d_nodes_xyz[2 * n_nodes + gid];
  surf_pos_global[pid] =
      make_float3(static_cast<float>(x), static_cast<float>(y),
                  static_cast<float>(z));
}

__global__ void AccumulateBodyCOMKernel(const int* surf_body_id,
                                       const float3* surf_pos_global,
                                       double3* body_com_sum,
                                       int n_surf_verts) {
  const int pid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (pid >= n_surf_verts)
    return;
  const int bi = surf_body_id[pid];
  const float3 p = surf_pos_global[pid];
  AtomicAddDouble3(&body_com_sum[bi],
                   make_double3(static_cast<double>(p.x),
                                static_cast<double>(p.y),
                                static_cast<double>(p.z)));
}

__global__ void NormalizeBodyCOMKernel(double3* body_com_sum,
                                      const double* inv_count,
                                      int n_bodies) {
  const int bi = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (bi >= n_bodies)
    return;
  const double inv = inv_count[bi];
  body_com_sum[bi].x *= inv;
  body_com_sum[bi].y *= inv;
  body_com_sum[bi].z *= inv;
}

__global__ void AccumulateBodyHKernel(const int* surf_body_id,
                                     const double3* surf_ref_local,
                                     const float3* surf_pos_global,
                                     const double3* body_com,
                                     double* body_H,
                                     int n_surf_verts) {
  const int pid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (pid >= n_surf_verts)
    return;
  const int bi = surf_body_id[pid];
  const double3 p = surf_ref_local[pid];
  const float3 pg = surf_pos_global[pid];
  const double3 c = body_com[bi];
  const double qx = static_cast<double>(pg.x) - c.x;
  const double qy = static_cast<double>(pg.y) - c.y;
  const double qz = static_cast<double>(pg.z) - c.z;
  const int base = 9 * bi;
  AtomicAddDouble(&body_H[base + 0], p.x * qx);
  AtomicAddDouble(&body_H[base + 1], p.x * qy);
  AtomicAddDouble(&body_H[base + 2], p.x * qz);
  AtomicAddDouble(&body_H[base + 3], p.y * qx);
  AtomicAddDouble(&body_H[base + 4], p.y * qy);
  AtomicAddDouble(&body_H[base + 5], p.y * qz);
  AtomicAddDouble(&body_H[base + 6], p.z * qx);
  AtomicAddDouble(&body_H[base + 7], p.z * qy);
  AtomicAddDouble(&body_H[base + 8], p.z * qz);
}

__global__ void TransformSurfGlobalToLocalKernel(const int* surf_body_id,
                                                 const float3* surf_pos_global,
                                                 const float3* body_pos,
                                                 const float4* body_quat,
                                                 float3* surf_pos_local,
                                                 int n_surf_verts) {
  const int pid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (pid >= n_surf_verts)
    return;
  const int bi = surf_body_id[pid];
  const float3 p = surf_pos_global[pid];
  const float3 c = body_pos[bi];
  const float4 q = body_quat[bi];
  const float3 v = make_float3(p.x - c.x, p.y - c.y, p.z - c.z);
  const float4 qc = make_float4(-q.x, -q.y, -q.z, q.w);
  surf_pos_local[pid] = QuatRotate(qc, v);
}

__global__ void BuildTriRelPosKernel(const int3* tri_surf_vert_ids,
                                    const float3* surf_pos_local,
                                    float3* relpos_n1,
                                    float3* relpos_n2,
                                    float3* relpos_n3,
                                    int n_tris) {
  const int ti = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (ti >= n_tris)
    return;
  const int3 ids = tri_surf_vert_ids[ti];
  relpos_n1[ti] = surf_pos_local[ids.x];
  relpos_n2[ti] = surf_pos_local[ids.y];
  relpos_n3[ti] = surf_pos_local[ids.z];
}

__global__ void ComputeTriCenterVelFromRigidKernel(const int3* tri_surf_vert_ids,
                                                   const int* surf_body_id,
                                                   const float3* relpos_n1,
                                                   const float3* relpos_n2,
                                                   const float3* relpos_n3,
                                                   const float4* body_quat,
                                                   const float3* body_lin_vel,
                                                   const float3* body_omega,
                                                   float3* tri_vel_center,
                                                   int n_tris) {
  const int ti = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (ti >= n_tris)
    return;
  const int3 ids = tri_surf_vert_ids[ti];
  const int bi = surf_body_id[ids.x];
  const float3 p0 = relpos_n1[ti];
  const float3 p1 = relpos_n2[ti];
  const float3 p2 = relpos_n3[ti];
  const float3 c_local =
      make_float3((p0.x + p1.x + p2.x) / 3.f, (p0.y + p1.y + p2.y) / 3.f,
                  (p0.z + p1.z + p2.z) / 3.f);
  const float3 r = QuatRotate(body_quat[bi], c_local);
  const float3 v = body_lin_vel[bi];
  const float3 w = body_omega[bi];
  const float3 wxr = cross(w, r);
  tri_vel_center[ti] = make_float3(v.x + wxr.x, v.y + wxr.y, v.z + wxr.z);
}

__global__ void ScatterContactForcesKNNKernel(
    const float3* contact_points, const float3* contact_forces,
    const uint32_t* contact_owner, int n_contacts,
    const int* owner_to_body, int owner_to_body_size, const int* body_vert_start,
    const int* body_vert_count, const uint8_t* body_skip_forces,
    int n_bodies, const float3* surf_pos_global, const int* surf_global_node_id,
    int k, double force_scale, double clamp_force_norm, double damping_scale,
    double* d_f_contact) {
  const int ci = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (ci >= n_contacts)
    return;
  const uint32_t owner = contact_owner[ci];
  int bi = -1;
  if (owner < static_cast<uint32_t>(owner_to_body_size)) {
    bi = owner_to_body[owner];
  }
  if (bi < 0)
    return;
  if (body_skip_forces[bi])
    return;

  const double combined_scale = force_scale * damping_scale;
  const float3 p = contact_points[ci];
  const float3 f0 = contact_forces[ci];
  double fx = static_cast<double>(f0.x) * combined_scale;
  double fy = static_cast<double>(f0.y) * combined_scale;
  double fz = static_cast<double>(f0.z) * combined_scale;

  if (clamp_force_norm > 0.0) {
    const double fn = std::sqrt(fx * fx + fy * fy + fz * fz);
    if (fn > clamp_force_norm && fn > 0.0) {
      const double s = clamp_force_norm / fn;
      fx *= s;
      fy *= s;
      fz *= s;
    }
  }

  constexpr int kMax = 8;
  const int kk = (k < kMax) ? k : kMax;
  if (kk <= 0)
    return;

  int best_idx[kMax];
  double best_d2[kMax];
  for (int j = 0; j < kk; ++j) {
    best_idx[j] = -1;
    best_d2[j] = 1e300;
  }

  const int start = body_vert_start[bi];
  const int count = body_vert_count[bi];
  for (int off = 0; off < count; ++off) {
    const int pid = start + off;
    const float3 v = surf_pos_global[pid];
    const double dx = static_cast<double>(v.x) - static_cast<double>(p.x);
    const double dy = static_cast<double>(v.y) - static_cast<double>(p.y);
    const double dz = static_cast<double>(v.z) - static_cast<double>(p.z);
    const double d2 = dx * dx + dy * dy + dz * dz;
    int worst = 0;
    for (int j = 1; j < kk; ++j) {
      if (best_d2[j] > best_d2[worst])
        worst = j;
    }
    if (d2 >= best_d2[worst])
      continue;
    best_d2[worst] = d2;
    best_idx[worst] = pid;
  }

  double wsum = 0.0;
  double w[kMax];
  for (int j = 0; j < kk; ++j) {
    if (best_idx[j] < 0) {
      w[j] = 0.0;
      continue;
    }
    w[j] = 1.0 / (best_d2[j] + 1e-18);
    wsum += w[j];
  }
  if (wsum <= 0.0)
    return;

  for (int j = 0; j < kk; ++j) {
    const int pid = best_idx[j];
    if (pid < 0)
      continue;
    const int global = surf_global_node_id[pid];
    const double a = w[j] / wsum;
    AtomicAddDouble(&d_f_contact[3 * global + 0], a * fx);
    AtomicAddDouble(&d_f_contact[3 * global + 1], a * fy);
    AtomicAddDouble(&d_f_contact[3 * global + 2], a * fz);
  }
}

__global__ void ComputeTriCenterVelKernel(const int3* tri_global_nodes,
                                          const double* d_vel_xyz,
                                          float3* tri_vel_center, int n_tris) {
  const int ti = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (ti >= n_tris)
    return;
  const int3 ids = tri_global_nodes[ti];
  const int i0   = ids.x;
  const int i1   = ids.y;
  const int i2   = ids.z;
  const double vx =
      (d_vel_xyz[3 * i0 + 0] + d_vel_xyz[3 * i1 + 0] + d_vel_xyz[3 * i2 + 0]) /
      3.0;
  const double vy =
      (d_vel_xyz[3 * i0 + 1] + d_vel_xyz[3 * i1 + 1] + d_vel_xyz[3 * i2 + 1]) /
      3.0;
  const double vz =
      (d_vel_xyz[3 * i0 + 2] + d_vel_xyz[3 * i1 + 2] + d_vel_xyz[3 * i2 + 2]) /
      3.0;
  tri_vel_center[ti] =
      make_float3(static_cast<float>(vx), static_cast<float>(vy),
                  static_cast<float>(vz));
}

}  // namespace

DemeMeshCollisionSystem::DemeMeshCollisionSystem(
    std::vector<DemeMeshCollisionBody> bodies, double friction,
    double stiffness, double restitution, bool enable_self_collision)
    : enable_self_collision_(enable_self_collision) {
  if (bodies.empty()) {
    throw std::invalid_argument(
        "DemeMeshCollisionSystem: bodies must be non-empty");
  }

  bodies_.reserve(bodies.size());
  size_t tri_start = 0;
  size_t surf_vert_start = 0;
  for (auto& body : bodies) {
    const auto& surf = body.surface;
    if (surf.vertices.size() != surf.global_node_ids.size()) {
      throw std::invalid_argument(
          "DemeMeshCollisionSystem: SurfaceTriMesh vertices size must match "
          "global_node_ids size");
    }
    if (surf.vertices.empty() || surf.triangles.empty()) {
      throw std::invalid_argument(
          "DemeMeshCollisionSystem: each body must have non-empty surface "
          "vertices and triangles");
    }
    const int n_verts = static_cast<int>(surf.vertices.size());
    for (const auto& tri : surf.triangles) {
      if (tri.x() < 0 || tri.y() < 0 || tri.z() < 0 || tri.x() >= n_verts ||
          tri.y() >= n_verts || tri.z() >= n_verts) {
        throw std::invalid_argument(
            "DemeMeshCollisionSystem: SurfaceTriMesh triangle indices out of "
            "range");
      }
    }

    RuntimeBody rb;
    rb.body      = std::move(body);
    rb.tri_start = tri_start;
    rb.tri_count = static_cast<int>(rb.body.surface.triangles.size());
    rb.surf_vert_start = static_cast<int>(surf_vert_start);
    rb.surf_vert_count = static_cast<int>(rb.body.surface.vertices.size());
    tri_start += rb.body.surface.triangles.size();
    surf_vert_start += rb.body.surface.vertices.size();
    bodies_.push_back(std::move(rb));
  }

  BuildSolver(friction, stiffness, restitution);

  // Build a triangle -> global node ID mapping (used to compute per-triangle
  // velocity estimates on GPU for DEME's patch-based friction/damping).
  n_tris_ = static_cast<int>(tri_start);
  if (n_tris_ > 0) {
    std::vector<int3> h_tri_global_nodes(static_cast<size_t>(n_tris_));
    for (const auto& rb : bodies_) {
      const auto& surf = rb.body.surface;
      for (size_t ti = 0; ti < surf.triangles.size(); ++ti) {
        const auto& tri = surf.triangles[ti];
        const int g0 = surf.global_node_ids[static_cast<size_t>(tri.x())];
        const int g1 = surf.global_node_ids[static_cast<size_t>(tri.y())];
        const int g2 = surf.global_node_ids[static_cast<size_t>(tri.z())];
        h_tri_global_nodes[rb.tri_start + ti] = make_int3(g0, g1, g2);
      }
    }
    HANDLE_ERROR(cudaMalloc(&d_tri_global_nodes_,
                            static_cast<size_t>(n_tris_) * sizeof(int3)));
    HANDLE_ERROR(cudaMemcpy(d_tri_global_nodes_, h_tri_global_nodes.data(),
                            static_cast<size_t>(n_tris_) * sizeof(int3),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMalloc(&d_tri_vel_center_,
                            static_cast<size_t>(n_tris_) * sizeof(float3)));
    HANDLE_ERROR(cudaMemset(d_tri_vel_center_, 0,
                            static_cast<size_t>(n_tris_) * sizeof(float3)));
  }

  n_surf_verts_ = 0;
  for (const auto& rb : bodies_) {
    n_surf_verts_ += rb.surf_vert_count;
  }

  n_bodies_ = static_cast<int>(bodies_.size());
  if (n_bodies_ > 0) {
    std::vector<uint32_t> h_body_owner_id(static_cast<size_t>(n_bodies_));
    std::vector<int> h_body_vert_start(static_cast<size_t>(n_bodies_));
    std::vector<int> h_body_vert_count(static_cast<size_t>(n_bodies_));
    std::vector<double> h_body_inv_vert_count(static_cast<size_t>(n_bodies_));
    std::vector<uint8_t> h_body_skip_forces(static_cast<size_t>(n_bodies_));
    std::vector<float3> h_body_pos_f(static_cast<size_t>(n_bodies_));
    std::vector<float4> h_body_quat_f(static_cast<size_t>(n_bodies_));
    std::vector<float3> h_body_lin_vel_f(static_cast<size_t>(n_bodies_));
    std::vector<float3> h_body_omega_f(static_cast<size_t>(n_bodies_));

    for (int bi = 0; bi < n_bodies_; ++bi) {
      const auto& rb = bodies_[static_cast<size_t>(bi)];
      h_body_owner_id[static_cast<size_t>(bi)] = static_cast<uint32_t>(rb.owner);
      h_body_vert_start[static_cast<size_t>(bi)] = rb.surf_vert_start;
      h_body_vert_count[static_cast<size_t>(bi)] = rb.surf_vert_count;
      h_body_inv_vert_count[static_cast<size_t>(bi)] =
          (rb.surf_vert_count > 0) ? (1.0 / static_cast<double>(rb.surf_vert_count)) : 0.0;
      h_body_skip_forces[static_cast<size_t>(bi)] = rb.body.skip_self_contact_forces ? 1u : 0u;
      h_body_pos_f[static_cast<size_t>(bi)] = ToFloat3(rb.prev_pos);
      h_body_quat_f[static_cast<size_t>(bi)] = IdentityQuat();
      h_body_lin_vel_f[static_cast<size_t>(bi)] = make_float3(0.f, 0.f, 0.f);
      h_body_omega_f[static_cast<size_t>(bi)] = make_float3(0.f, 0.f, 0.f);
    }

    HANDLE_ERROR(cudaMalloc(&d_body_owner_id_, static_cast<size_t>(n_bodies_) * sizeof(uint32_t)));
    HANDLE_ERROR(cudaMalloc(&d_body_vert_start_, static_cast<size_t>(n_bodies_) * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&d_body_vert_count_, static_cast<size_t>(n_bodies_) * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&d_body_inv_vert_count_, static_cast<size_t>(n_bodies_) * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_body_skip_forces_, static_cast<size_t>(n_bodies_) * sizeof(uint8_t)));
    HANDLE_ERROR(cudaMalloc(&d_body_pos_f_, static_cast<size_t>(n_bodies_) * sizeof(float3)));
    HANDLE_ERROR(cudaMalloc(&d_body_quat_f_, static_cast<size_t>(n_bodies_) * sizeof(float4)));
    HANDLE_ERROR(cudaMalloc(&d_body_lin_vel_f_, static_cast<size_t>(n_bodies_) * sizeof(float3)));
    HANDLE_ERROR(cudaMalloc(&d_body_omega_f_, static_cast<size_t>(n_bodies_) * sizeof(float3)));

    HANDLE_ERROR(cudaMemcpy(d_body_owner_id_, h_body_owner_id.data(),
                            static_cast<size_t>(n_bodies_) * sizeof(uint32_t),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_body_vert_start_, h_body_vert_start.data(),
                            static_cast<size_t>(n_bodies_) * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_body_vert_count_, h_body_vert_count.data(),
                            static_cast<size_t>(n_bodies_) * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_body_inv_vert_count_, h_body_inv_vert_count.data(),
                            static_cast<size_t>(n_bodies_) * sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_body_skip_forces_, h_body_skip_forces.data(),
                            static_cast<size_t>(n_bodies_) * sizeof(uint8_t),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_body_pos_f_, h_body_pos_f.data(),
                            static_cast<size_t>(n_bodies_) * sizeof(float3), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_body_quat_f_, h_body_quat_f.data(),
                            static_cast<size_t>(n_bodies_) * sizeof(float4), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_body_lin_vel_f_, h_body_lin_vel_f.data(),
                            static_cast<size_t>(n_bodies_) * sizeof(float3), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_body_omega_f_, h_body_omega_f.data(),
                            static_cast<size_t>(n_bodies_) * sizeof(float3), cudaMemcpyHostToDevice));

    const int n_owners = static_cast<int>(solver_->GetNumOwners());
    uint32_t max_owner = 0;
    for (int bi = 0; bi < n_bodies_; ++bi) {
      max_owner = std::max(max_owner, h_body_owner_id[static_cast<size_t>(bi)]);
    }
    owner_to_body_size_ = std::max(n_owners, static_cast<int>(max_owner) + 1);
    if (owner_to_body_size_ > 0) {
      std::vector<int> h_owner_to_body(static_cast<size_t>(owner_to_body_size_), -1);
      for (int bi = 0; bi < n_bodies_; ++bi) {
        const uint32_t owner = h_body_owner_id[static_cast<size_t>(bi)];
        if (owner < static_cast<uint32_t>(owner_to_body_size_)) {
          h_owner_to_body[static_cast<size_t>(owner)] = bi;
        }
      }

      owners_dense_ = (n_owners == n_bodies_) &&
                      (max_owner + 1u == static_cast<uint32_t>(n_owners));
      if (owners_dense_) {
        for (int owner = 0; owner < n_owners; ++owner) {
          if (h_owner_to_body[static_cast<size_t>(owner)] < 0) {
            owners_dense_ = false;
            break;
          }
        }
      }

      if (d_owner_to_body_ != nullptr) {
        cudaFree(d_owner_to_body_);
        d_owner_to_body_ = nullptr;
      }
      HANDLE_ERROR(cudaMalloc(
          &d_owner_to_body_,
          static_cast<size_t>(owner_to_body_size_) * sizeof(int)));
      HANDLE_ERROR(cudaMemcpy(
          d_owner_to_body_, h_owner_to_body.data(),
          static_cast<size_t>(owner_to_body_size_) * sizeof(int),
          cudaMemcpyHostToDevice));
    }
  }

  if (n_surf_verts_ > 0) {
    std::vector<int> h_surf_global_node_id(static_cast<size_t>(n_surf_verts_));
    std::vector<int> h_surf_body_id(static_cast<size_t>(n_surf_verts_));
    std::vector<double3> h_surf_ref_local(static_cast<size_t>(n_surf_verts_));

    for (size_t bi = 0; bi < bodies_.size(); ++bi) {
      const auto& rb   = bodies_[bi];
      const auto& surf = rb.body.surface;
      for (int i = 0; i < rb.surf_vert_count; ++i) {
        const int pid = rb.surf_vert_start + i;
        h_surf_global_node_id[static_cast<size_t>(pid)] =
            surf.global_node_ids[static_cast<size_t>(i)];
        h_surf_body_id[static_cast<size_t>(pid)] = static_cast<int>(bi);
        const Eigen::Vector3d& p = rb.ref_vertices_local[static_cast<size_t>(i)];
        h_surf_ref_local[static_cast<size_t>(pid)] =
            make_double3(p.x(), p.y(), p.z());
      }
    }

    HANDLE_ERROR(cudaMalloc(&d_surf_global_node_id_,
                            static_cast<size_t>(n_surf_verts_) * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&d_surf_body_id_,
                            static_cast<size_t>(n_surf_verts_) * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&d_surf_ref_local_,
                            static_cast<size_t>(n_surf_verts_) * sizeof(double3)));
    HANDLE_ERROR(cudaMalloc(&d_surf_pos_global_,
                            static_cast<size_t>(n_surf_verts_) * sizeof(float3)));
    HANDLE_ERROR(cudaMalloc(&d_surf_pos_local_,
                            static_cast<size_t>(n_surf_verts_) * sizeof(float3)));

    HANDLE_ERROR(cudaMemcpy(d_surf_global_node_id_, h_surf_global_node_id.data(),
                            static_cast<size_t>(n_surf_verts_) * sizeof(int),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_surf_body_id_, h_surf_body_id.data(),
                            static_cast<size_t>(n_surf_verts_) * sizeof(int),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_surf_ref_local_, h_surf_ref_local.data(),
                            static_cast<size_t>(n_surf_verts_) * sizeof(double3),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemset(d_surf_pos_global_, 0,
                            static_cast<size_t>(n_surf_verts_) * sizeof(float3)));
    HANDLE_ERROR(cudaMemset(d_surf_pos_local_, 0,
                            static_cast<size_t>(n_surf_verts_) * sizeof(float3)));
  }

  if (n_tris_ > 0 && n_surf_verts_ > 0) {
    std::vector<int3> h_tri_surf_vert_ids(static_cast<size_t>(n_tris_));
    for (const auto& rb : bodies_) {
      const auto& surf = rb.body.surface;
      for (size_t ti = 0; ti < surf.triangles.size(); ++ti) {
        const auto& tri = surf.triangles[ti];
        const int p0 = rb.surf_vert_start + static_cast<int>(tri.x());
        const int p1 = rb.surf_vert_start + static_cast<int>(tri.y());
        const int p2 = rb.surf_vert_start + static_cast<int>(tri.z());
        h_tri_surf_vert_ids[rb.tri_start + ti] = make_int3(p0, p1, p2);
      }
    }

    HANDLE_ERROR(cudaMalloc(&d_tri_surf_vert_ids_,
                            static_cast<size_t>(n_tris_) * sizeof(int3)));
    HANDLE_ERROR(cudaMemcpy(d_tri_surf_vert_ids_, h_tri_surf_vert_ids.data(),
                            static_cast<size_t>(n_tris_) * sizeof(int3),
                            cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc(&d_tri_relpos_n1_,
                            static_cast<size_t>(n_tris_) * sizeof(float3)));
    HANDLE_ERROR(cudaMalloc(&d_tri_relpos_n2_,
                            static_cast<size_t>(n_tris_) * sizeof(float3)));
    HANDLE_ERROR(cudaMalloc(&d_tri_relpos_n3_,
                            static_cast<size_t>(n_tris_) * sizeof(float3)));
    HANDLE_ERROR(cudaMemset(d_tri_relpos_n1_, 0,
                            static_cast<size_t>(n_tris_) * sizeof(float3)));
    HANDLE_ERROR(cudaMemset(d_tri_relpos_n2_, 0,
                            static_cast<size_t>(n_tris_) * sizeof(float3)));
    HANDLE_ERROR(cudaMemset(d_tri_relpos_n3_, 0,
                            static_cast<size_t>(n_tris_) * sizeof(float3)));
  }

  if (!bodies_.empty()) {
    const size_t nb = bodies_.size();
    HANDLE_ERROR(cudaMalloc(&d_body_com_, nb * sizeof(double3)));
    HANDLE_ERROR(cudaMalloc(&d_body_H_, nb * 9 * sizeof(double)));
    HANDLE_ERROR(cudaMemset(d_body_com_, 0, nb * sizeof(double3)));
    HANDLE_ERROR(cudaMemset(d_body_H_, 0, nb * 9 * sizeof(double)));
  }
}

DemeMeshCollisionSystem::~DemeMeshCollisionSystem() {
  if (d_f_contact_ != nullptr) {
    cudaFree(d_f_contact_);
    d_f_contact_ = nullptr;
  }
  if (d_owner_to_body_ != nullptr) {
    cudaFree(d_owner_to_body_);
    d_owner_to_body_ = nullptr;
  }
  if (d_tri_global_nodes_ != nullptr) {
    cudaFree(d_tri_global_nodes_);
    d_tri_global_nodes_ = nullptr;
  }
  if (d_tri_vel_center_ != nullptr) {
    cudaFree(d_tri_vel_center_);
    d_tri_vel_center_ = nullptr;
  }
  if (d_surf_global_node_id_ != nullptr) {
    cudaFree(d_surf_global_node_id_);
    d_surf_global_node_id_ = nullptr;
  }
  if (d_surf_body_id_ != nullptr) {
    cudaFree(d_surf_body_id_);
    d_surf_body_id_ = nullptr;
  }
  if (d_surf_ref_local_ != nullptr) {
    cudaFree(d_surf_ref_local_);
    d_surf_ref_local_ = nullptr;
  }
  if (d_surf_pos_global_ != nullptr) {
    cudaFree(d_surf_pos_global_);
    d_surf_pos_global_ = nullptr;
  }
  if (d_surf_pos_local_ != nullptr) {
    cudaFree(d_surf_pos_local_);
    d_surf_pos_local_ = nullptr;
  }
  if (d_body_owner_id_ != nullptr) {
    cudaFree(d_body_owner_id_);
    d_body_owner_id_ = nullptr;
  }
  if (d_body_vert_start_ != nullptr) {
    cudaFree(d_body_vert_start_);
    d_body_vert_start_ = nullptr;
  }
  if (d_body_vert_count_ != nullptr) {
    cudaFree(d_body_vert_count_);
    d_body_vert_count_ = nullptr;
  }
  if (d_body_inv_vert_count_ != nullptr) {
    cudaFree(d_body_inv_vert_count_);
    d_body_inv_vert_count_ = nullptr;
  }
  if (d_body_skip_forces_ != nullptr) {
    cudaFree(d_body_skip_forces_);
    d_body_skip_forces_ = nullptr;
  }
  if (d_body_pos_f_ != nullptr) {
    cudaFree(d_body_pos_f_);
    d_body_pos_f_ = nullptr;
  }
  if (d_body_quat_f_ != nullptr) {
    cudaFree(d_body_quat_f_);
    d_body_quat_f_ = nullptr;
  }
  if (d_body_lin_vel_f_ != nullptr) {
    cudaFree(d_body_lin_vel_f_);
    d_body_lin_vel_f_ = nullptr;
  }
  if (d_body_omega_f_ != nullptr) {
    cudaFree(d_body_omega_f_);
    d_body_omega_f_ = nullptr;
  }
  if (d_body_com_ != nullptr) {
    cudaFree(d_body_com_);
    d_body_com_ = nullptr;
  }
  if (d_body_H_ != nullptr) {
    cudaFree(d_body_H_);
    d_body_H_ = nullptr;
  }
  if (d_tri_surf_vert_ids_ != nullptr) {
    cudaFree(d_tri_surf_vert_ids_);
    d_tri_surf_vert_ids_ = nullptr;
  }
  if (d_tri_relpos_n1_ != nullptr) {
    cudaFree(d_tri_relpos_n1_);
    d_tri_relpos_n1_ = nullptr;
  }
  if (d_tri_relpos_n2_ != nullptr) {
    cudaFree(d_tri_relpos_n2_);
    d_tri_relpos_n2_ = nullptr;
  }
  if (d_tri_relpos_n3_ != nullptr) {
    cudaFree(d_tri_relpos_n3_);
    d_tri_relpos_n3_ = nullptr;
  }
  if (d_contact_points_ != nullptr) {
    cudaFree(d_contact_points_);
    d_contact_points_ = nullptr;
  }
  if (d_contact_forces_ != nullptr) {
    cudaFree(d_contact_forces_);
    d_contact_forces_ = nullptr;
  }
  if (d_contact_owner_ != nullptr) {
    cudaFree(d_contact_owner_);
    d_contact_owner_ = nullptr;
  }
}

void DemeMeshCollisionSystem::BuildSolver(double friction, double stiffness,
                                          double restitution) {
  ConfigureDemeRuntimePathsFromBazelBin();

  solver_ = std::make_unique<deme::DEMSolver>(1);
  solver_->SetVerbosity("ERROR");

  Eigen::Vector3d mn(1e30, 1e30, 1e30);
  Eigen::Vector3d mx(-1e30, -1e30, -1e30);
  for (const auto& rb : bodies_) {
    for (const auto& v : rb.body.surface.vertices)
      ExtendBBox(mn, mx, v);
  }

  const double pad = 1.0;
  solver_->InstructBoxDomainDimension(
      {static_cast<float>(mn.x() - pad), static_cast<float>(mx.x() + pad)},
      {static_cast<float>(mn.y() - pad), static_cast<float>(mx.y() + pad)},
      {static_cast<float>(mn.z() - pad), static_cast<float>(mx.z() + pad)});

  solver_->SetGravitationalAcceleration(make_float3(0.f, 0.f, 0.f));
  solver_->SetMeshUniversalContact(true);

  // Contact stiffness tuning:
  // - DEM-Engine uses a Hertzian-style contact model which can easily produce
  //   forces that are too stiff for our implicit FE solve if E is large.
  // - Use the provided stiffness parameter; env var `DEME_CONTACT_E` overrides.
  const float default_E   = static_cast<float>(stiffness);
  const float contact_E   = EnvFloatOr("DEME_CONTACT_E", default_E);
  const float contact_nu  = EnvFloatOr("DEME_CONTACT_NU", 0.3f);
  const float default_cor = static_cast<float>(std::clamp(restitution, 0.0, 1.0));
  const float contact_cor = EnvFloatOr("DEME_CONTACT_COR", default_cor);

  const float mu = static_cast<float>(std::max(0.0, friction));
  auto mat       = solver_->LoadMaterial({{"E", contact_E},
                                          {"nu", contact_nu},
                                          {"CoR", contact_cor},
                                          {"mu", mu},
                                          {"Crr", 0.0f}});

  auto make_mesh = [&](const ANCFCPUUtils::SurfaceTriMesh& surf,
                       unsigned int family, bool split_into_patches,
                       float patch_angle_deg, const Eigen::Vector3d& centroid,
                       const std::vector<float3>& vertices_local,
                       float body_mass, const char* label) {
    deme::DEMMesh mesh;
    mesh.SetFamily(family);

    // DEME expects mesh vertices in the owner's local frame. Keep the owner
    // init position meaningful by centering vertices at the initial centroid.
    mesh.SetInitPos(ToFloat3(centroid));

    mesh.m_vertices = vertices_local;

    mesh.m_face_v_indices.reserve(surf.triangles.size());
    for (const auto& tri : surf.triangles) {
      mesh.m_face_v_indices.push_back(make_int3(tri.x(), tri.y(), tri.z()));
    }
    mesh.nTri = surf.triangles.size();
    // Default: treat the whole mesh as a single patch.
    mesh.SetPatchIDs(std::vector<deme::patchID_t>(mesh.nTri, 0));
    if (split_into_patches) {
      // Smaller angle => more patches => more mesh-mesh contact points (patch
      // pairs). For highly concave meshes, this is important: DEME's
      // patch-level contact can otherwise collapse to ~one contact point.
      const float angle = patch_angle_deg;
      if (angle > 0.0f && angle < 360.0f) {
        // NOTE: patchID_t is int16; keep patch count in-range.
        constexpr unsigned int kMaxPatches =
            static_cast<unsigned int>(
                std::numeric_limits<deme::patchID_t>::max()) +
            1u;

        float try_angle        = angle;
        unsigned int n_patches = 0;
        for (int attempt = 0; attempt < 6; ++attempt) {
          n_patches = mesh.SplitIntoConvexPatches(try_angle);
          if (n_patches > 0 && n_patches <= kMaxPatches)
            break;
          if (try_angle >= 180.0f)
            break;
          try_angle = std::min(180.0f, try_angle * 1.5f);
        }

        if (n_patches == 0) {
          std::cout << "[DEME] Patch splitting produced 0 patches for " << label
                    << " (angle_threshold_deg=" << try_angle
                    << "); keeping single patch\n";
          mesh.SetPatchIDs(std::vector<deme::patchID_t>(mesh.nTri, 0));
        } else if (n_patches > kMaxPatches) {
          std::cout << "[DEME] Patch splitting produced too many patches for "
                    << label << " (" << n_patches << " > " << kMaxPatches
                    << "); keeping single patch\n";
          mesh.SetPatchIDs(std::vector<deme::patchID_t>(mesh.nTri, 0));
        } else {
          std::cout << "[DEME] Split " << label << " into " << n_patches
                    << " patches (angle_threshold_deg=" << try_angle << ")\n";
        }
      } else {
        std::cout << "[DEME] Patch splitting disabled for " << label
                  << " (angle_threshold_deg=" << angle << ")\n";
      }
    }

    // IMPORTANT: DEME's SplitIntoConvexPatches currently errors if material was
    // set before splitting (it checks materials.size() != nPatches before
    // broadcasting). Set material after patch IDs are finalized.
    mesh.SetMaterial(mat);

    mesh.SetInitQuat(IdentityQuat());
    mesh.SetMass(body_mass);
    // Approximate MOI as uniform sphere with equivalent mass (placeholder)
    const float moi = body_mass * 0.4f;
    mesh.SetMOI(make_float3(moi, moi, moi));

    return mesh;
  };

  // Patch splitting controls (env override): `DEME_PATCH_ANGLE_DEG`.
  // A medium-aggressive default (20 deg) works better for concave meshes.
  // Back-compat alias: `DEME_ITEM_PATCH_ANGLE_DEG`.
  const float default_patch_angle_deg = EnvFloatOrFallback(
      "DEME_PATCH_ANGLE_DEG", "DEME_ITEM_PATCH_ANGLE_DEG", 20.0f);

  // Family usage is global in DEME; self-collision can only be toggled
  // per-family.
  std::unordered_map<unsigned int, int> family_counts;
  for (const auto& rb : bodies_)
    family_counts[rb.body.family] += 1;

  std::vector<unsigned int> unique_families;
  unique_families.reserve(family_counts.size());
  for (const auto& it : family_counts)
    unique_families.push_back(it.first);

  if (!enable_self_collision_) {
    for (const auto& it : family_counts) {
      const unsigned int fam = it.first;
      const int count        = it.second;
      if (count == 1) {
        solver_->DisableContactBetweenFamilies(fam, fam);
      } else {
        std::cout
            << "[DEME] Family " << fam << " is shared by " << count
            << " bodies; cannot disable self-collision without also disabling "
               "cross-body contacts in that family. Leaving self-contact "
               "enabled.\n";
      }
    }
  }

  for (size_t bi = 0; bi < bodies_.size(); ++bi) {
    auto& rb          = bodies_[bi];
    const auto& body  = rb.body;
    const float angle = (body.patch_angle_deg > 0.0f) ? body.patch_angle_deg
                                                      : default_patch_angle_deg;
    const float patch_angle = body.split_into_patches ? angle : 0.0f;
    const std::string label = "mesh[" + std::to_string(bi) + "]";

    const Eigen::Vector3d centroid = ComputeCentroid(body.surface.vertices);
    rb.ref_vertices_local.clear();
    rb.ref_vertices_local.reserve(body.surface.vertices.size());
    std::vector<float3> vertices_local;
    vertices_local.reserve(body.surface.vertices.size());
    for (const auto& v : body.surface.vertices) {
      const Eigen::Vector3d local = v - centroid;
      rb.ref_vertices_local.push_back(local);
      vertices_local.push_back(ToFloat3(local));
    }
    rb.prev_pos  = centroid;
    rb.prev_quat = Eigen::Quaterniond::Identity();

    auto mesh = make_mesh(body.surface, body.family, body.split_into_patches,
                          patch_angle, centroid, vertices_local, body.mass,
                          label.c_str());
    rb.mesh_handle = solver_->AddMesh(mesh);
  }

  // Keep all mesh families fully prescribed (we will update their nodes
  // directly).
  for (const unsigned int fam : unique_families) {
    solver_->SetFamilyPrescribedPosition(fam);
    solver_->SetFamilyPrescribedQuaternion(fam);
    solver_->SetFamilyPrescribedLinVel(fam);
    solver_->SetFamilyPrescribedAngVel(fam);
  }

  // For co-simulation-style "externally deformed" meshes, update every step.
  solver_->SetCDUpdateFreq(1);
  solver_->DisableAdaptiveUpdateFreq();

  // Use externally provided relative velocity (from FE nodal velocities) in
  // DEME's patch-based mesh contact damping/friction.
  solver_->SetUsePatchRelativeVelocityOverride(true);

  solver_->SetInitTimeStep(1e-4);
  solver_->Initialize();

  // DEME assigns mesh owner IDs during initialization. Cache them now so we
  // don't accidentally use `NULL_BODYID` (which would segfault in DEME
  // setters).
  for (auto& rb : bodies_) {
    if (!rb.mesh_handle) {
      throw std::runtime_error(
          "DemeMeshCollisionSystem: mesh handle is null after AddMesh");
    }
    if (rb.mesh_handle->owner == deme::NULL_BODYID) {
      throw std::runtime_error(
          "DemeMeshCollisionSystem: DEME mesh owner ID was not assigned after "
          "Initialize()");
    }
    rb.owner = static_cast<unsigned int>(rb.mesh_handle->owner);
  }
}

void DemeMeshCollisionSystem::BindNodesDevicePtr(double* d_nodes_xyz,
                                                 int n_nodes) {
  if (d_nodes_xyz == nullptr) {
    throw std::invalid_argument("BindNodesDevicePtr: d_nodes_xyz is null");
  }
  if (n_nodes <= 0) {
    throw std::invalid_argument("BindNodesDevicePtr: n_nodes must be > 0");
  }
  d_nodes_xyz_ = d_nodes_xyz;
  n_nodes_     = n_nodes;

  h_nodes_xyz_.resize(static_cast<size_t>(3 * n_nodes_));
  h_f_contact_.resize(static_cast<size_t>(3 * n_nodes_));
  h_tri_vel_center_.resize(static_cast<size_t>(n_tris_));
  h_body_com_.resize(static_cast<size_t>(bodies_.size()));
  h_body_H_.resize(static_cast<size_t>(9 * bodies_.size()));

  h_body_pos_f_.resize(static_cast<size_t>(bodies_.size()));
  h_body_quat_f_.resize(static_cast<size_t>(bodies_.size()));
  h_body_lin_vel_f_.resize(static_cast<size_t>(bodies_.size()));
  h_body_omega_f_.resize(static_cast<size_t>(bodies_.size()));

  if (owners_dense_ && owner_to_body_size_ > 0) {
    h_owner_pos_.resize(static_cast<size_t>(owner_to_body_size_));
    h_owner_quat_.resize(static_cast<size_t>(owner_to_body_size_));
    h_owner_lin_vel_.resize(static_cast<size_t>(owner_to_body_size_));
    h_owner_omega_.resize(static_cast<size_t>(owner_to_body_size_));
  }

  // Initialize velocity tracking: one COM position per body
  first_step_ = true;

  if (d_f_contact_ == nullptr) {
    HANDLE_ERROR(cudaMalloc(
        &d_f_contact_, static_cast<size_t>(3 * n_nodes_) * sizeof(double)));
  }
}

void DemeMeshCollisionSystem::Step(const CollisionSystemInput& in,
                                   const CollisionSystemParams& params) {
  if (d_nodes_xyz_ == nullptr || n_nodes_ <= 0) {
    throw std::runtime_error(
        "DemeMeshCollisionSystem::Step called before BindNodesDevicePtr");
  }
  if (in.dt <= 0.0) {
    throw std::invalid_argument(
        "DemeMeshCollisionSystem::Step: dt must be > 0");
  }
  const bool have_fea_vel = (in.d_vel_xyz != nullptr);

  constexpr int kThreads = 256;
  const int nb = n_bodies_;
  const bool was_first_step = first_step_;

  if (n_surf_verts_ > 0) {
    const int blocks = (n_surf_verts_ + kThreads - 1) / kThreads;
    GatherSurfPosGlobalKernel<<<blocks, kThreads>>>(
        d_nodes_xyz_, n_nodes_, d_surf_global_node_id_, d_surf_pos_global_,
        n_surf_verts_);
    HANDLE_ERROR(cudaGetLastError());
  }

  if (nb > 0) {
    HANDLE_ERROR(cudaMemset(d_body_com_, 0,
                            static_cast<size_t>(nb) * sizeof(double3)));
    HANDLE_ERROR(cudaMemset(d_body_H_, 0,
                            static_cast<size_t>(9 * nb) * sizeof(double)));
  }

  if (n_surf_verts_ > 0 && nb > 0) {
    const int blocks = (n_surf_verts_ + kThreads - 1) / kThreads;
    AccumulateBodyCOMKernel<<<blocks, kThreads>>>(d_surf_body_id_,
                                                 d_surf_pos_global_, d_body_com_,
                                                 n_surf_verts_);
    HANDLE_ERROR(cudaGetLastError());
    const int bblocks = (nb + kThreads - 1) / kThreads;
    NormalizeBodyCOMKernel<<<bblocks, kThreads>>>(d_body_com_,
                                                 d_body_inv_vert_count_, nb);
    HANDLE_ERROR(cudaGetLastError());
    AccumulateBodyHKernel<<<blocks, kThreads>>>(d_surf_body_id_,
                                               d_surf_ref_local_,
                                               d_surf_pos_global_, d_body_com_,
                                               d_body_H_, n_surf_verts_);
    HANDLE_ERROR(cudaGetLastError());
  }

  if (nb > 0) {
    HANDLE_ERROR(cudaMemcpy(h_body_com_.data(), d_body_com_,
                            static_cast<size_t>(nb) * sizeof(double3),
                            cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_body_H_.data(), d_body_H_,
                            static_cast<size_t>(9 * nb) * sizeof(double),
                            cudaMemcpyDeviceToHost));
  }

  if (h_body_pos_f_.size() != static_cast<size_t>(nb)) {
    h_body_pos_f_.resize(static_cast<size_t>(nb));
    h_body_quat_f_.resize(static_cast<size_t>(nb));
    h_body_lin_vel_f_.resize(static_cast<size_t>(nb));
    h_body_omega_f_.resize(static_cast<size_t>(nb));
  }

  if (owners_dense_ && owner_to_body_size_ > 0) {
    if (h_owner_pos_.size() != static_cast<size_t>(owner_to_body_size_)) {
      h_owner_pos_.resize(static_cast<size_t>(owner_to_body_size_));
      h_owner_quat_.resize(static_cast<size_t>(owner_to_body_size_));
      h_owner_lin_vel_.resize(static_cast<size_t>(owner_to_body_size_));
      h_owner_omega_.resize(static_cast<size_t>(owner_to_body_size_));
    }
  }

  // DEME expects triangle node positions in the mesh owner's local frame, and
  // uses owner position + patch locations for contact detection. Keep owner
  // poses (pos+orientation) updated, and send local vertex positions in that
  // owner frame. This keeps patch locations consistent and avoids encoding
  // rigid-body rotations as "mesh deformation" (which breaks patch contact and
  // friction).
  std::vector<float3> one_pos;
  std::vector<float4> one_quat;
  std::vector<float3> one_vel;
  std::vector<float3> one_omega;
  if (!owners_dense_) {
    one_pos.resize(1);
    one_quat.resize(1);
    one_vel.resize(1);
    one_omega.resize(1);
  }

  for (size_t bi = 0; bi < bodies_.size(); ++bi) {
    auto& rb = bodies_[bi];

    const double3 com_d = h_body_com_[bi];
    const float3 com_f = make_float3(static_cast<float>(com_d.x),
                                     static_cast<float>(com_d.y),
                                     static_cast<float>(com_d.z));
    const Eigen::Vector3d com(static_cast<double>(com_d.x),
                              static_cast<double>(com_d.y),
                              static_cast<double>(com_d.z));
    if (rb.owner == static_cast<unsigned int>(deme::NULL_BODYID)) {
      throw std::runtime_error(
          "DemeMeshCollisionSystem::Step: DEME mesh owner ID is NULL_BODYID");
    }

    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    const double* hH = h_body_H_.data() + 9 * static_cast<int>(bi);
    H(0, 0) = hH[0];
    H(0, 1) = hH[1];
    H(0, 2) = hH[2];
    H(1, 0) = hH[3];
    H(1, 1) = hH[4];
    H(1, 2) = hH[5];
    H(2, 0) = hH[6];
    H(2, 1) = hH[7];
    H(2, 2) = hH[8];

    const Eigen::Matrix3d R = BestFitRotationFromH(H);
    Eigen::Quaterniond quat(R);
    quat.normalize();

    const float4 quat_f = ToQuatXYZW(quat);
    h_body_pos_f_[bi] = com_f;
    h_body_quat_f_[bi] = quat_f;
    if (owners_dense_ && rb.owner < static_cast<unsigned int>(owner_to_body_size_)) {
      h_owner_pos_[static_cast<size_t>(rb.owner)] = com_f;
      h_owner_quat_[static_cast<size_t>(rb.owner)] = quat_f;
    } else if (!owners_dense_) {
      one_pos[0] = com_f;
      one_quat[0] = quat_f;
      solver_->SetOwnerPosition(rb.owner, one_pos);
      solver_->SetOwnerOriQ(rb.owner, one_quat);
    }

    // Compute and update velocity for DEME friction computation.
    // DEME's Hertzian friction model uses relative tangential velocity to
    // determine friction force direction. Without velocity updates, DEME sees
    // all bodies as stationary and friction is not computed correctly.
    Eigen::Vector3d lin_vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d omega   = Eigen::Vector3d::Zero();
    if (!first_step_) {
      const double inv_dt = 1.0 / in.dt;
      lin_vel = (com - rb.prev_pos) * inv_dt;

      Eigen::Quaterniond q_rel = quat * rb.prev_quat.conjugate();
      if (q_rel.w() < 0.0) {
        q_rel.coeffs() *= -1.0;
      }
      Eigen::AngleAxisd aa(q_rel);
      const double ang = aa.angle();
      if (std::isfinite(ang) && ang > 1e-10) {
        omega = aa.axis() * (ang * inv_dt);
      }
    }

    h_body_lin_vel_f_[bi] = make_float3(static_cast<float>(lin_vel.x()),
                                        static_cast<float>(lin_vel.y()),
                                        static_cast<float>(lin_vel.z()));
    h_body_omega_f_[bi] = make_float3(static_cast<float>(omega.x()),
                                      static_cast<float>(omega.y()),
                                      static_cast<float>(omega.z()));
    if (owners_dense_ && rb.owner < static_cast<unsigned int>(owner_to_body_size_)) {
      h_owner_lin_vel_[static_cast<size_t>(rb.owner)] = h_body_lin_vel_f_[bi];
      h_owner_omega_[static_cast<size_t>(rb.owner)] = h_body_omega_f_[bi];
    } else if (!owners_dense_ && !first_step_) {
      one_vel[0] = h_body_lin_vel_f_[bi];
      one_omega[0] = h_body_omega_f_[bi];
      solver_->SetOwnerVelocity(rb.owner, one_vel);
      solver_->SetOwnerAngVel(rb.owner, one_omega);
    }
    rb.prev_pos  = com;
    rb.prev_quat = quat;
  }

  if (owners_dense_ && owner_to_body_size_ > 0) {
    solver_->SetOwnerPosition(0, h_owner_pos_);
    solver_->SetOwnerOriQ(0, h_owner_quat_);
    if (!was_first_step) {
      solver_->SetOwnerVelocity(0, h_owner_lin_vel_);
      solver_->SetOwnerAngVel(0, h_owner_omega_);
    }
  }

  first_step_ = false;

  if (nb > 0) {
    HANDLE_ERROR(cudaMemcpy(d_body_pos_f_, h_body_pos_f_.data(),
                            static_cast<size_t>(nb) * sizeof(float3),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_body_quat_f_, h_body_quat_f_.data(),
                            static_cast<size_t>(nb) * sizeof(float4),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_body_lin_vel_f_, h_body_lin_vel_f_.data(),
                            static_cast<size_t>(nb) * sizeof(float3),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_body_omega_f_, h_body_omega_f_.data(),
                            static_cast<size_t>(nb) * sizeof(float3),
                            cudaMemcpyHostToDevice));
  }

  if (n_surf_verts_ > 0 && nb > 0) {
    const int blocks = (n_surf_verts_ + kThreads - 1) / kThreads;
    TransformSurfGlobalToLocalKernel<<<blocks, kThreads>>>(
        d_surf_body_id_, d_surf_pos_global_, d_body_pos_f_, d_body_quat_f_,
        d_surf_pos_local_, n_surf_verts_);
    HANDLE_ERROR(cudaGetLastError());
  }

  // Provide DEME with a per-triangle (triangle-center) velocity estimate in
  // global frame, derived from FE nodal velocities. DEME's patch-based
  // friction/damping can use this to compute relative velocity at contacts
  // (see `SetUsePatchRelativeVelocityOverride(true)` in BuildSolver).
  bool set_tri_vel = false;
  if (n_tris_ > 0 && n_surf_verts_ > 0) {
    const int blocks = (n_tris_ + kThreads - 1) / kThreads;
    BuildTriRelPosKernel<<<blocks, kThreads>>>(
        d_tri_surf_vert_ids_, d_surf_pos_local_, d_tri_relpos_n1_,
        d_tri_relpos_n2_, d_tri_relpos_n3_, n_tris_);
    HANDLE_ERROR(cudaGetLastError());
  }
  if (n_tris_ > 0 && d_tri_vel_center_ != nullptr) {
    if (have_fea_vel && d_tri_global_nodes_ != nullptr) {
      const int blocks       = (n_tris_ + kThreads - 1) / kThreads;
      ComputeTriCenterVelKernel<<<blocks, kThreads>>>(
          d_tri_global_nodes_, in.d_vel_xyz, d_tri_vel_center_, n_tris_);
      HANDLE_ERROR(cudaGetLastError());
      set_tri_vel = true;
    } else if (!have_fea_vel && d_tri_surf_vert_ids_ != nullptr &&
               d_surf_body_id_ != nullptr && d_body_quat_f_ != nullptr &&
               d_body_lin_vel_f_ != nullptr && d_body_omega_f_ != nullptr) {
      const int blocks = (n_tris_ + kThreads - 1) / kThreads;
      ComputeTriCenterVelFromRigidKernel<<<blocks, kThreads>>>(
          d_tri_surf_vert_ids_, d_surf_body_id_, d_tri_relpos_n1_,
          d_tri_relpos_n2_, d_tri_relpos_n3_, d_body_quat_f_, d_body_lin_vel_f_,
          d_body_omega_f_, d_tri_vel_center_, n_tris_);
      HANDLE_ERROR(cudaGetLastError());
      set_tri_vel = true;
    }
  }

  if (n_tris_ > 0) {
    HANDLE_ERROR(cudaStreamSynchronize(0));
    if (n_surf_verts_ > 0) {
      solver_->SetTriNodeRelPosDevice(0, d_tri_relpos_n1_, d_tri_relpos_n2_,
                                      d_tri_relpos_n3_,
                                      static_cast<size_t>(n_tris_));
    }
    if (set_tri_vel) {
      solver_->SetTriVelCenterDevice(0, d_tri_vel_center_,
                                     static_cast<size_t>(n_tris_));
    }
  }

  solver_->DoDynamics(in.dt);

  num_contacts_ = 0;
  HANDLE_ERROR(cudaMemset(d_f_contact_, 0,
                          static_cast<size_t>(3 * n_nodes_) * sizeof(double)));

  // Debug/safety controls for coupling DEM contact forces into FE:
  // - `DEME_FORCE_SCALE`: multiplies all contact forces (default 1).
  // - `DEME_FORCE_CLAMP`: clamps each contact force vector's norm (N); 0
  // disables.
  // - `DEME_FORCE_DISTRIB_K`: distribute each point force to K nearest surface
  //   vertices (default 4).
  const double force_scale = EnvDoubleOr("DEME_FORCE_SCALE", 1.0);
  const double clamp_norm  = EnvDoubleOr("DEME_FORCE_CLAMP", 0.0);
  const int k = static_cast<int>(EnvDoubleOr("DEME_FORCE_DISTRIB_K", 4.0));
  (void)params;
  const double damping_scale = 1.0;

  auto ensure_contact_capacity = [&](size_t need) {
    if (need <= contact_capacity_)
      return;
    size_t new_cap = std::max(need, contact_capacity_ * 2 + 1);
    if (d_contact_points_)
      cudaFree(d_contact_points_);
    if (d_contact_forces_)
      cudaFree(d_contact_forces_);
    if (d_contact_owner_)
      cudaFree(d_contact_owner_);
    d_contact_points_ = nullptr;
    d_contact_forces_ = nullptr;
    d_contact_owner_  = nullptr;
    HANDLE_ERROR(cudaMalloc(&d_contact_points_, new_cap * sizeof(float3)));
    HANDLE_ERROR(cudaMalloc(&d_contact_forces_, new_cap * sizeof(float3)));
    HANDLE_ERROR(cudaMalloc(&d_contact_owner_, new_cap * sizeof(uint32_t)));
    contact_capacity_ = new_cap;
  };

  const size_t need_contacts = solver_->GetNumContacts();
  if (need_contacts > 0 && n_surf_verts_ > 0 && nb > 0) {
    const size_t request_cap = need_contacts * 2;
    ensure_contact_capacity(request_cap);
    std::vector<deme::bodyID_t> owner_ids;
    owner_ids.reserve(bodies_.size());
    for (const auto& rb : bodies_) {
      if (!rb.body.skip_self_contact_forces) {
        owner_ids.push_back(static_cast<deme::bodyID_t>(rb.owner));
      }
    }
    if (!owner_ids.empty()) {
      const size_t n_useful = solver_->GetOwnerContactForcesDevice(
          owner_ids, d_contact_points_, d_contact_forces_,
          reinterpret_cast<deme::bodyID_t*>(d_contact_owner_),
          contact_capacity_);
      num_contacts_ = static_cast<int>(std::min<size_t>(
          n_useful, static_cast<size_t>(std::numeric_limits<int>::max())));
      if (n_useful > 0) {
        const int blocks =
            (static_cast<int>(n_useful) + kThreads - 1) / kThreads;
        ScatterContactForcesKNNKernel<<<blocks, kThreads>>>(
            d_contact_points_, d_contact_forces_, d_contact_owner_,
            static_cast<int>(n_useful), d_owner_to_body_, owner_to_body_size_,
            d_body_vert_start_, d_body_vert_count_, d_body_skip_forces_, nb,
            d_surf_pos_global_, d_surf_global_node_id_, k, force_scale,
            clamp_norm, damping_scale, d_f_contact_);
        HANDLE_ERROR(cudaGetLastError());
      }
    }
  }
}

const double* DemeMeshCollisionSystem::GetExternalForcesDevicePtr() const {
  return d_f_contact_;
}

int DemeMeshCollisionSystem::GetNumContacts() const {
  return num_contacts_;
}
