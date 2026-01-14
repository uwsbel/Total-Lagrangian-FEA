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

static float3 MeanOf(const std::vector<float3>& nodes) {
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

static std::vector<float3> Subtract(const std::vector<float3>& nodes,
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

static float3 Vec3FromNodesXYZ(const std::vector<double>& h_nodes_xyz,
                               int n_nodes, int global_node_id) {
  const double x = h_nodes_xyz[global_node_id];
  const double y = h_nodes_xyz[n_nodes + global_node_id];
  const double z = h_nodes_xyz[2 * n_nodes + global_node_id];
  return make_float3(static_cast<float>(x), static_cast<float>(y),
                     static_cast<float>(z));
}

static void AccumulatePointForcesToKNearestNodes(
    const std::vector<float3>& mesh_vertices,
    const std::vector<int>& global_node_ids, const std::vector<float3>& points,
    const std::vector<float3>& forces, int k, double force_scale,
    double clamp_force_norm, std::vector<double>& h_f_contact) {
  if (k <= 0)
    return;
  for (size_t i = 0; i < points.size() && i < forces.size(); ++i) {
    float3 f = forces[i];
    f.x      = static_cast<float>(static_cast<double>(f.x) * force_scale);
    f.y      = static_cast<float>(static_cast<double>(f.y) * force_scale);
    f.z      = static_cast<float>(static_cast<double>(f.z) * force_scale);

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

}  // namespace

DemeMeshCollisionSystem::DemeMeshCollisionSystem(
    std::vector<DemeMeshCollisionBody> bodies, double friction,
    bool enable_self_collision)
    : enable_self_collision_(enable_self_collision) {
  if (bodies.empty()) {
    throw std::invalid_argument(
        "DemeMeshCollisionSystem: bodies must be non-empty");
  }

  bodies_.reserve(bodies.size());
  size_t tri_start = 0;
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
    tri_start += rb.body.surface.triangles.size();
    bodies_.push_back(std::move(rb));
  }

  BuildSolver(friction);
}

DemeMeshCollisionSystem::~DemeMeshCollisionSystem() {
  if (d_f_contact_ != nullptr) {
    cudaFree(d_f_contact_);
    d_f_contact_ = nullptr;
  }
}

void DemeMeshCollisionSystem::BuildSolver(double friction) {
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
  // - Default `E` here is chosen to roughly match this repo's `test_item_drop`
  //   material scale; override with env var `DEME_CONTACT_E` when needed.
  const float contact_E   = EnvFloatOr("DEME_CONTACT_E", 1.0e7f);
  const float contact_nu  = EnvFloatOr("DEME_CONTACT_NU", 0.3f);
  const float contact_cor = EnvFloatOr("DEME_CONTACT_COR", 0.0f);

  const float mu = static_cast<float>(std::max(0.0, friction));
  auto mat       = solver_->LoadMaterial({{"E", contact_E},
                                          {"nu", contact_nu},
                                          {"CoR", contact_cor},
                                          {"mu", mu},
                                          {"Crr", 0.0f}});

  auto make_mesh = [&](const ANCFCPUUtils::SurfaceTriMesh& surf,
                       unsigned int family, bool split_into_patches,
                       float patch_angle_deg, const char* label) {
    deme::DEMMesh mesh;
    mesh.SetFamily(family);

    // DEME expects mesh vertices in the owner's local frame. Keep the owner
    // init position meaningful by centering vertices at the initial centroid.
    const Eigen::Vector3d centroid = ComputeCentroid(surf.vertices);
    mesh.SetInitPos(ToFloat3(centroid));

    mesh.m_vertices.reserve(surf.vertices.size());
    for (const auto& v : surf.vertices) {
      mesh.m_vertices.push_back(ToFloat3(v - centroid));
    }

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
    mesh.SetMass(1.f);
    mesh.SetMOI(make_float3(1.f, 1.f, 1.f));

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

    auto mesh = make_mesh(body.surface, body.family, body.split_into_patches,
                          patch_angle, label.c_str());
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

  if (d_f_contact_ == nullptr) {
    HANDLE_ERROR(cudaMalloc(
        &d_f_contact_, static_cast<size_t>(3 * n_nodes_) * sizeof(double)));
  }
}

void DemeMeshCollisionSystem::Step(const CollisionSystemInput& in,
                                   const CollisionSystemParams& params) {
  (void)params;
  if (d_nodes_xyz_ == nullptr || n_nodes_ <= 0) {
    throw std::runtime_error(
        "DemeMeshCollisionSystem::Step called before BindNodesDevicePtr");
  }
  if (in.dt <= 0.0) {
    throw std::invalid_argument(
        "DemeMeshCollisionSystem::Step: dt must be > 0");
  }

  HANDLE_ERROR(cudaMemcpy(h_nodes_xyz_.data(), d_nodes_xyz_,
                          static_cast<size_t>(3 * n_nodes_) * sizeof(double),
                          cudaMemcpyDeviceToHost));

  std::vector<std::vector<float3>> nodes_global(bodies_.size());

  // DEME expects triangle node positions in the mesh owner's local frame, and
  // uses owner position + patch locations for contact detection. Keep owner
  // positions updated, and send local vertex positions.
  for (size_t bi = 0; bi < bodies_.size(); ++bi) {
    const auto& rb   = bodies_[bi];
    const auto& surf = rb.body.surface;

    auto& nodes = nodes_global[bi];
    nodes.resize(surf.vertices.size());
    for (size_t i = 0; i < surf.global_node_ids.size(); ++i) {
      nodes[i] =
          Vec3FromNodesXYZ(h_nodes_xyz_, n_nodes_, surf.global_node_ids[i]);
    }

    const float3 com = MeanOf(nodes);
    if (rb.owner == static_cast<unsigned int>(deme::NULL_BODYID)) {
      throw std::runtime_error(
          "DemeMeshCollisionSystem::Step: DEME mesh owner ID is NULL_BODYID");
    }
    solver_->SetOwnerPosition(rb.owner, std::vector<float3>{com});
    const std::vector<float3> nodes_local = Subtract(nodes, com);
    solver_->SetTriNodeRelPos(rb.owner, rb.tri_start, nodes_local);
  }

  solver_->DoDynamics(in.dt);

  std::fill(h_f_contact_.begin(), h_f_contact_.end(), 0.0);
  num_contacts_ = 0;

  // Debug/safety controls for coupling DEM contact forces into FE:
  // - `DEME_FORCE_SCALE`: multiplies all contact forces (default 1).
  // - `DEME_FORCE_CLAMP`: clamps each contact force vector's norm (N); 0
  // disables.
  // - `DEME_FORCE_DISTRIB_K`: distribute each point force to K nearest surface
  //   vertices (default 4).
  const double force_scale = EnvDoubleOr("DEME_FORCE_SCALE", 1.0);
  const double clamp_norm  = EnvDoubleOr("DEME_FORCE_CLAMP", 0.0);
  const int k = static_cast<int>(EnvDoubleOr("DEME_FORCE_DISTRIB_K", 4.0));

  for (size_t bi = 0; bi < bodies_.size(); ++bi) {
    const auto& rb    = bodies_[bi];
    const auto& surf  = rb.body.surface;
    const auto& nodes = nodes_global[bi];

    std::vector<float3> points, forces;
    num_contacts_ += static_cast<int>(
        solver_->GetOwnerContactForces({rb.owner}, points, forces));
    AccumulatePointForcesToKNearestNodes(nodes, surf.global_node_ids, points,
                                         forces, k, force_scale, clamp_norm,
                                         h_f_contact_);
  }

  HANDLE_ERROR(cudaMemcpy(d_f_contact_, h_f_contact_.data(),
                          static_cast<size_t>(3 * n_nodes_) * sizeof(double),
                          cudaMemcpyHostToDevice));
}

const double* DemeMeshCollisionSystem::GetExternalForcesDevicePtr() const {
  return d_f_contact_;
}

int DemeMeshCollisionSystem::GetNumContacts() const {
  return num_contacts_;
}
