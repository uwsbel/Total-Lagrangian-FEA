#pragma once

#include "Narrowphase.cuh"

// Device / kernel functions for Narrowphase

// Tolerance for geometric comparisons
__device__ const double NP_EPS = 1e-9;

// ----------------------------------------------------------------------------
// Phase 1: Build affine pressure field from tetrahedron vertices and pressures
// ----------------------------------------------------------------------------

/**
 * Solve 3x3 linear system Ax = b using Cramer's rule.
 * Returns false if matrix is singular.
 */
__device__ bool solve3x3(const double A[3][3], const double b[3], double x[3]) {
  // Compute determinant
  double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
               A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
               A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);

  if (fabs(det) < 1e-14) {
    return false;  // Singular matrix
  }

  double inv_det = 1.0 / det;

  // Cramer's rule
  x[0] = inv_det * (b[0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                    A[0][1] * (b[1] * A[2][2] - A[1][2] * b[2]) +
                    A[0][2] * (b[1] * A[2][1] - A[1][1] * b[2]));

  x[1] = inv_det * (A[0][0] * (b[1] * A[2][2] - A[1][2] * b[2]) -
                    b[0] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                    A[0][2] * (A[1][0] * b[2] - b[1] * A[2][0]));

  x[2] = inv_det * (A[0][0] * (A[1][1] * b[2] - b[1] * A[2][1]) -
                    A[0][1] * (A[1][0] * b[2] - b[1] * A[2][0]) +
                    b[0] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]));

  return true;
}

/**
 * Build affine pressure field p(x) = a·x + b from tet vertices and pressures.
 *
 * Given tetrahedron vertices v[0..3] and scalar pressures p[0..3],
 * finds unique affine field coefficients (a, b) such that:
 *   p(v[i]) = a·v[i] + b = p[i]  for i = 0..3
 *
 * @param v     Tetrahedron vertices [4][3]
 * @param p     Pressure values at vertices [4]
 * @param a_out Output gradient vector [3]
 * @param b_out Output scalar offset
 * @return      True if successful, false if tet is degenerate
 */
__device__ bool affineFromTet(const double3 v[4], const double p[4],
                              double3& a_out, double& b_out) {
  // Build matrix T = [e1|e2|e3] where ei = v[i] - v[0] are column vectors
  // We need to solve T^T * a = w where w = [p1-p0, p2-p0, p3-p0]
  double3 e1 = v[1] - v[0];
  double3 e2 = v[2] - v[0];
  double3 e3 = v[3] - v[0];

  // T^T matrix (row-major): rows are edge vectors
  // T^T[i][j] = T[j][i] = e_{i+1}[j] = edge i+1, component j
  double TT[3][3] = {
      {e1.x, e1.y, e1.z}, {e2.x, e2.y, e2.z}, {e3.x, e3.y, e3.z}};

  double w[3] = {p[1] - p[0], p[2] - p[0], p[3] - p[0]};

  double a[3];
  if (!solve3x3(TT, w, a)) {
    return false;  // Degenerate tetrahedron
  }

  a_out = make_double3(a[0], a[1], a[2]);
  b_out = p[0] - dot(a_out, v[0]);
  return true;
}

// ----------------------------------------------------------------------------
// Phase 2: Intersect plane with tetrahedron
// ----------------------------------------------------------------------------

/**
 * Intersect plane Π = {x | n·x + c = 0} with tetrahedron.
 *
 * @param v       Tetrahedron vertices [4]
 * @param n       Plane normal
 * @param c       Plane offset
 * @param poly    Output polygon (up to 6 vertices for plane-tet intersection)
 * @return        Number of vertices in resulting polygon
 */
__device__ int planeTetIntersection(const double3 v[4], double3 n, double c,
                                    ClipPolygon& poly) {
  poly.clear();

  // Compute signed distances from each vertex to plane
  double g[4];
  for (int i = 0; i < 4; i++) {
    g[i] = dot(n, v[i]) + c;
  }

  double g_max = fmax(fmax(g[0], g[1]), fmax(g[2], g[3]));
  double g_min = fmin(fmin(g[0], g[1]), fmin(g[2], g[3]));

  // All vertices on one side -> no intersection
  if (g_max < -NP_EPS || g_min > NP_EPS) {
    return 0;
  }

  // Collect intersection points
  double3 pts[7];  // Max 6 from edges + possibly some on-plane vertices
  int n_pts = 0;

  // Vertices exactly on plane
  for (int i = 0; i < 4; i++) {
    if (fabs(g[i]) <= NP_EPS) {
      pts[n_pts++] = v[i];
    }
  }

  // Edge crossings (6 edges of tetrahedron)
  const int edges[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

  for (int e = 0; e < 6; e++) {
    int i     = edges[e][0];
    int j     = edges[e][1];
    double gi = g[i];
    double gj = g[j];

    // Check if edge crosses plane (opposite signs)
    if (gi * gj < -NP_EPS * NP_EPS) {
      double t     = gi / (gi - gj);
      double3 x    = (1.0 - t) * v[i] + t * v[j];
      pts[n_pts++] = x;
    }
  }

  if (n_pts < 3) {
    return 0;
  }

  // Deduplicate points (simple O(n^2) check)
  double3 unique_pts[7];
  int n_unique = 0;

  for (int i = 0; i < n_pts; i++) {
    bool is_dup = false;
    for (int j = 0; j < n_unique; j++) {
      double3 diff = pts[i] - unique_pts[j];
      if (length(diff) < NP_EPS * 10) {
        is_dup = true;
        break;
      }
    }
    if (!is_dup && n_unique < 7) {
      unique_pts[n_unique++] = pts[i];
    }
  }

  if (n_unique < 3) {
    return 0;
  }

  // Order vertices into convex polygon (project onto plane and sort by angle)
  // Compute centroid
  double3 centroid = make_double3(0, 0, 0);
  for (int i = 0; i < n_unique; i++) {
    centroid = centroid + unique_pts[i];
  }
  centroid = (1.0 / n_unique) * centroid;

  // Build in-plane coordinate system
  double3 n_hat = normalize(n);
  double3 u, w;

  // Find first non-zero in-plane vector
  double3 v0 = unique_pts[0] - centroid;
  v0         = v0 - dot(v0, n_hat) * n_hat;

  if (length(v0) < NP_EPS) {
    // Use default direction
    v0 = make_double3(1.0, 0.0, 0.0);
    v0 = v0 - dot(v0, n_hat) * n_hat;
    if (length(v0) < NP_EPS) {
      v0 = make_double3(0.0, 1.0, 0.0);
      v0 = v0 - dot(v0, n_hat) * n_hat;
    }
  }
  u = normalize(v0);
  w = cross(n_hat, u);

  // Compute angles and sort
  double angles[7];
  int order[7];
  for (int i = 0; i < n_unique; i++) {
    double3 rel = unique_pts[i] - centroid;
    double xu   = dot(rel, u);
    double xv   = dot(rel, w);
    angles[i]   = atan2(xv, xu);
    order[i]    = i;
  }

  // Simple bubble sort (n_unique is small)
  for (int i = 0; i < n_unique - 1; i++) {
    for (int j = i + 1; j < n_unique; j++) {
      if (angles[order[j]] < angles[order[i]]) {
        int tmp  = order[i];
        order[i] = order[j];
        order[j] = tmp;
      }
    }
  }

  // Store ordered vertices in polygon
  for (int i = 0; i < n_unique && i < MAX_POLYGON_VERTS; i++) {
    poly.addVertex(unique_pts[order[i]]);
  }

  return poly.count;
}

// ----------------------------------------------------------------------------
// Phase 3: Sutherland-Hodgman polygon clipping against halfspace
// ----------------------------------------------------------------------------

/**
 * Clip polygon against halfspace H = {x | n·(x - p0) <= 0}.
 *
 * @param poly_in   Input polygon
 * @param n         Halfspace normal (points outward)
 * @param p0        Point on halfspace boundary
 * @param poly_out  Output clipped polygon
 */
__device__ void clipPolygonHalfspace(const ClipPolygon& poly_in, double3 n,
                                     double3 p0, ClipPolygon& poly_out) {
  poly_out.clear();

  if (poly_in.count == 0) {
    return;
  }

  int m = poly_in.count;

  for (int i = 0; i < m; i++) {
    double3 A = poly_in.verts[i];
    double3 B = poly_in.verts[(i + 1) % m];

    double sA = dot(n, A - p0);
    double sB = dot(n, B - p0);

    bool insideA = (sA <= NP_EPS);
    bool insideB = (sB <= NP_EPS);

    if (insideA && insideB) {
      // Both inside -> keep B
      poly_out.addVertex(B);
    } else if (insideA && !insideB) {
      // A inside, B outside -> add intersection
      double t  = sA / (sA - sB);
      double3 X = (1.0 - t) * A + t * B;
      poly_out.addVertex(X);
    } else if (!insideA && insideB) {
      // A outside, B inside -> add intersection and B
      double t  = sA / (sA - sB);
      double3 X = (1.0 - t) * A + t * B;
      poly_out.addVertex(X);
      poly_out.addVertex(B);
    }
    // else: both outside -> skip
  }
}

// ----------------------------------------------------------------------------
// Phase 4: Clip polygon with tetrahedron (4 halfspaces)
// ----------------------------------------------------------------------------

/**
 * Clip polygon against tetrahedron using Sutherland-Hodgman.
 * Clips against all 4 faces of the tetrahedron.
 *
 * @param poly_in   Input polygon
 * @param tet       Tetrahedron vertices [4]
 * @param poly_out  Output clipped polygon
 */
__device__ void clipPolygonWithTet(const ClipPolygon& poly_in,
                                   const double3 tet[4],
                                   ClipPolygon& poly_out) {
  // Face definitions: (i, j, k, opposite_vertex)
  // Face normal computed as cross(v[j]-v[i], v[k]-v[i])
  // Oriented so interior is n·(x - v[i]) <= 0
  const int faces[4][4] = {
      {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 3, 1}, {1, 2, 3, 0}};

  ClipPolygon temp1, temp2;
  ClipPolygon* current = &temp1;
  ClipPolygon* next    = &temp2;

  // Copy input to current
  *current = poly_in;

  for (int f = 0; f < 4; f++) {
    if (current->count == 0)
      break;

    int i = faces[f][0];
    int j = faces[f][1];
    int k = faces[f][2];
    int o = faces[f][3];

    double3 p0    = tet[i];
    double3 n_raw = cross(tet[j] - tet[i], tet[k] - tet[i]);

    // Orient normal so interior (opposite vertex) satisfies n·(x-p0) <= 0
    double3 n;
    if (dot(n_raw, tet[o] - p0) > 0) {
      n = -1.0 * n_raw;
    } else {
      n = n_raw;
    }

    clipPolygonHalfspace(*current, n, p0, *next);

    // Swap buffers
    ClipPolygon* tmp = current;
    current          = next;
    next             = tmp;
  }

  // Reorder vertices by angle around centroid to ensure proper polygon winding
  if (current->count >= 3) {
    // Compute centroid
    double3 cent = make_double3(0, 0, 0);
    for (int i = 0; i < current->count; i++) {
      cent = cent + current->verts[i];
    }
    cent = (1.0 / current->count) * cent;

    // Compute polygon normal from first 3 vertices
    double3 n_poly = cross(current->verts[1] - current->verts[0],
                           current->verts[2] - current->verts[0]);
    double n_len   = length(n_poly);
    if (n_len > NP_EPS) {
      n_poly = (1.0 / n_len) * n_poly;
    } else {
      // Fallback: use Z-axis
      n_poly = make_double3(0, 0, 1);
    }

    // Build in-plane coordinate system
    double3 u_dir = current->verts[0] - cent;
    u_dir         = u_dir - dot(u_dir, n_poly) * n_poly;
    double u_len  = length(u_dir);
    if (u_len < NP_EPS) {
      u_dir = make_double3(1, 0, 0);
      u_dir = u_dir - dot(u_dir, n_poly) * n_poly;
      u_len = length(u_dir);
      if (u_len < NP_EPS) {
        u_dir = make_double3(0, 1, 0);
        u_dir = u_dir - dot(u_dir, n_poly) * n_poly;
        u_len = length(u_dir);
      }
    }
    u_dir         = (1.0 / u_len) * u_dir;
    double3 v_dir = cross(n_poly, u_dir);

    // Compute angles and sort
    double angles[MAX_POLYGON_VERTS];
    int order[MAX_POLYGON_VERTS];
    for (int i = 0; i < current->count; i++) {
      double3 rel = current->verts[i] - cent;
      angles[i]   = atan2(dot(rel, v_dir), dot(rel, u_dir));
      order[i]    = i;
    }

    // Bubble sort by angle
    for (int i = 0; i < current->count - 1; i++) {
      for (int j = i + 1; j < current->count; j++) {
        if (angles[order[j]] < angles[order[i]]) {
          int tmp  = order[i];
          order[i] = order[j];
          order[j] = tmp;
        }
      }
    }

    // Store reordered vertices
    poly_out.clear();
    for (int i = 0; i < current->count; i++) {
      poly_out.addVertex(current->verts[order[i]]);
    }
  } else {
    poly_out = *current;
  }
}

// ----------------------------------------------------------------------------
// Phase 5: Compute polygon area and centroid
// ----------------------------------------------------------------------------

/**
 * Compute area and centroid of a convex polygon.
 * Uses triangulation from first vertex.
 *
 * @param poly      Input polygon
 * @param area      Output area
 * @param centroid  Output centroid
 */
__device__ void computePolygonAreaAndCentroid(const ClipPolygon& poly,
                                              double& area, double3& centroid) {
  area     = 0.0;
  centroid = make_double3(0, 0, 0);

  if (poly.count < 3) {
    return;
  }

  // Triangulate from vertex 0
  double3 v0           = poly.verts[0];
  double3 weighted_sum = make_double3(0, 0, 0);
  double total_area    = 0.0;

  for (int i = 1; i < poly.count - 1; i++) {
    double3 v1 = poly.verts[i];
    double3 v2 = poly.verts[i + 1];

    // Triangle area = 0.5 * |cross(v1-v0, v2-v0)|
    double3 c       = cross(v1 - v0, v2 - v0);
    double tri_area = 0.5 * length(c);

    // Triangle centroid
    double3 tri_centroid = (1.0 / 3.0) * (v0 + v1 + v2);

    weighted_sum = weighted_sum + tri_area * tri_centroid;
    total_area += tri_area;
  }

  area = total_area;
  if (total_area > NP_EPS) {
    centroid = (1.0 / total_area) * weighted_sum;
  } else {
    // Fallback: average of vertices
    for (int i = 0; i < poly.count; i++) {
      centroid = centroid + poly.verts[i];
    }
    centroid = (1.0 / poly.count) * centroid;
  }
}

// ----------------------------------------------------------------------------
// Main narrowphase kernel: Compute contact patches for collision pairs
// ----------------------------------------------------------------------------

__global__ void computeContactPatchesKernel(Narrowphase* np,
                                            ContactPatch* patches,
                                            const int* collisionPairs,
                                            int numPairs) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numPairs)
    return;

  // Get tet pair indices
  int tetA = collisionPairs[2 * idx];
  int tetB = collisionPairs[2 * idx + 1];

  // Get mesh IDs for each element
  // Normal direction: from lower mesh ID to higher mesh ID
  int meshIdA = np->d_elementMeshIds[tetA];
  int meshIdB = np->d_elementMeshIds[tetB];

  // Swap if necessary: ensure tetA belongs to mesh with lower ID
  // This guarantees normal points from lower mesh ID to higher mesh ID
  if (meshIdA > meshIdB) {
    int temp = tetA;
    tetA     = tetB;
    tetB     = temp;
    // Swap mesh IDs too for clarity (though not used after this)
    int tempId = meshIdA;
    meshIdA    = meshIdB;
    meshIdB    = tempId;
  }

  // Initialize output patch
  ContactPatch& patch = patches[idx];
  patch.tetA_idx      = tetA;
  patch.tetB_idx      = tetB;
  patch.isValid       = false;

  // Get tet A vertices and pressures (first 4 nodes = corners)
  double3 vA[4];
  double pA[4];
  for (int i = 0; i < 4; i++) {
    int node_id = np->element_node(tetA, i);
    vA[i]       = np->getNode(node_id);
    pA[i]       = np->getPressure(node_id);
  }

  // Get tet B vertices and pressures
  double3 vB[4];
  double pB[4];
  for (int i = 0; i < 4; i++) {
    int node_id = np->element_node(tetB, i);
    vB[i]       = np->getNode(node_id);
    pB[i]       = np->getPressure(node_id);
  }

  // Phase 1: Build affine pressure fields
  double3 aA, aB;
  double bA, bB;

  if (!affineFromTet(vA, pA, aA, bA) || !affineFromTet(vB, pB, aB, bB)) {
    return;  // Degenerate tetrahedron
  }

  // Compute iso-pressure plane: p_A(x) = p_B(x)
  // => (aA - aB)·x + (bA - bB) = 0
  // => n·x + c = 0
  double3 n = aA - aB;
  double c  = bA - bB;

  double n_norm = length(n);
  if (n_norm < NP_EPS) {
    return;  // Parallel pressure fields (no iso-surface)
  }

  // Phase 2: Intersect plane with tet A
  ClipPolygon poly_A;
  if (planeTetIntersection(vA, n, c, poly_A) < 3) {
    return;  // No intersection with tet A
  }

  // Phase 3-4: Clip polygon with tet B
  ClipPolygon poly_final;
  clipPolygonWithTet(poly_A, vB, poly_final);

  if (poly_final.count < 3) {
    return;  // No intersection after clipping
  }

  // Phase 5: Compute area and centroid
  double area;
  double3 centroid;
  computePolygonAreaAndCentroid(poly_final, area, centroid);

  if (area < NP_EPS * NP_EPS) {
    return;  // Degenerate patch
  }

  // Compute contact normal (normalized iso-plane normal)
  double3 nhat = (1.0 / n_norm) * n;

  // Compute directional gradients (Drake convention)
  // g_A = -∂p_A/∂n (moving into A increases p_A)
  // g_B = ∂p_B/∂n (moving into B increases p_B)
  double g_A = -dot(aA, nhat);
  double g_B = dot(aB, nhat);

  bool valid_orientation = true;

  // Enforce Drake convention: g_A > 0 and g_B > 0
  // If not satisfied, try flipping the normal
  if (g_A <= 0 || g_B <= 0) {
    nhat = -1.0 * nhat;
    g_A  = -dot(aA, nhat);
    g_B  = dot(aB, nhat);

    if (g_A <= 0 || g_B <= 0) {
      valid_orientation = false;
      // Revert to original normal
      nhat = (1.0 / n_norm) * n;
      g_A  = -dot(aA, nhat);
      g_B  = dot(aB, nhat);
    }
  }

  // Compute equilibrium pressure at centroid
  double p_eq = dot(aA, centroid) + bA;

  // Store results
  patch.numVertices = poly_final.count;
  for (int i = 0; i < poly_final.count; i++) {
    patch.vertices[i] = poly_final.verts[i];
  }
  patch.normal           = nhat;
  patch.centroid         = centroid;
  patch.area             = area;
  patch.g_A              = g_A;
  patch.g_B              = g_B;
  patch.p_equilibrium    = p_eq;
  patch.isValid          = true;
  patch.validOrientation = valid_orientation;
}
