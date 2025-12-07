#!/usr/bin/env python3
import numpy as np

# ------------------------------------------------------------
# Affine pressure field from tet vertices + vertex pressures
# ------------------------------------------------------------

def affine_from_tet(verts, p):
    """
    Given a tetrahedron with vertices verts[0..3] in R^3 and
    scalar values p[0..3] at those vertices, construct the
    unique affine field p(x) = a · x + b that interpolates them.

    verts: (4,3)
    p    : (4,)
    returns: a (3,), b (scalar)
    """
    verts = np.asarray(verts)
    p = np.asarray(p)
    a0 = verts[0]
    T = np.column_stack((verts[1] - a0,
                         verts[2] - a0,
                         verts[3] - a0))        # 3x3
    w = np.array([p[1] - p[0],
                  p[2] - p[0],
                  p[3] - p[0]])               # 3,
    # a = T^{-T} w
    a = np.linalg.solve(T.T, w)
    b = p[0] - a @ a0
    return a, b


# ------------------------------------------------------------
# Plane–tet intersection: Π ∩ τ
# ------------------------------------------------------------

def plane_tet_intersection(verts, n, c, eps=1e-9):
    """
    Intersect plane Π = { x | n·x + c = 0 } with a tetrahedron
    given by verts[4,3].

    Returns:
        poly: (k,3) array of polygon vertices ordered CCW in the plane,
              or empty array if no intersection (area ~ 0).
    """
    verts = np.asarray(verts)
    g = verts @ n + c          # g_i = n·a_i + c
    g_max = g.max()
    g_min = g.min()

    # All strictly on one side -> no intersection
    if g_max < -eps or g_min > eps:
        return np.zeros((0, 3))

    pts = []

    # vertices on plane
    for i in range(4):
        if abs(g[i]) <= eps:
            pts.append(verts[i])

    # edges that cross plane
    edges = [(0, 1), (0, 2), (0, 3),
             (1, 2), (1, 3), (2, 3)]

    for i, j in edges:
        gi, gj = g[i], g[j]
        if gi * gj < -eps**2:  # opposite signs
            t = gi / (gi - gj)
            x = (1.0 - t) * verts[i] + t * verts[j]
            pts.append(x)

    if not pts:
        return np.zeros((0, 3))

    pts = np.vstack(pts)

    # dedupe approximately
    pts_round = np.round(pts / eps) * eps
    _, idx = np.unique(pts_round, axis=0, return_index=True)
    pts = pts[idx]

    if pts.shape[0] < 3:
        return np.zeros((0, 3))

    # order into polygon (project onto in-plane basis)
    centroid = pts.mean(axis=0)
    n_hat = n / np.linalg.norm(n)

    # in-plane axes u, v
    v0 = pts[0] - centroid
    v0 = v0 - n_hat * (np.dot(v0, n_hat))
    if np.linalg.norm(v0) < eps:
        v0 = np.array([1.0, 0.0, 0.0])
        v0 = v0 - n_hat * np.dot(v0, n_hat)
    u = v0 / np.linalg.norm(v0)
    v = np.cross(n_hat, u)

    rel = pts - centroid
    xu = rel @ u
    xv = rel @ v
    angles = np.arctan2(xv, xu)
    order = np.argsort(angles)
    return pts[order]


# ------------------------------------------------------------
# Sutherland–Hodgman clipping for poly ∩ tet
# ------------------------------------------------------------

def clip_polygon_halfspace(poly, n, p0, eps=1e-9):
    """
    Clip polygon (k,3) by halfspace H = { x | n·(x - p0) <= 0 }.
    """
    if poly.shape[0] == 0:
        return poly

    new_pts = []
    m = poly.shape[0]

    def side(x):
        return np.dot(n, x - p0)

    for i in range(m):
        A = poly[i]
        B = poly[(i + 1) % m]
        sA = side(A)
        sB = side(B)
        insideA = sA <= eps
        insideB = sB <= eps

        if insideA and insideB:
            new_pts.append(B)
        elif insideA and not insideB:
            t = sA / (sA - sB)
            X = (1.0 - t) * A + t * B
            new_pts.append(X)
        elif not insideA and insideB:
            t = sA / (sA - sB)
            X = (1.0 - t) * A + t * B
            new_pts.append(X)
            new_pts.append(B)

    if not new_pts:
        return np.zeros((0, 3))

    return np.vstack(new_pts)


def clip_polygon_with_tet(poly, tet_verts, eps=1e-9):
    """
    Clip polygon poly (k,3) with tetrahedron tet_verts[4,3].
    Returns polygon P = poly ∩ τ, possibly empty.
    """
    v = tet_verts
    # faces: (i,j,k, opposite_index)
    faces = [
        (0, 1, 2, 3),
        (0, 1, 3, 2),
        (0, 2, 3, 1),
        (1, 2, 3, 0)
    ]

    P = poly.copy()
    for i, j, k, o in faces:
        p0 = v[i]
        n_raw = np.cross(v[j] - v[i], v[k] - v[i])

        # orient normal so interior is n·(x - p0) <= 0
        if np.dot(n_raw, v[o] - p0) > 0:
            n = -n_raw
        else:
            n = n_raw

        P = clip_polygon_halfspace(P, n, p0, eps=eps)
        if P.shape[0] == 0:
            break

    return P
