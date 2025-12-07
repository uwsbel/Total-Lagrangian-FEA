import argparse
import numpy as np


def load_tetgen_nodes(node_path):
    """
    Load TetGen .node file.

    Returns:
        coords: (N, 3) float array
        index_offset: int, so that coords[node_id - index_offset] is valid.
    """
    with open(node_path, "r") as f:
        # header: <#points> <dim> <#attrs> <#bmarkers>
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            header = line.split()
            break

        if len(header) < 2:
            raise ValueError("Invalid .node header")

        n_points = int(header[0])
        dim = int(header[1])
        if dim != 3:
            raise ValueError(f"Only 3D nodes are supported, got dim={dim}")

        ids = []
        coords_list = []

        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 1 + dim:
                continue
            node_id = int(parts[0])
            xyz = [float(v) for v in parts[1:1 + dim]]
            ids.append(node_id)
            coords_list.append(xyz)

    ids = np.array(ids, dtype=int)
    coords_list = np.array(coords_list, dtype=float)

    if len(coords_list) != n_points:
        print(f"Warning: header says {n_points} points but found {len(coords_list)}")

    min_id = int(ids.min())
    max_id = int(ids.max())
    index_offset = min_id
    N = max_id - min_id + 1

    if N != len(coords_list):
        raise ValueError(
            "Node ids are not contiguous; this loader assumes contiguous ids."
        )

    coords = np.zeros((N, dim), dtype=float)
    for raw_idx, node_id in enumerate(ids):
        coords[node_id - index_offset] = coords_list[raw_idx]

    return coords, index_offset


def load_tetgen_elements(ele_path, index_offset):
    """
    Load TetGen .ele file.

    Returns:
        tets: (M, K) int array of node indices (0-based).
    """
    with open(ele_path, "r") as f:
        # header: <#elems> <nodes_per_elem> <#attrs>
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            header = line.split()
            break

        if len(header) < 2:
            raise ValueError("Invalid .ele header")

        n_elems = int(header[0])
        nodes_per_elem = int(header[1])

        tets = np.zeros((n_elems, nodes_per_elem), dtype=int)
        count = 0

        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 1 + nodes_per_elem:
                continue

            # parts[0] = element id; next nodes_per_elem entries are node ids
            node_ids = [int(v) - index_offset for v in parts[1:1 + nodes_per_elem]]
            tets[count, :] = node_ids
            count += 1
            if count >= n_elems:
                break

    if count != n_elems:
        print(f"Warning: header says {n_elems} elements but found {count}")

    return tets


def compute_tet_aabbs(nodes, tets):
    """
    nodes: (N, 3)
    tets: (M, K) int

    Returns:
        mins: (M, 3)
        maxs: (M, 3)
    """
    tet_xyz = nodes[tets]  # (M, K, 3)
    mins = tet_xyz.min(axis=1)
    maxs = tet_xyz.max(axis=1)
    return mins, maxs


def sweep_and_prune_aabbs(mins, maxs):
    """
    1D sweep-and-prune on x, refine with y/z AABB overlap.

    mins, maxs: (M, 3)

    Returns:
        pairs: sorted list of (i, j) with i < j
    """
    n = mins.shape[0]
    events = []
    for i in range(n):
        events.append((mins[i, 0], 0, i))  # start
        events.append((maxs[i, 0], 1, i))  # end

    events.sort(key=lambda e: (e[0], e[1]))

    active = set()
    pairs = []

    for coord, typ, idx in events:
        if typ == 0:
            # new interval entering
            for j in active:
                # refine in y,z
                if not (
                    mins[idx, 1] > maxs[j, 1] or maxs[idx, 1] < mins[j, 1] or
                    mins[idx, 2] > maxs[j, 2] or maxs[idx, 2] < mins[j, 2]
                ):
                    a, b = (idx, j) if idx < j else (j, idx)
                    pairs.append((a, b))
            active.add(idx)
        else:
            active.discard(idx)

    pairs = sorted(set(pairs))
    return pairs


def compute_neighbor_pairs(tets, use_all_nodes=True):
    """
    Build a set of (i, j) tet pairs that are neighbors,
    meaning they share at least one vertex.

    tets: (M, K) int array of node indices.
    use_all_nodes:
        - True  -> use all K nodes (for 10-node tets, includes midside nodes)
        - False -> only first 4 corner nodes (for 10-node tets)
    """
    n_elems, k = tets.shape
    node_count = k if use_all_nodes else min(4, k)

    # node -> list of tets using that node
    node_to_tets = {}

    for e in range(n_elems):
        for local in range(node_count):
            v = int(tets[e, local])
            if v not in node_to_tets:
                node_to_tets[v] = [e]
            else:
                node_to_tets[v].append(e)

    neighbor_pairs = set()

    # any two tets that share this node are neighbors
    for v, elems in node_to_tets.items():
        if len(elems) <= 1:
            continue
        L = len(elems)
        for i in range(L):
            ei = elems[i]
            for j in range(i + 1, L):
                ej = elems[j]
                a, b = (ei, ej) if ei < ej else (ej, ei)
                neighbor_pairs.add((a, b))

    return neighbor_pairs


def visualize(nodes, tets, mins, maxs, pairs, max_tets=None):
    """
    Visualize mesh + some overlapping AABBs using matplotlib.

    - For 10-node tets (TetGen -o2), the first 4 indices are the corners.
    - `pairs` is typically the non-neighbor overlap list.
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("matplotlib not installed; skipping visualization.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # ----- plot tetrahedra -----
    n_tets = tets.shape[0]
    if max_tets is None or max_tets <= 0 or max_tets > n_tets:
        max_tets = n_tets

    for ti in range(max_tets):
        # For second-order tets: [v0, v1, v2, v3, m01, m02, m03, m12, m13, m23]
        corner_ids = tets[ti, :4]
        pts = nodes[corner_ids]  # (4, 3)
        edges = [
            (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 3),
            (2, 3)
        ]
        for a, b in edges:
            xs = [pts[a, 0], pts[b, 0]]
            ys = [pts[a, 1], pts[b, 1]]
            zs = [pts[a, 2], pts[b, 2]]
            ax.plot(xs, ys, zs, linewidth=0.4)

    # ----- draw AABBs for some overlapping pairs -----
    def draw_aabb(mn, mx):
        x0, y0, z0 = mn
        x1, y1, z1 = mx
        corners = np.array([
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ])
        idx_pairs = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for i0, i1 in idx_pairs:
            xs = [corners[i0, 0], corners[i1, 0]]
            ys = [corners[i0, 1], corners[i1, 1]]
            zs = [corners[i0, 2], corners[i1, 2]]
            ax.plot(xs, ys, zs, linestyle="--", linewidth=0.8)

    max_pairs_to_show = min(30, len(pairs))
    for k in range(max_pairs_to_show):
        i, j = pairs[k]
        draw_aabb(mins[i], maxs[i])
        draw_aabb(mins[j], maxs[j])

    # equal aspect ratio so the mesh doesn’t look squashed
    xyz = nodes
    x_range = xyz[:, 0].max() - xyz[:, 0].min()
    y_range = xyz[:, 1].max() - xyz[:, 1].min()
    z_range = xyz[:, 2].max() - xyz[:, 2].min()
    max_range = max(x_range, y_range, z_range)
    x_mid = (xyz[:, 0].max() + xyz[:, 0].min()) / 2
    y_mid = (xyz[:, 1].max() + xyz[:, 1].min()) / 2
    z_mid = (xyz[:, 2].max() + xyz[:, 2].min()) / 2

    ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Tet mesh + non-neighbor overlapping AABBs (broadphase)")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Build AABBs per tet and detect overlaps via sweep-and-prune, with neighbor filtering."
    )
    parser.add_argument("node_file", help="TetGen .node file")
    parser.add_argument("ele_file", help="TetGen .ele file")
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable matplotlib visualization"
    )
    parser.add_argument(
        "--max-tets", type=int, default=-1,
        help="Max tets to plot (<=0 means all)"
    )
    parser.add_argument(
        "--neighbors-corners-only", action="store_true",
        help="When building neighbor pairs, only consider shared corner nodes (first 4). "
             "Default: use all nodes (including midside nodes)."
    )
    args = parser.parse_args()

    print("Loading nodes...")
    nodes, index_offset = load_tetgen_nodes(args.node_file)
    print(f"  Loaded {nodes.shape[0]} nodes (index offset = {index_offset}).")

    print("Loading elements...")
    tets = load_tetgen_elements(args.ele_file, index_offset=index_offset)
    print(f"  Loaded {tets.shape[0]} elements with {tets.shape[1]} nodes each.")

    print("Computing AABBs...")
    mins, maxs = compute_tet_aabbs(nodes, tets)

    print("Running sweep-and-prune to find ALL overlapping AABBs...")
    pairs_all = sweep_and_prune_aabbs(mins, maxs)
    print(f"  Found {len(pairs_all)} overlapping AABB pairs (including neighbors).")

    print("Computing neighbor tet pairs (sharing at least one vertex)...")
    use_all_nodes = not args.neighbors_corners_only
    neighbor_pairs = compute_neighbor_pairs(tets, use_all_nodes=use_all_nodes)
    print(f"  Found {len(neighbor_pairs)} neighbor pairs from connectivity.")

    print("Filtering out neighbor pairs from AABB overlaps...")
    neighbor_pairs_set = neighbor_pairs  # already a set
    pairs_non_neighbor = [p for p in pairs_all if p not in neighbor_pairs_set]
    print(f"  Remaining non-neighbor overlapping AABB pairs: {len(pairs_non_neighbor)}")

    if not args.no_plot:
        visualize(nodes, tets, mins, maxs, pairs_non_neighbor, max_tets=args.max_tets)


if __name__ == "__main__":
    main()

