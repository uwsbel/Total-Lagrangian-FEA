import numpy as np

def remap_tetgen_tet10_indices(tetgen_elem):
    # TetGen order: [v0, v1, v2, v3, (3-4), (1-4), (1-2), (2-3), (2-4), (1-3)]
    # Standard order: [v0, v1, v2, v3, (0-1), (1-2), (0-2), (0-3), (1-3), (2-3)]
    tetgen_to_standard = [0, 1, 2, 3, 6, 7, 9, 5, 8, 4]
    return [tetgen_elem[i] for i in tetgen_to_standard]

def read_node(fname):
    with open(fname) as f:
        n, dim, *_ = map(int, f.readline().split())
        X = np.zeros((n, dim))
        for _ in range(n):
            parts = f.readline().split()
            if parts:
                i = int(parts[0]) - 1
                X[i] = list(map(float, parts[1:4]))
    return X

def read_ele(fname):
    with open(fname) as f:
        m, k, *_ = map(int, f.readline().split())
        E = np.zeros((m, k), dtype=int)
        for _ in range(m):
            parts = f.readline().split()
            if parts:
                i = int(parts[0]) - 1
                raw_elem = [int(v) - 1 for v in parts[1:1+k]]
                E[i] = remap_tetgen_tet10_indices(raw_elem)
    return E

