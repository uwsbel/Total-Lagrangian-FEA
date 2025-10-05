import numpy as np

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
                E[i] = [int(v) - 1 for v in parts[1:1+k]]
    return E

X = read_node("cube.1.node")
E = read_ele("cube.1.ele")

print("Nodes:", X.shape, "  Elems:", E.shape)
print("Indices OK:", E.min() >= 0 and E.max() < len(X))
