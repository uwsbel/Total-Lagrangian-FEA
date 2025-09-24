import matplotlib.pyplot as plt

from grid_mesh_generator import GridMesh


def print_mesh(mesh: GridMesh) -> None:
    print("Nodes (id, i, j, x, y):")
    for n in mesh.nodes:
        print(n.id, n.i, n.j, n.x, n.y)

    print("\nElements (id, n0, n1, orientation, length) and DOF indices:")
    for e in mesh.elements:
        print(e.id, e.n0, e.n1, e.orientation, e.length)
        dof_idx = mesh.get_element_dof_indices(e.id)
        print("  dof_idx:", dof_idx)

    print("\nConnectivity (node_id: [element_ids]):")
    for nid, elist in mesh.connectivity.items():
        print(nid, elist)


def plot_mesh(mesh: GridMesh) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    # plot nodes
    xs = [n.x for n in mesh.nodes]
    ys = [n.y for n in mesh.nodes]
    ax.scatter(xs, ys, s=20, c='k', zorder=3)

    # plot elements
    for e in mesh.elements:
        n0 = mesh.nodes[e.n0]
        n1 = mesh.nodes[e.n1]
        color = 'tab:blue' if e.orientation == 'H' else 'tab:orange'
        ax.plot([n0.x, n1.x], [n0.y, n1.y], color=color, linewidth=2, zorder=2)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('GridMesh')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


def main() -> None:
    # Example configurations
    examples = [
        # (4.0, 4.0, 2.0, None, None),  # full grid 2x2
        (4.0, 0.0, 2.0, None, None),  # horizontal-only
        # (0.0, 4.0, 2.0, None, None),  # vertical-only
    ]

    for X, Y, L, ih, iv in examples:
        print("\n=== Mesh X=", X, "Y=", Y, "L=", L, "===")
        mesh = GridMesh(X, Y, L, include_horizontal=ih, include_vertical=iv)
        print_mesh(mesh)
        plot_mesh(mesh)


if __name__ == "__main__":
    main()


