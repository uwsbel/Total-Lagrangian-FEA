from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


@dataclass(frozen=True)
class Node:
    id: int
    i: int
    j: int
    x: float
    y: float
    dof_x: float
    dof_dx_du: float
    dof_dx_dv: float
    dof_dx_dw: float


@dataclass(frozen=True)
class Element:
    id: int
    n0: int
    n1: int
    orientation: str  # 'H' or 'V'
    length: float


class GridMesh:
    """Structured grid of beams with 4 DOF per node and 2 nodes per element.

    - Nodes are generated on a regular lattice with spacing L within [0, X] x [0, Y].
    - Elements are generated with all horizontals first (row-major), then verticals (column-major).
    - Each node contributes 4 DOFs ordered as [x, dx/du, dx/dv, dx/dw].
    - Each element has 8 DOFs: node0's 4 followed by node1's 4.
    """

    def __init__(self, X: float, Y: float, L: float, include_horizontal: Optional[bool] = None, include_vertical: Optional[bool] = None) -> None:
        if L <= 0:
            raise ValueError("L must be > 0")
        if abs(round(X / L) * L - X) > 1e-12 or abs(round(Y / L) * L - Y) > 1e-12:
            raise ValueError("X and Y must be exact multiples of L")

        self.X: float = float(X)
        self.Y: float = float(Y)
        self.L: float = float(L)

        self.nx: int = int(round(self.X / self.L))  # number of intervals in x
        self.ny: int = int(round(self.Y / self.L))  # number of intervals in y

        # If flags are not provided, infer from geometry:
        # - horizontal elements exist only if nx > 0 (X > 0)
        # - vertical elements exist only if ny > 0 (Y > 0)
        self.include_horizontal: bool = (include_horizontal if include_horizontal is not None else (self.nx > 0))
        self.include_vertical: bool = (include_vertical if include_vertical is not None else (self.ny > 0))

        self.nodes: List[Node] = self._generate_nodes()
        self.elements: List[Element] = self._generate_elements()
        self.connectivity: Dict[int, List[int]] = self._build_connectivity()

        # Optional global DOF vector: 4 DOF per node
        self.u: List[float] = [0.0] * (4 * len(self.nodes))

    # -------------------- generation --------------------
    def _generate_nodes(self) -> List[Node]:
        nodes: List[Node] = []
        node_id_counter = 0
        for j in range(self.ny + 1):  # row-major by j then i
            y = j * self.L
            for i in range(self.nx + 1):
                x = i * self.L
                # Initialize DOF values to 0, will be set by generate_nodal_coordinates
                nodes.append(Node(id=node_id_counter, i=i, j=j, x=x, y=y, 
                                dof_x=0.0, dof_dx_du=0.0, dof_dx_dv=0.0, dof_dx_dw=0.0))
                node_id_counter += 1
        return nodes

    def _generate_elements(self) -> List[Element]:
        elements: List[Element] = []
        eid = 0

        # Horizontals first: for each row j, elements between (i,j) -> (i+1,j)
        if self.include_horizontal:
            for j in range(self.ny + 1):
                for i in range(self.nx):
                    n0 = self.node_id(i, j)
                    n1 = self.node_id(i + 1, j)
                    elements.append(Element(id=eid, n0=n0, n1=n1, orientation='H', length=self.L))
                    eid += 1

        # Verticals next: for each column i, elements between (i,j) -> (i,j+1)
        if self.include_vertical:
            for i in range(self.nx + 1):
                for j in range(self.ny):
                    n0 = self.node_id(i, j)
                    n1 = self.node_id(i, j + 1)
                    elements.append(Element(id=eid, n0=n0, n1=n1, orientation='V', length=self.L))
                    eid += 1

        return elements

    def _build_connectivity(self) -> Dict[int, List[int]]:
        conn: Dict[int, List[int]] = {node.id: [] for node in self.nodes}
        for e in self.elements:
            conn[e.n0].append(e.id)
            conn[e.n1].append(e.id)
        return conn

    # -------------------- nodal coordinates generation --------------------
    def generate_nodal_coordinates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate nodal coordinates for all nodes using beam pattern.
        
        Returns:
            x_coords, y_coords, z_coords: arrays of length 4 * num_nodes
            Each node contributes 4 DOFs: [x, dx/du, dx/dv, dx/dw]
        """
        n_nodes = len(self.nodes)
        x_coords = np.zeros(4 * n_nodes, dtype=float)
        y_coords = np.zeros(4 * n_nodes, dtype=float)
        z_coords = np.zeros(4 * n_nodes, dtype=float)
        
        # Define the pattern for each node (from beam_mesh_generator)
        x_pattern = [1.0, 0.0, 0.0]  # [dx/du, dx/dv, dx/dw] for x-coordinate
        y_pattern = [1.0, 0.0, 1.0, 0.0]  # [y, dx/du, dx/dv, dx/dw] for y-coordinate  
        z_pattern = [0.0, 0.0, 0.0, 1.0]  # [z, dx/du, dx/dv, dx/dw] for z-coordinate
        
        # Fill the arrays
        for node in self.nodes:
            idx = 4 * node.id
            x_coords[idx] = node.x  # x position
            x_coords[idx+1:idx+4] = x_pattern  # dx/du, dx/dv, dx/dw for x
            y_coords[idx:idx+4] = y_pattern  # y, dx/du, dx/dv, dx/dw for y
            z_coords[idx:idx+4] = z_pattern  # z, dx/du, dx/dv, dx/dw for z
            
        return x_coords, y_coords, z_coords

    # -------------------- indexing helpers --------------------
    def node_id(self, i: int, j: int) -> int:
        if not (0 <= i <= self.nx and 0 <= j <= self.ny):
            raise IndexError("(i,j) out of range")
        return j * (self.nx + 1) + i

    @staticmethod
    def global_dof_indices_for_node(node_id_value: int) -> Tuple[int, int, int, int]:
        base = 4 * node_id_value
        return base + 0, base + 1, base + 2, base + 3

    # -------------------- accessors --------------------
    def get_element(self, e_id: int) -> Element:
        if not (0 <= e_id < len(self.elements)):
            raise IndexError("element id out of range")
        return self.elements[e_id]

    def get_element_node_ids(self, e_id: int) -> Tuple[int, int]:
        e = self.get_element(e_id)
        return e.n0, e.n1

    def get_element_dof_indices(self, e_id: int) -> List[int]:
        e = self.get_element(e_id)
        n0d = self.global_dof_indices_for_node(e.n0)
        n1d = self.global_dof_indices_for_node(e.n1)
        return [n0d[0], n0d[1], n0d[2], n0d[3], n1d[0], n1d[1], n1d[2], n1d[3]]

    def get_element_dofs(self, e_id: int, u: Optional[Sequence[float]] = None) -> List[float]:
        dof_idx = self.get_element_dof_indices(e_id)
        src = self.u if u is None else u
        if len(src) < max(dof_idx) + 1:
            raise ValueError("Provided DOF vector is too small for the requested element")
        return [src[k] for k in dof_idx]

    # -------------------- counts & summaries --------------------
    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_elements(self) -> int:
        return len(self.elements)

    @property
    def num_horizontal_elements(self) -> int:
        return ((self.ny + 1) * self.nx) if self.include_horizontal else 0

    @property
    def num_vertical_elements(self) -> int:
        return ((self.nx + 1) * self.ny) if self.include_vertical else 0

    def summary(self) -> dict:
        return {
            "X": self.X,
            "Y": self.Y,
            "L": self.L,
            "nx": self.nx,
            "ny": self.ny,
            "num_nodes": self.num_nodes,
            "num_elements": self.num_elements,
            "num_horizontal_elements": self.num_horizontal_elements,
            "num_vertical_elements": self.num_vertical_elements,
        }


