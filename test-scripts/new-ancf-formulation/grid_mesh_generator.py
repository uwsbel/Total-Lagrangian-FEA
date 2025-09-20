import numpy as np

class GridConfig:
    def __init__(self, nx, ny, element_length, element_width, element_height, 
                 spacing_x=0, spacing_y=0, position=(0, 0, 0)):
        """
        Configuration for 2D grid mesh
        
        Args:
            nx: Number of elements in x direction
            ny: Number of elements in y direction  
            element_length: Length of each beam element
            element_width: Width of each beam element
            element_height: Height of each beam element
            spacing_x: Spacing between elements in x direction
            spacing_y: Spacing between elements in y direction
            position: Bottom-left corner position of grid
        """
        self.nx = nx
        self.ny = ny
        self.element_length = element_length
        self.element_width = element_width
        self.element_height = element_height
        self.spacing_x = spacing_x
        self.spacing_y = spacing_y
        self.position = position
        
    def validate(self):
        """Validate grid configuration"""
        if self.nx < 0 or self.ny < 0:
            raise ValueError("Grid dimensions must be >= 0")
        if self.nx == 0 and self.ny == 0:
            raise ValueError("At least one grid dimension must be > 0")
        if self.element_length <= 0 or self.element_width <= 0 or self.element_height <= 0:
            raise ValueError("Element dimensions must be > 0")
        return True
    
    def get_total_nodes(self):
        """Get total number of nodes in grid"""
        if self.nx == 0:  # Pure vertical grid
            return self.ny + 1
        elif self.ny == 0:  # Pure horizontal grid
            return self.nx + 1
        else:  # 2D grid
            return (self.nx + 1) * (self.ny + 1)
    
    def get_total_elements(self):
        """Get total number of beam elements in grid"""
        if self.nx == 0:  # Pure vertical grid
            return self.ny
        elif self.ny == 0:  # Pure horizontal grid
            return self.nx
        else:  # 2D grid
            return self.nx * (self.ny + 1) + self.ny * (self.nx + 1)
    
    def get_total_dofs(self):
        """Get total number of DOFs (4 per node)"""
        return self.get_total_nodes() * 4


class GridMeshGenerator:
    def __init__(self, grid_config):
        self.config = grid_config
        self.config.validate()
    
    def generate_node_coordinates(self):
        """Generate coordinates for all nodes in the grid"""
        nodes = []
        nx, ny = self.config.nx, self.config.ny
        L, W, H = self.config.element_length, self.config.element_width, self.config.element_height
        sx, sy = self.config.spacing_x, self.config.spacing_y
        px, py, pz = self.config.position
        
        if nx == 0:  # Pure vertical grid
            for j in range(ny + 1):
                x = px
                y = py + j * (W + sy)
                z = pz
                nodes.append((x, y, z))
        elif ny == 0:  # Pure horizontal grid
            for i in range(nx + 1):
                x = px + i * (L + sx)
                y = py
                z = pz
                nodes.append((x, y, z))
        else:  # 2D grid
            for j in range(ny + 1):  # y direction (rows)
                for i in range(nx + 1):  # x direction (cols)
                    x = px + i * (L + sx)
                    y = py + j * (W + sy)  
                    z = pz
                    nodes.append((x, y, z))
        
        return np.array(nodes)
    
    def generate_element_connectivity(self):
        """Generate beam element connectivity matrix"""
        elements = []
        nx, ny = self.config.nx, self.config.ny
        
        if nx == 0:  # Pure vertical grid
            for j in range(ny):
                node1 = j
                node2 = j + 1
                elements.append((node1, node2, 'vertical'))
        elif ny == 0:  # Pure horizontal grid
            for i in range(nx):
                node1 = i
                node2 = i + 1
                elements.append((node1, node2, 'horizontal'))
        else:  # 2D grid
            # Horizontal beams (along x direction)
            for j in range(ny + 1):
                for i in range(nx):
                    node1 = j * (nx + 1) + i
                    node2 = j * (nx + 1) + i + 1
                    elements.append((node1, node2, 'horizontal'))
            
            # Vertical beams (along y direction)  
            for j in range(ny):
                for i in range(nx + 1):
                    node1 = j * (nx + 1) + i
                    node2 = (j + 1) * (nx + 1) + i
                    elements.append((node1, node2, 'vertical'))
        
        return elements
    
    def generate_dof_mapping(self):
        """Generate DOF mapping for each element"""
        elements = self.generate_element_connectivity()
        dof_mappings = []
        
        for node1, node2, orientation in elements:
            # Each node has 4 DOFs
            dof1_start = node1 * 4
            dof1_end = dof1_start + 3
            dof2_start = node2 * 4  
            dof2_end = dof2_start + 3
            dof_mappings.append({
                'element_nodes': (node1, node2),
                'dof_ranges': ((dof1_start, dof1_end), (dof2_start, dof2_end)),
                'orientation': orientation
            })
        
        return dof_mappings
    
    def get_element_dimensions(self, orientation):
        """Get element dimensions based on orientation"""
        if orientation == 'horizontal':
            return self.config.element_length, self.config.element_width, self.config.element_height
        else:  # vertical
            return self.config.element_width, self.config.element_length, self.config.element_height
