import numpy as np

class BeamConfig:
    """Configuration class for beam properties"""
    def __init__(self, length, position, orientation='x'):
        """
        Initialize beam configuration
        
        Args:
            length: Beam length along primary axis
            position: Center position (x, y, z) of the first beam
            orientation: Primary axis direction ('x', 'y', or 'z')
        """
        self.length = length
        self.position = position
        self.orientation = orientation
        
    def validate(self):
        """Validate beam configuration parameters"""
        if self.length <= 0:
            raise ValueError("Beam length must be > 0")
        if self.orientation not in ['x', 'y', 'z']:
            raise ValueError("Orientation must be 'x', 'y', or 'z'")
        return True


class BeamMeshGenerator:
    """Generator for automated mesh creation of multiple beams"""
    
    def __init__(self, n_beams, beam_config):
        """
        Initialize mesh generator
        
        Args:
            n_beams: Number of beam elements
            beam_config: BeamConfig object defining beam properties
        """
        self.n_beams = n_beams
        self.beam_config = beam_config
        
        # Validate configuration
        self.beam_config.validate()
        if n_beams < 1:
            raise ValueError("Number of beams must be >= 1")
    
    def generate_beam_positions(self):
        """Generate center positions for all beams"""
        L = self.beam_config.length
        x0 = float(self.beam_config.position[0])
        positions = []
        for i in range(self.n_beams):
            if self.beam_config.orientation == 'x':
                pos = (x0 + i * L, 0, 0)
            positions.append(pos)
        return positions
    
    def generate_nodal_coordinates(self):
        """Generate all nodal coordinates for all beams"""
        L = self.beam_config.length
        centers = self.generate_beam_positions()

        # for each beam, there are two nodes
        # Each node has 4 DOFs (x, dx/du, dx/dv, dx/dw)
        # The nodal coordinates are:
        # x1, dx1/du, dx1/dv, dx1/dw, x2, dx2/du, dx2/dv, dx2/dw
        
        node_x = [centers[0][0] - L/2]
        for cx in centers:
            node_x.append(cx[0] + L/2)

        # Create arrays for nodal coordinates
        n_nodes = len(node_x)
        x_coords = np.zeros(4 * n_nodes, dtype=float)
        y_coords = np.zeros(4 * n_nodes, dtype=float)
        z_coords = np.zeros(4 * n_nodes, dtype=float)
        
        # Define the pattern for each node
        x_pattern = [1.0, 0.0, 0.0]
        y_pattern = [1.0, 0.0, 1.0, 0.0]
        z_pattern = [0.0, 0.0, 0.0, 1.0]
        
        # Fill the arrays
        for i, x in enumerate(node_x):
            idx = 4 * i
            x_coords[idx] = x
            x_coords[idx+1:idx+4] = x_pattern
            y_coords[idx:idx+4] = y_pattern
            z_coords[idx:idx+4] = z_pattern

        return x_coords, y_coords, z_coords
            
    def generate_dof_mapping(self):
        """Generate DOF mapping for all beams"""
        # The original code uses overlapping DOF ranges:
        # Beam 0: indices 0-7 (8 coordinates)
        # Beam 1: indices 4-11 (8 coordinates, with overlap)
        # This matches the original offset calculation: offset_start[i] = i * 4
        beam_dof_ranges = []
        
        for i in range(self.n_beams):
            start_dof = i * 4
            end_dof = start_dof + 7
            beam_dof_ranges.append((start_dof, end_dof))
        
        return beam_dof_ranges
    
    def get_total_dofs(self):
        """Get total number of DOFs for all beams"""
        x_coords, y_coords, z_coords = self.generate_nodal_coordinates()
        return len(x_coords)
