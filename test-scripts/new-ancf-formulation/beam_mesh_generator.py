import numpy as np

class BeamConfig:
    """Configuration class for beam properties"""
    def __init__(self, length, width, height, position=(0, 0, 0), orientation='x'):
        """
        Initialize beam configuration
        
        Args:
            length: Beam length along primary axis
            width: Beam width
            height: Beam height
            position: Center position (x, y, z) of the beam
            orientation: Primary axis direction ('x', 'y', or 'z')
        """
        self.length = length
        self.width = width
        self.height = height
        self.position = position
        self.orientation = orientation
        
    def validate(self):
        """Validate beam configuration parameters"""
        if self.length <= 0:
            raise ValueError("Beam length must be > 0")
        if self.width <= 0:
            raise ValueError("Beam width must be > 0")
        if self.height <= 0:
            raise ValueError("Beam height must be > 0")
        if self.orientation not in ['x', 'y', 'z']:
            raise ValueError("Orientation must be 'x', 'y', or 'z'")
        return True


class BeamMeshGenerator:
    """Generator for automated mesh creation of multiple beams"""
    
    def __init__(self, n_beams, beam_config, spacing=2.0):
        """
        Initialize mesh generator
        
        Args:
            n_beams: Number of beam elements
            beam_config: BeamConfig object defining beam properties
            spacing: Distance between beam centers
        """
        self.n_beams = n_beams
        self.beam_config = beam_config
        self.spacing = spacing
        
        # Validate configuration
        self.beam_config.validate()
        if n_beams < 1:
            raise ValueError("Number of beams must be >= 1")
        if spacing < 0:
            raise ValueError("Spacing must be >= 0")
    
    def generate_beam_positions(self):
        """Generate center positions for all beams"""
        positions = []
        for i in range(self.n_beams):
            if self.beam_config.orientation == 'x':
                pos = (i * (self.beam_config.length + self.spacing), 0, 0)
            elif self.beam_config.orientation == 'y':
                pos = (0, i * (self.beam_config.length + self.spacing), 0)
            else:  # 'z'
                pos = (0, 0, i * (self.beam_config.length + self.spacing))
            positions.append(pos)
        return positions
    
    def _generate_local_beam_coords(self):
        """Generate local coordinates for a single beam"""
        L = self.beam_config.length
        W = self.beam_config.width
        H = self.beam_config.height
        
        # This replicates the exact coordinate pattern from 3-beam-debug.py
        # The pattern is: [x1, dx1/du, dx1/dv, dx1/dw, x2, dx2/du, dx2/dv, dx2/dw]
        # For the first beam: x1=-L/2, x2=L/2, derivatives are 0 or 1
        x_local = np.array([-L/2, 1.0, 0.0, 0.0, L/2, 1.0, 0.0, 0.0])
        y_local = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        z_local = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        
        return {'x': x_local, 'y': y_local, 'z': z_local}
    
    def generate_nodal_coordinates(self):
        """Generate all nodal coordinates for all beams"""
        # Start with the first beam coordinates (exactly as in original)
        x_coords = np.array([-1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        y_coords = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        z_coords = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        
        # Append additional beams using the same logic as original
        for i in range(2, self.n_beams + 1):  # MATLAB loop is inclusive
            x_offset = self.spacing
            
            # Only shift the first entry of the last beam by x_offset
            x_block = x_coords[-4:].copy()
            x_block[0] += x_offset
            
            # Last 4 entries for y and z (unchanged)
            y_block = y_coords[-4:]
            z_block = z_coords[-4:]
            
            # Append to the nodal arrays
            x_coords = np.concatenate([x_coords, x_block])
            y_coords = np.concatenate([y_coords, y_block])
            z_coords = np.concatenate([z_coords, z_block])
        
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
