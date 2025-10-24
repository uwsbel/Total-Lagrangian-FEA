#pragma once

#include <Eigen/Dense>
#include <string>

namespace ANCFCPUUtils {

/**
 * Construct the B matrix for ANCF3243 elements and compute its inverse
 * transpose
 * @param L Length parameter of the element
 * @param W Width parameter of the element
 * @param H Height parameter of the element
 * @param B_inv_out Output matrix (inverse transpose of B matrix)
 * @param n_shape Number of shape functions (16 for ANCF3243)
 */
void ANCF3243_B12_matrix(double L, double W, double H,
                         Eigen::MatrixXd &B_inv_out, int n_shape);

/**
 * Generate 3D beam element coordinates for multiple ANCF3243 elements
 * @param n_beam Number of 3D elements
 * @param x12 Output x coordinates for all nodes
 * @param y12 Output y coordinates for all nodes
 * @param z12 Output z coordinates for all nodes
 */
void ANCF3243_generate_beam_coordinates(int n_beam, Eigen::VectorXd &x12,
                                        Eigen::VectorXd &y12,
                                        Eigen::VectorXd &z12);

/**
 * Calculate offset indices for ANCF3243 elements
 * @param n_beam Number of 3D elements
 * @param offset_start Output start indices for each element's nodes
 * @param offset_end Output end indices for each element's nodes
 */
void ANCF3243_calculate_offsets(int n_beam, Eigen::VectorXi &offset_start,
                                Eigen::VectorXi &offset_end);

/**
 * Construct the B matrix for ANCF3443 elements and compute its inverse
 * transpose
 * @param L Length parameter of the element
 * @param W Width parameter of the element
 * @param H Height parameter of the element
 * @param B_inv_out Output matrix (inverse transpose of B matrix)
 * @param n_shape Number of shape functions (16 for ANCF3443)
 */
void ANCF3443_B12_matrix(double L, double W, double H,
                         Eigen::MatrixXd &B_inv_out, int n_shape);

/**
 * Generate 3D element coordinates for multiple ANCF3443 elements
 * @param n_beam Number of 3D elements
 * @param x12 Output x coordinates for all nodes
 * @param y12 Output y coordinates for all nodes
 * @param z12 Output z coordinates for all nodes
 */
void ANCF3443_generate_beam_coordinates(int n_beam, Eigen::VectorXd &x12,
                                        Eigen::VectorXd &y12,
                                        Eigen::VectorXd &z12,
                                        Eigen::MatrixXi &element_connectivity);

/**
 * Calculate offset indices for ANCF3443 shell elements
 * @param n_beam Number of 3D elements
 * @param offset_start Output start indices for each element's nodes
 * @param offset_end Output end indices for each element's nodes
 */
void ANCF3443_calculate_offsets(int n_beam, Eigen::VectorXi &offset_start,
                                Eigen::VectorXi &offset_end);

/**
 * Remap TetGen T10 tetrahedral element indices to standard order
 * @param tetgen_elem Input array with TetGen node ordering (size 10)
 * @param standard_elem Output array with standard node ordering (size 10)
 */
void FEAT10_remap_tetgen_indices(const Eigen::VectorXi &tetgen_elem,
                                 Eigen::VectorXi &standard_elem);

/**
 * Read node coordinates from TetGen .node file
 * @param filename Path to the .node file
 * @param nodes Output matrix with node coordinates (n_nodes × 3)
 * @return Number of nodes read
 */
int FEAT10_read_nodes(const std::string &filename, Eigen::MatrixXd &nodes);

/**
 * Read element connectivity from TetGen .ele file
 * @param filename Path to the .ele file
 * @param elements Output matrix with element connectivity (n_elements × 10)
 * @return Number of elements read
 */
int FEAT10_read_elements(const std::string &filename,
                         Eigen::MatrixXi &elements);

/**
 * MeshGenerator class for generating beam mesh coordinates and DOF mappings
 */
class MeshGenerator {
public:
    /**
     * Constructor for beam mesh generator
     * @param length Beam length along x-axis
     * @param width Beam width
     * @param height Beam height
     * @param start_x Starting x position of first beam
     * @param n_beams Number of beam elements
     */
    MeshGenerator(double length, double width, double height, 
                  double start_x, int n_beams);
    
    /**
     * Generate nodal coordinates for all beams
     */
    void generate_coordinates();
    
    /**
     * Get generated coordinates
     * @param x Output x coordinates
     * @param y Output y coordinates  
     * @param z Output z coordinates
     */
    void get_coordinates(Eigen::VectorXd& x, Eigen::VectorXd& y, Eigen::VectorXd& z);
    
    /**
     * Get DOF ranges for all beams
     * @param start Output start indices for each beam's DOFs
     * @param end Output end indices for each beam's DOFs
     */
    void get_dof_ranges(Eigen::VectorXi& start, Eigen::VectorXi& end);
    
    /**
     * Get total number of DOFs
     * @return Total number of DOFs
     */
    int get_total_dofs() const;
    
    /**
     * Get DOF range for a specific beam
     * @param beam_id Beam index
     * @return Pair of (start_dof, end_dof)
     */
    std::pair<int, int> get_beam_dof_range(int beam_id) const;

private:
    double length_, width_, height_;
    double start_x_;
    int n_beams_;
    
    std::vector<double> x_coords_, y_coords_, z_coords_;
    std::vector<std::pair<int, int>> dof_ranges_;
    int total_dofs_;
};

}  // namespace ANCFCPUUtils