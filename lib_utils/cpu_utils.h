#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <set>

namespace ANCFCPUUtils {

/**
 * Build vertex adjacency graph from element connectivity
 * Two nodes are adjacent if they belong to the same element
 * @param element_connectivity Element connectivity matrix (n_elem × nodes_per_elem)
 * @param n_nodes Total number of nodes
 * @return Adjacency list for each node
 */
std::vector<std::set<int>> BuildVertexAdjacency(
    const Eigen::MatrixXi &element_connectivity, int n_nodes);

/**
 * Greedy vertex coloring for parallel VBD updates
 * Colors are assigned such that no two adjacent nodes have the same color
 * Uses degree ordering heuristic for better coloring
 * @param adjacency Adjacency list for each node
 * @return Color assignment for each node
 */
Eigen::VectorXi GreedyVertexColoring(
    const std::vector<std::set<int>> &adjacency);

/**
 * Validate that coloring is valid (no element has two nodes of same color)
 * @param element_connectivity Element connectivity matrix
 * @param colors Color assignment for each node
 * @return true if coloring is valid
 */
bool ValidateColoring(const Eigen::MatrixXi &element_connectivity,
                      const Eigen::VectorXi &colors);

/**
 * Build incidence list mapping each node to (element_idx, local_node_idx) pairs
 * @param element_connectivity Element connectivity matrix (n_elem × nodes_per_elem)
 * @param n_nodes Total number of nodes
 * @return For each node, a vector of (element_idx, local_node_idx) pairs
 */
std::vector<std::vector<std::pair<int, int>>> BuildNodeIncidence(
    const Eigen::MatrixXi &element_connectivity, int n_nodes);

/**
 * Organize nodes by color for VBD parallel processing
 * @param colors Color assignment for each node
 * @param n_colors Number of colors used
 * @return For each color, a vector of node indices
 */
std::vector<std::vector<int>> BuildColorToNodes(const Eigen::VectorXi &colors,
                                                 int n_colors);

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

}  // namespace ANCFCPUUtils