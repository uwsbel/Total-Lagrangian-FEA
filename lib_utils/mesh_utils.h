#pragma once

#include <Eigen/Dense>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace ANCFCPUUtils {

/**
 * Node structure for GridMesh
 */
struct GridNode {
  int id;
  int i, j;  // grid coordinates
  double x, y;
  double dof_x, dof_dx_du, dof_dx_dv, dof_dx_dw;
};

/**
 * Element structure for GridMesh
 */
struct GridElement {
  int id;
  int n0, n1;               // node IDs
  std::string orientation;  // "H" or "V"
  double length;
};

/**
 * GridMeshGenerator class for generating structured grid meshes
 */
class GridMeshGenerator {
 public:
  /**
   * Constructor for grid mesh generator
   * @param X Domain width in x-direction
   * @param Y Domain height in y-direction
   * @param L Element length (spacing)
   * @param include_horizontal Whether to include horizontal elements
   * @param include_vertical Whether to include vertical elements
   */
  GridMeshGenerator(double X, double Y, double L,
                    bool include_horizontal = true,
                    bool include_vertical   = true);

  /**
   * Generate the mesh (nodes and elements)
   */
  void generate_mesh();

  /**
   * Get generated coordinates
   * @param x Output x coordinates (4 DOFs per node)
   * @param y Output y coordinates (4 DOFs per node)
   * @param z Output z coordinates (4 DOFs per node)
   */
  void get_coordinates(Eigen::VectorXd& x, Eigen::VectorXd& y,
                       Eigen::VectorXd& z);

  /**
   * Get element connectivity matrix
   * @param connectivity Output connectivity matrix (n_elements × 2)
   */
  void get_element_connectivity(Eigen::MatrixXi& connectivity);

  /**
   * Get number of nodes
   * @return Number of nodes
   */
  int get_num_nodes() const;

  /**
   * Get number of elements
   * @return Number of elements
   */
  int get_num_elements() const;

  /**
   * Get mesh summary
   * @return Summary dictionary
   */
  std::map<std::string, double> summary() const;

 private:
  double X_, Y_, L_;
  bool include_horizontal_, include_vertical_;
  int nx_, ny_;  // number of intervals in x and y
  std::vector<GridNode> nodes_;
  std::vector<GridElement> elements_;

  void generate_nodes();
  void generate_elements();
  int node_id(int i, int j) const;
  static std::tuple<int, int, int, int> global_dof_indices_for_node(
      int node_id);
};

}  // namespace ANCFCPUUtils
