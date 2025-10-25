#include "mesh_utils.h"

#include <cmath>
#include <stdexcept>
#include <iostream>

namespace ANCFCPUUtils {

// MeshGenerator implementation
MeshGenerator::MeshGenerator(double length, double width, double height, 
                             double start_x, int n_beams)
    : length_(length), width_(width), height_(height), start_x_(start_x), n_beams_(n_beams) {
  if (n_beams < 1) {
    throw std::invalid_argument("Number of beams must be >= 1");
  }
  if (length <= 0) {
    throw std::invalid_argument("Beam length must be > 0");
  }
}

void MeshGenerator::generate_coordinates() {
  // Generate node positions along x-axis
  std::vector<double> node_x;
  
  // First node (start of first beam)
  node_x.push_back(start_x_ - length_ / 2.0);
  
  // Add nodes for each beam
  for (int i = 0; i < n_beams_; i++) {
    double beam_center = start_x_ + i * length_;
    node_x.push_back(beam_center + length_ / 2.0);
  }
  
  int n_nodes = node_x.size();
  total_dofs_ = 4 * n_nodes;
  
  // Resize coordinate arrays
  x_coords_.resize(total_dofs_);
  y_coords_.resize(total_dofs_);
  z_coords_.resize(total_dofs_);
  
  // Define the pattern for each node (4 DOFs per node)
  // Pattern: [x, dx/du, dx/dv, dx/dw] for each coordinate
  std::vector<double> x_pattern = {1.0, 1.0, 0.0, 0.0};
  std::vector<double> y_pattern = {1.0, 0.0, 1.0, 0.0};
  std::vector<double> z_pattern = {0.0, 0.0, 0.0, 1.0};
  
  // Fill the coordinate arrays
  for (int i = 0; i < n_nodes; i++) {
    int idx = 4 * i;
    
    // X coordinates
    x_coords_[idx] = node_x[i];
    for (int j = 1; j < 4; j++) {
      x_coords_[idx + j] = x_pattern[j];
    }
    
    // Y coordinates
    for (int j = 0; j < 4; j++) {
      y_coords_[idx + j] = y_pattern[j];
    }
    
    // Z coordinates
    for (int j = 0; j < 4; j++) {
      z_coords_[idx + j] = z_pattern[j];
    }
  }
  
  // Generate DOF ranges for each beam
  dof_ranges_.clear();
  for (int i = 0; i < n_beams_; i++) {
    int start_dof = i * 4;
    int end_dof = start_dof + 7;
    dof_ranges_.push_back({start_dof, end_dof});
  }
}

void MeshGenerator::get_coordinates(Eigen::VectorXd& x, Eigen::VectorXd& y, Eigen::VectorXd& z) {
  x.resize(total_dofs_);
  y.resize(total_dofs_);
  z.resize(total_dofs_);
  
  for (int i = 0; i < total_dofs_; i++) {
    x(i) = x_coords_[i];
    y(i) = y_coords_[i];
    z(i) = z_coords_[i];
  }
}

void MeshGenerator::get_dof_ranges(Eigen::VectorXi& start, Eigen::VectorXi& end) {
  start.resize(n_beams_);
  end.resize(n_beams_);
  
  for (int i = 0; i < n_beams_; i++) {
    start(i) = dof_ranges_[i].first;
    end(i) = dof_ranges_[i].second;
  }
}

int MeshGenerator::get_total_dofs() const {
  return total_dofs_;
}

std::pair<int, int> MeshGenerator::get_beam_dof_range(int beam_id) const {
  if (beam_id < 0 || beam_id >= n_beams_) {
    throw std::invalid_argument("Invalid beam_id");
  }
  return dof_ranges_[beam_id];
}

// GridMeshGenerator implementation
GridMeshGenerator::GridMeshGenerator(double X, double Y, double L, 
                                     bool include_horizontal, bool include_vertical)
    : X_(X), Y_(Y), L_(L), include_horizontal_(include_horizontal), include_vertical_(include_vertical) {
    if (L_ <= 0) {
        throw std::invalid_argument("L must be > 0");
    }
    
    // Check if X and Y are exact multiples of L
    if (std::abs(std::round(X_ / L_) * L_ - X_) > 1e-12 || 
        std::abs(std::round(Y_ / L_) * L_ - Y_) > 1e-12) {
        throw std::invalid_argument("X and Y must be exact multiples of L");
    }
    
    nx_ = static_cast<int>(std::round(X_ / L_));  // number of intervals in x
    ny_ = static_cast<int>(std::round(Y_ / L_));  // number of intervals in y

    std::cout << "nx: " << nx_ << ", ny: " << ny_ << std::endl;
    
    // If flags are not provided, infer from geometry
    if (include_horizontal_ && nx_ == 0) {
        include_horizontal_ = false;
    }
    if (include_vertical_ && ny_ == 0) {
        include_vertical_ = false;
    }
}

void GridMeshGenerator::generate_mesh() {
    generate_nodes();
    generate_elements();
}

void GridMeshGenerator::generate_nodes() {
    nodes_.clear();
    int node_id_counter = 0;
    
    for (int j = 0; j <= ny_; j++) {  // row-major by j then i
        double y = j * L_;
        for (int i = 0; i <= nx_; i++) {
            double x = -1.0 + i * L_;  // Start at -1.0 instead of 0.0 (TODO: change this to 0.0)
            nodes_.push_back({node_id_counter, i, j, x, y, 0.0, 0.0, 0.0, 0.0});
            node_id_counter++;
        }
    }
}

void GridMeshGenerator::generate_elements() {
    elements_.clear();
    int eid = 0;
    
    // Horizontals first: for each row j, elements between (i,j) -> (i+1,j)
    if (include_horizontal_) {
        for (int j = 0; j <= ny_; j++) {
            for (int i = 0; i < nx_; i++) {
                int n0 = node_id(i, j);
                int n1 = node_id(i + 1, j);
                elements_.push_back({eid, n0, n1, "H", L_});
                eid++;
            }
        }
    }
    
    // Verticals next: for each column i, elements between (i,j) -> (i,j+1)
    if (include_vertical_) {
        for (int i = 0; i <= nx_; i++) {
            for (int j = 0; j < ny_; j++) {
                int n0 = node_id(i, j);
                int n1 = node_id(i, j + 1);
                elements_.push_back({eid, n0, n1, "V", L_});
                eid++;
            }
        }
    }
}

int GridMeshGenerator::node_id(int i, int j) const {
    if (i < 0 || i > nx_ || j < 0 || j > ny_) {
        throw std::out_of_range("(i,j) out of range");
    }
    return j * (nx_ + 1) + i;
}

std::tuple<int, int, int, int> GridMeshGenerator::global_dof_indices_for_node(int node_id) {
    int base = 4 * node_id;
    return std::make_tuple(base, base + 1, base + 2, base + 3);
}

void GridMeshGenerator::get_coordinates(Eigen::VectorXd& x, Eigen::VectorXd& y, Eigen::VectorXd& z) {
    int n_nodes = static_cast<int>(nodes_.size());
    int total_dofs = 4 * n_nodes;
    
    x.resize(total_dofs);
    y.resize(total_dofs);
    z.resize(total_dofs);
    
    // Define the pattern for each node (from beam_mesh_generator)
    std::vector<double> x_pattern = {1.0, 0.0, 0.0};  // [dx/du, dx/dv, dx/dw] for x-coordinate
    std::vector<double> y_pattern = {1.0, 0.0, 1.0, 0.0};  // [y, dx/du, dx/dv, dx/dw] for y-coordinate  
    std::vector<double> z_pattern = {0.0, 0.0, 0.0, 1.0};  // [z, dx/du, dx/dv, dx/dw] for z-coordinate
    
    // Fill the arrays
    for (const auto& node : nodes_) {
        int idx = 4 * node.id;
        x(idx) = node.x;  // x position
        for (int j = 1; j < 4; j++) {
            x(idx + j) = x_pattern[j - 1];  // dx/du, dx/dv, dx/dw for x
        }
        for (int j = 0; j < 4; j++) {
            y(idx + j) = y_pattern[j];  // y, dx/du, dx/dv, dx/dw for y
            z(idx + j) = z_pattern[j];  // z, dx/du, dx/dv, dx/dw for z
        }
    }
}

void GridMeshGenerator::get_element_connectivity(Eigen::MatrixXi& connectivity) {
    int n_elements = static_cast<int>(elements_.size());
    connectivity.resize(n_elements, 2);
    
    for (int i = 0; i < n_elements; i++) {
        connectivity(i, 0) = elements_[i].n0;
        connectivity(i, 1) = elements_[i].n1;
    }
}

int GridMeshGenerator::get_num_nodes() const {
    return static_cast<int>(nodes_.size());
}

int GridMeshGenerator::get_num_elements() const {
    return static_cast<int>(elements_.size());
}

std::map<std::string, double> GridMeshGenerator::summary() const {
    std::map<std::string, double> result;
    result["X"] = X_;
    result["Y"] = Y_;
    result["L"] = L_;
    result["nx"] = nx_;
    result["ny"] = ny_;
    result["num_nodes"] = get_num_nodes();
    result["num_elements"] = get_num_elements();
    result["num_horizontal_elements"] = include_horizontal_ ? ((ny_ + 1) * nx_) : 0;
    result["num_vertical_elements"] = include_vertical_ ? ((nx_ + 1) * ny_) : 0;
    return result;
}

}  // namespace ANCFCPUUtils

