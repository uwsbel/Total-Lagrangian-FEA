#include "mesh_utils.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ANCFCPUUtils {

// GridMeshGenerator implementation
GridMeshGenerator::GridMeshGenerator(double X, double Y, double L,
                                     bool include_horizontal,
                                     bool include_vertical)
    : X_(X),
      Y_(Y),
      L_(L),
      include_horizontal_(include_horizontal),
      include_vertical_(include_vertical) {
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
      double x = i * L_;
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

std::tuple<int, int, int, int> GridMeshGenerator::global_dof_indices_for_node(
    int node_id) {
  int base = 4 * node_id;
  return std::make_tuple(base, base + 1, base + 2, base + 3);
}

void GridMeshGenerator::get_coordinates(Eigen::VectorXd& x, Eigen::VectorXd& y,
                                        Eigen::VectorXd& z) {
  int n_nodes    = static_cast<int>(nodes_.size());
  int total_dofs = 4 * n_nodes;

  x.resize(total_dofs);
  y.resize(total_dofs);
  z.resize(total_dofs);

  // Define the pattern for each node (from beam_mesh_generator)
  std::vector<double> x_pattern = {
      1.0, 0.0, 0.0};  // [dx/du, dx/dv, dx/dw] for x-coordinate
  std::vector<double> y_pattern = {
      1.0, 0.0, 1.0, 0.0};  // [y, dx/du, dx/dv, dx/dw] for y-coordinate
  std::vector<double> z_pattern = {
      0.0, 0.0, 0.0, 1.0};  // [z, dx/du, dx/dv, dx/dw] for z-coordinate

  // Fill the arrays
  for (const auto& node : nodes_) {
    int idx = 4 * node.id;
    x(idx)  = node.x;  // x position
    for (int j = 1; j < 4; j++) {
      x(idx + j) = x_pattern[j - 1];  // dx/du, dx/dv, dx/dw for x
    }
    for (int j = 0; j < 4; j++) {
      y(idx + j) = y_pattern[j];  // y, dx/du, dx/dv, dx/dw for y
      z(idx + j) = z_pattern[j];  // z, dx/du, dx/dv, dx/dw for z
    }
  }
}

void GridMeshGenerator::get_element_connectivity(
    Eigen::MatrixXi& connectivity) {
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
  result["X"]            = X_;
  result["Y"]            = Y_;
  result["L"]            = L_;
  result["nx"]           = nx_;
  result["ny"]           = ny_;
  result["num_nodes"]    = get_num_nodes();
  result["num_elements"] = get_num_elements();
  result["num_horizontal_elements"] =
      include_horizontal_ ? ((ny_ + 1) * nx_) : 0;
  result["num_vertical_elements"] = include_vertical_ ? ((nx_ + 1) * ny_) : 0;
  return result;
}

// ============================================================
// LinearConstraintBuilder
// ============================================================

LinearConstraintBuilder::LinearConstraintBuilder(int n_dofs) : n_dofs_(n_dofs) {
  if (n_dofs_ <= 0) {
    throw std::invalid_argument("LinearConstraintBuilder: n_dofs must be > 0");
  }
  offsets_.push_back(0);
}

LinearConstraintBuilder::LinearConstraintBuilder(
    int n_dofs, const LinearConstraintCSR& initial)
    : n_dofs_(n_dofs),
      offsets_(initial.offsets),
      columns_(initial.columns),
      values_(initial.values),
      rhs_(static_cast<size_t>(initial.rhs.size()), 0.0) {
  if (n_dofs_ <= 0) {
    throw std::invalid_argument("LinearConstraintBuilder: n_dofs must be > 0");
  }
  if (static_cast<int>(offsets_.size()) != initial.NumRows() + 1) {
    throw std::invalid_argument(
        "LinearConstraintBuilder: initial offsets size mismatch");
  }
  if (static_cast<int>(columns_.size()) != initial.NumNonZeros() ||
      static_cast<int>(values_.size()) != initial.NumNonZeros()) {
    throw std::invalid_argument(
        "LinearConstraintBuilder: initial nnz mismatch");
  }
  if (!rhs_.empty()) {
    Eigen::Map<Eigen::VectorXd>(rhs_.data(), initial.rhs.size()) = initial.rhs;
  }
  if (offsets_.empty() || offsets_.front() != 0 ||
      offsets_.back() != static_cast<int>(columns_.size())) {
    throw std::invalid_argument(
        "LinearConstraintBuilder: initial CSR offsets invalid");
  }
}

int LinearConstraintBuilder::AddRow(
    const std::vector<std::pair<int, double>>& entries, double rhs) {
  if (entries.empty()) {
    throw std::invalid_argument("LinearConstraintBuilder::AddRow: empty row");
  }

  for (const auto& [col, val] : entries) {
    if (col < 0 || col >= n_dofs_) {
      throw std::out_of_range(
          "LinearConstraintBuilder::AddRow: col out of range");
    }
    if (val == 0.0)
      continue;
    columns_.push_back(col);
    values_.push_back(val);
  }

  rhs_.push_back(rhs);
  offsets_.push_back(static_cast<int>(columns_.size()));
  return static_cast<int>(rhs_.size()) - 1;
}

int LinearConstraintBuilder::AddFixedDof(int col, double rhs) {
  return AddRow({{col, 1.0}}, rhs);
}

LinearConstraintCSR LinearConstraintBuilder::ToCSR() const {
  LinearConstraintCSR out;
  out.offsets = offsets_;
  out.columns = columns_;
  out.values  = values_;
  out.rhs     = Eigen::VectorXd::Zero(static_cast<int>(rhs_.size()));
  if (!rhs_.empty()) {
    out.rhs = Eigen::Map<const Eigen::VectorXd>(rhs_.data(),
                                                static_cast<int>(rhs_.size()));
  }
  return out;
}

// ============================================================
// ANCF3243 constraints helpers
// ============================================================

namespace {

int ANCF3243DofCol(int node_id, int coef_slot, int component) {
  const int coef_index = node_id * 4 + coef_slot;
  return coef_index * 3 + component;
}

}  // namespace

void AppendANCF3243VectorEqualityConstraint(LinearConstraintBuilder& builder,
                                            int node_a, int node_b,
                                            int coef_slot) {
  if (coef_slot < 0 || coef_slot > 3) {
    throw std::out_of_range(
        "AppendANCF3243VectorEqualityConstraint: coef_slot out of range");
  }
  for (int c = 0; c < 3; ++c) {
    const int col_b = ANCF3243DofCol(node_b, coef_slot, c);
    const int col_a = ANCF3243DofCol(node_a, coef_slot, c);
    builder.AddRow({{col_b, 1.0}, {col_a, -1.0}}, 0.0);
  }
}

void AppendANCF3243VectorWeldedConstraint(LinearConstraintBuilder& builder,
                                          int node_a, int node_b, int coef_slot,
                                          const Eigen::Matrix3d& Q) {
  if (coef_slot < 0 || coef_slot > 3) {
    throw std::out_of_range(
        "AppendANCF3243VectorWeldedConstraint: coef_slot out of range");
  }

  for (int row = 0; row < 3; ++row) {
    std::vector<std::pair<int, double>> entries;
    entries.reserve(4);
    const int col_b = ANCF3243DofCol(node_b, coef_slot, row);
    entries.push_back({col_b, 1.0});
    for (int k = 0; k < 3; ++k) {
      const double w = -Q(row, k);
      if (w == 0.0)
        continue;
      const int col_a = ANCF3243DofCol(node_a, coef_slot, k);
      entries.push_back({col_a, w});
    }
    builder.AddRow(entries, 0.0);
  }
}

void AppendANCF3243FixedCoefficient(LinearConstraintBuilder& builder,
                                    int coef_index,
                                    const Eigen::VectorXd& x12_ref,
                                    const Eigen::VectorXd& y12_ref,
                                    const Eigen::VectorXd& z12_ref) {
  if (coef_index < 0 || coef_index >= x12_ref.size() ||
      coef_index >= y12_ref.size() || coef_index >= z12_ref.size()) {
    throw std::out_of_range(
        "AppendANCF3243FixedCoefficient: coef_index out of range");
  }

  builder.AddFixedDof(coef_index * 3 + 0, x12_ref(coef_index));
  builder.AddFixedDof(coef_index * 3 + 1, y12_ref(coef_index));
  builder.AddFixedDof(coef_index * 3 + 2, z12_ref(coef_index));
}

// ============================================================
// ANCF3243 mesh reader (.ancf3243mesh)
// ============================================================

namespace {

std::string StripComment(const std::string& line) {
  const size_t pos = line.find('#');
  if (pos == std::string::npos)
    return line;
  return line.substr(0, pos);
}

void TrimInPlace(std::string& s) {
  auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                  [&](char ch) { return !is_space(ch); }));
  s.erase(
      std::find_if(s.rbegin(), s.rend(), [&](char ch) { return !is_space(ch); })
          .base(),
      s.end());
}

bool ReadNextRecord(std::ifstream& file, std::string& out_line) {
  std::string line;
  while (std::getline(file, line)) {
    line = StripComment(line);
    TrimInPlace(line);
    if (line.empty())
      continue;
    out_line = line;
    return true;
  }
  return false;
}

bool ParseIntStrict(const std::string& s, int& out) {
  try {
    size_t idx = 0;
    int v      = std::stoi(s, &idx);
    if (idx != s.size())
      return false;
    out = v;
    return true;
  } catch (...) {
    return false;
  }
}

bool ParseDoubleStrict(const std::string& s, double& out) {
  try {
    size_t idx = 0;
    double v   = std::stod(s, &idx);
    if (idx != s.size())
      return false;
    out = v;
    return true;
  } catch (...) {
    return false;
  }
}

std::vector<std::string> Tokenize(const std::string& s) {
  std::vector<std::string> tokens;
  std::istringstream iss(s);
  std::string w;
  while (iss >> w)
    tokens.push_back(w);
  return tokens;
}

void SetError(std::string* err, const std::string& msg) {
  if (err)
    *err = msg;
}

}  // namespace

bool ReadANCF3243MeshFromFile(const std::string& path, ANCF3243Mesh& out,
                              std::string* error) {
  out = ANCF3243Mesh();

  std::ifstream file(path);
  if (!file.is_open()) {
    SetError(error, "ReadANCF3243MeshFromFile: failed to open " + path);
    return false;
  }

  std::string line;
  if (!ReadNextRecord(file, line)) {
    SetError(error, "ReadANCF3243MeshFromFile: empty file");
    return false;
  }
  {
    const auto t = Tokenize(line);
    if (t.size() != 2 || t[0] != "ancf3243_mesh") {
      SetError(error,
               "ReadANCF3243MeshFromFile: expected header 'ancf3243_mesh "
               "<version>'");
      return false;
    }
    int version = 0;
    if (!ParseIntStrict(t[1], version) || version <= 0) {
      SetError(error, "ReadANCF3243MeshFromFile: invalid mesh version");
      return false;
    }
    out.version = version;
  }

  // Optional grid metadata line.
  std::streampos pos_after_header = file.tellg();
  if (ReadNextRecord(file, line)) {
    const auto t = Tokenize(line);
    if (!t.empty() && t[0] == "grid") {
      // Expected:
      // grid nx <nx> ny <ny> L <L> origin <ox> <oy> <oz>
      if (t.size() != 11 || t[1] != "nx" || t[3] != "ny" || t[5] != "L" ||
          t[7] != "origin") {
        SetError(error, "ReadANCF3243MeshFromFile: invalid grid line");
        return false;
      }
      int nx = 0, ny = 0;
      double L = 0.0, ox = 0.0, oy = 0.0, oz = 0.0;
      if (!ParseIntStrict(t[2], nx) || !ParseIntStrict(t[4], ny) ||
          !ParseDoubleStrict(t[6], L) || !ParseDoubleStrict(t[8], ox) ||
          !ParseDoubleStrict(t[9], oy) || !ParseDoubleStrict(t[10], oz)) {
        SetError(error,
                 "ReadANCF3243MeshFromFile: failed to parse grid values");
        return false;
      }
      out.grid_nx     = nx;
      out.grid_ny     = ny;
      out.grid_L      = L;
      out.grid_origin = Eigen::Vector3d(ox, oy, oz);
    } else {
      // Not a grid line; rewind and treat it as the next section header.
      file.clear();
      file.seekg(pos_after_header);
    }
  }

  // nodes N
  if (!ReadNextRecord(file, line)) {
    SetError(error, "ReadANCF3243MeshFromFile: missing nodes section");
    return false;
  }
  int n_nodes = 0;
  {
    const auto t = Tokenize(line);
    if (t.size() != 2 || t[0] != "nodes" || !ParseIntStrict(t[1], n_nodes) ||
        n_nodes <= 0) {
      SetError(error, "ReadANCF3243MeshFromFile: invalid nodes header");
      return false;
    }
  }

  out.n_nodes = n_nodes;
  out.node_family.assign(static_cast<size_t>(n_nodes), "");
  out.x12.resize(4 * n_nodes);
  out.y12.resize(4 * n_nodes);
  out.z12.resize(4 * n_nodes);
  std::vector<bool> seen_node(static_cast<size_t>(n_nodes), false);

  for (int i = 0; i < n_nodes; ++i) {
    if (!ReadNextRecord(file, line)) {
      SetError(error, "ReadANCF3243MeshFromFile: unexpected EOF in nodes");
      return false;
    }
    const auto t = Tokenize(line);
    if (t.size() != 14) {
      SetError(
          error,
          "ReadANCF3243MeshFromFile: invalid node line (expected 14 tokens)");
      return false;
    }
    int node_id = -1;
    if (!ParseIntStrict(t[0], node_id) || node_id < 0 || node_id >= n_nodes) {
      SetError(error, "ReadANCF3243MeshFromFile: node id out of range");
      return false;
    }
    if (seen_node[static_cast<size_t>(node_id)]) {
      SetError(error, "ReadANCF3243MeshFromFile: duplicate node id");
      return false;
    }
    seen_node[static_cast<size_t>(node_id)] = true;

    out.node_family[static_cast<size_t>(node_id)] = t[1];

    double vals[12];
    for (int k = 0; k < 12; ++k) {
      if (!ParseDoubleStrict(t[2 + k], vals[k])) {
        SetError(error, "ReadANCF3243MeshFromFile: failed to parse node dofs");
        return false;
      }
    }

    const int base    = 4 * node_id;
    out.x12(base + 0) = vals[0];
    out.x12(base + 1) = vals[1];
    out.x12(base + 2) = vals[2];
    out.x12(base + 3) = vals[3];

    out.y12(base + 0) = vals[4];
    out.y12(base + 1) = vals[5];
    out.y12(base + 2) = vals[6];
    out.y12(base + 3) = vals[7];

    out.z12(base + 0) = vals[8];
    out.z12(base + 1) = vals[9];
    out.z12(base + 2) = vals[10];
    out.z12(base + 3) = vals[11];
  }

  for (bool ok : seen_node) {
    if (!ok) {
      SetError(error, "ReadANCF3243MeshFromFile: missing node id(s)");
      return false;
    }
  }

  // elements M
  if (!ReadNextRecord(file, line)) {
    SetError(error, "ReadANCF3243MeshFromFile: missing elements section");
    return false;
  }
  int n_elements = 0;
  {
    const auto t = Tokenize(line);
    if (t.size() != 2 || t[0] != "elements" ||
        !ParseIntStrict(t[1], n_elements) || n_elements <= 0) {
      SetError(error, "ReadANCF3243MeshFromFile: invalid elements header");
      return false;
    }
  }

  out.n_elements = n_elements;
  out.element_connectivity.resize(n_elements, 2);
  std::vector<bool> seen_elem(static_cast<size_t>(n_elements), false);

  for (int i = 0; i < n_elements; ++i) {
    if (!ReadNextRecord(file, line)) {
      SetError(error, "ReadANCF3243MeshFromFile: unexpected EOF in elements");
      return false;
    }
    const auto t = Tokenize(line);
    if (t.size() != 4) {
      SetError(
          error,
          "ReadANCF3243MeshFromFile: invalid element line (expected 4 tokens)");
      return false;
    }
    int elem_id = -1;
    int n0 = -1, n1 = -1;
    if (!ParseIntStrict(t[0], elem_id) || elem_id < 0 ||
        elem_id >= n_elements) {
      SetError(error, "ReadANCF3243MeshFromFile: element id out of range");
      return false;
    }
    if (seen_elem[static_cast<size_t>(elem_id)]) {
      SetError(error, "ReadANCF3243MeshFromFile: duplicate element id");
      return false;
    }
    seen_elem[static_cast<size_t>(elem_id)] = true;

    if (!ParseIntStrict(t[2], n0) || !ParseIntStrict(t[3], n1) || n0 < 0 ||
        n1 < 0 || n0 >= n_nodes || n1 >= n_nodes) {
      SetError(error, "ReadANCF3243MeshFromFile: element node id out of range");
      return false;
    }
    out.element_connectivity(elem_id, 0) = n0;
    out.element_connectivity(elem_id, 1) = n1;
  }

  for (bool ok : seen_elem) {
    if (!ok) {
      SetError(error, "ReadANCF3243MeshFromFile: missing element id(s)");
      return false;
    }
  }

  // constraints K (optional)
  if (!ReadNextRecord(file, line)) {
    // No constraints section: treat as empty constraints.
    out.constraints = LinearConstraintCSR{};
    return true;
  }

  int n_constraints_header = 0;
  {
    const auto t = Tokenize(line);
    if (t.size() != 2 || t[0] != "constraints" ||
        !ParseIntStrict(t[1], n_constraints_header) ||
        n_constraints_header < 0) {
      SetError(error, "ReadANCF3243MeshFromFile: invalid constraints header");
      return false;
    }
  }

  const int n_coef = 4 * n_nodes;
  const int n_dofs = n_coef * 3;
  LinearConstraintBuilder builder(n_dofs);

  for (int i = 0; i < n_constraints_header; ++i) {
    if (!ReadNextRecord(file, line)) {
      SetError(error,
               "ReadANCF3243MeshFromFile: unexpected EOF in constraints");
      return false;
    }
    const auto t = Tokenize(line);
    if (t.empty()) {
      SetError(error, "ReadANCF3243MeshFromFile: empty constraint line");
      return false;
    }
    if (t[0] == "pinned") {
      if (t.size() != 3) {
        SetError(error,
                 "ReadANCF3243MeshFromFile: pinned expects 'pinned a b'");
        return false;
      }
      int a = -1, b = -1;
      if (!ParseIntStrict(t[1], a) || !ParseIntStrict(t[2], b) || a < 0 ||
          b < 0 || a >= n_nodes || b >= n_nodes) {
        SetError(error,
                 "ReadANCF3243MeshFromFile: pinned node id out of range");
        return false;
      }
      AppendANCF3243VectorEqualityConstraint(builder, a, b, /*coef_slot=*/0);
    } else if (t[0] == "welded") {
      if (t.size() != 12) {
        SetError(
            error,
            "ReadANCF3243MeshFromFile: welded expects 'welded a b q00..q22'");
        return false;
      }
      int a = -1, b = -1;
      if (!ParseIntStrict(t[1], a) || !ParseIntStrict(t[2], b) || a < 0 ||
          b < 0 || a >= n_nodes || b >= n_nodes) {
        SetError(error,
                 "ReadANCF3243MeshFromFile: welded node id out of range");
        return false;
      }
      double q[9];
      for (int k = 0; k < 9; ++k) {
        if (!ParseDoubleStrict(t[3 + k], q[k])) {
          SetError(error, "ReadANCF3243MeshFromFile: welded failed to parse Q");
          return false;
        }
      }
      Eigen::Matrix3d Q;
      Q << q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];

      // Position continuity (no rotation).
      AppendANCF3243VectorEqualityConstraint(builder, a, b, /*coef_slot=*/0);
      // Gradient continuity with Q mapping.
      AppendANCF3243VectorWeldedConstraint(builder, a, b, /*coef_slot=*/1, Q);
      AppendANCF3243VectorWeldedConstraint(builder, a, b, /*coef_slot=*/2, Q);
      AppendANCF3243VectorWeldedConstraint(builder, a, b, /*coef_slot=*/3, Q);
    } else {
      SetError(error, "ReadANCF3243MeshFromFile: unknown constraint type '" +
                          t[0] + "'");
      return false;
    }
  }

  out.constraints = builder.ToCSR();
  return true;
}

}  // namespace ANCFCPUUtils
