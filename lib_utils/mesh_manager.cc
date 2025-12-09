#include "mesh_manager.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "cpu_utils.h"

namespace ANCFCPUUtils {

// ============================================================================
// NPZ file reading utilities (minimal implementation)
// NPZ is a ZIP archive containing .npy files
// ============================================================================

namespace {

// Read a 4-byte little-endian uint32
uint32_t readUint32LE(std::istream& is) {
  uint8_t buf[4];
  is.read(reinterpret_cast<char*>(buf), 4);
  return buf[0] | (buf[1] << 8) | (buf[2] << 16) | (buf[3] << 24);
}

// Read a 2-byte little-endian uint16
uint16_t readUint16LE(std::istream& is) {
  uint8_t buf[2];
  is.read(reinterpret_cast<char*>(buf), 2);
  return buf[0] | (buf[1] << 8);
}

// Parse NPY header to get shape and dtype
bool parseNpyHeader(const std::vector<char>& data, size_t& offset,
                    std::vector<size_t>& shape, bool& is_double) {
  // NPY magic: \x93NUMPY
  if (data.size() < 10)
    return false;
  if (data[0] != '\x93' || data[1] != 'N' || data[2] != 'U' || data[3] != 'M' ||
      data[4] != 'P' || data[5] != 'Y') {
    return false;
  }

  uint8_t major = static_cast<uint8_t>(data[6]);
  // uint8_t minor = static_cast<uint8_t>(data[7]);

  uint32_t header_len;
  if (major == 1) {
    header_len =
        static_cast<uint8_t>(data[8]) | (static_cast<uint8_t>(data[9]) << 8);
    offset = 10 + header_len;
  } else {
    header_len = static_cast<uint8_t>(data[8]) |
                 (static_cast<uint8_t>(data[9]) << 8) |
                 (static_cast<uint8_t>(data[10]) << 16) |
                 (static_cast<uint8_t>(data[11]) << 24);
    offset = 12 + header_len;
  }

  // Parse the header dict string
  std::string header(data.begin() + (major == 1 ? 10 : 12),
                     data.begin() + offset);

  // Check dtype
  is_double = (header.find("'<f8'") != std::string::npos ||
               header.find("'float64'") != std::string::npos);

  // Parse shape - find 'shape': (x,) or 'shape': (x, y)
  auto shape_pos = header.find("'shape':");
  if (shape_pos == std::string::npos)
    return false;

  auto paren_start = header.find('(', shape_pos);
  auto paren_end   = header.find(')', shape_pos);
  if (paren_start == std::string::npos || paren_end == std::string::npos)
    return false;

  std::string shape_str =
      header.substr(paren_start + 1, paren_end - paren_start - 1);
  std::stringstream ss(shape_str);
  std::string item;
  shape.clear();
  while (std::getline(ss, item, ',')) {
    item.erase(std::remove_if(item.begin(), item.end(), ::isspace), item.end());
    if (!item.empty()) {
      shape.push_back(std::stoul(item));
    }
  }

  return true;
}

// Simple ZIP local file header parser
struct ZipEntry {
  std::string filename;
  uint32_t compressed_size;
  uint32_t uncompressed_size;
  uint16_t compression_method;
  size_t data_offset;
};

bool readZipEntries(std::istream& is, std::vector<ZipEntry>& entries) {
  entries.clear();

  while (is.good()) {
    uint32_t sig = readUint32LE(is);
    if (sig != 0x04034b50)
      break;  // Not a local file header

    is.seekg(2, std::ios::cur);  // version needed
    is.seekg(2, std::ios::cur);  // flags
    uint16_t compression = readUint16LE(is);
    is.seekg(4, std::ios::cur);  // mod time/date
    is.seekg(4, std::ios::cur);  // CRC32
    uint32_t compressed_size   = readUint32LE(is);
    uint32_t uncompressed_size = readUint32LE(is);
    uint16_t filename_len      = readUint16LE(is);
    uint16_t extra_len         = readUint16LE(is);

    std::string filename(filename_len, '\0');
    is.read(&filename[0], filename_len);
    is.seekg(extra_len, std::ios::cur);  // skip extra field

    ZipEntry entry;
    entry.filename           = filename;
    entry.compressed_size    = compressed_size;
    entry.uncompressed_size  = uncompressed_size;
    entry.compression_method = compression;
    entry.data_offset        = is.tellg();

    entries.push_back(entry);
    is.seekg(compressed_size, std::ios::cur);
  }

  return !entries.empty();
}

}  // anonymous namespace

int MeshManager::LoadMesh(const std::string& node_file,
                          const std::string& elem_file,
                          const std::string& name) {
  Eigen::MatrixXd nodes;
  Eigen::MatrixXi elements;

  int n_nodes = FEAT10_read_nodes(node_file, nodes);
  int n_elems = FEAT10_read_elements(elem_file, elements);

  if (n_nodes <= 0 || n_elems <= 0) {
    std::cerr << "MeshManager: Failed to load mesh from " << node_file
              << " and " << elem_file << std::endl;
    return -1;
  }

  // Create mesh instance
  MeshInstance instance;
  instance.node_offset    = total_nodes_;
  instance.element_offset = total_elements_;
  instance.num_nodes      = n_nodes;
  instance.num_elements   = n_elems;
  instance.name =
      name.empty() ? "mesh_" + std::to_string(mesh_instances_.size()) : name;

  // Store buffers
  node_buffers_.push_back(nodes);
  elem_buffers_.push_back(elements);
  mesh_instances_.push_back(instance);

  // Update totals
  total_nodes_ += n_nodes;
  total_elements_ += n_elems;

  // Initialize empty scalar field buffer for this mesh
  scalar_field_buffers_.push_back(Eigen::VectorXd());

  // Rebuild unified arrays
  RebuildUnifiedArrays();

  return static_cast<int>(mesh_instances_.size()) - 1;
}

bool MeshManager::LoadScalarFieldFromNpz(int mesh_id,
                                         const std::string& npz_file,
                                         const std::string& field_key) {
  if (mesh_id < 0 || mesh_id >= static_cast<int>(mesh_instances_.size())) {
    std::cerr << "MeshManager: Invalid mesh_id " << mesh_id << std::endl;
    return false;
  }

  std::ifstream file(npz_file, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "MeshManager: Failed to open " << npz_file << std::endl;
    return false;
  }

  std::vector<ZipEntry> entries;
  if (!readZipEntries(file, entries)) {
    std::cerr << "MeshManager: Failed to parse NPZ file " << npz_file
              << std::endl;
    return false;
  }

  // Find the requested field
  std::string target_name     = field_key + ".npy";
  const ZipEntry* field_entry = nullptr;
  for (const auto& e : entries) {
    if (e.filename == target_name) {
      field_entry = &e;
      break;
    }
  }

  if (!field_entry) {
    std::cerr << "MeshManager: Field '" << field_key << "' not found in "
              << npz_file << std::endl;
    return false;
  }

  // Only support uncompressed NPZ (compression_method == 0)
  if (field_entry->compression_method != 0) {
    std::cerr << "MeshManager: Compressed NPZ not supported. "
              << "Please save with np.savez (not np.savez_compressed)"
              << std::endl;
    return false;
  }

  // Read the NPY data
  file.seekg(field_entry->data_offset);
  std::vector<char> npy_data(field_entry->uncompressed_size);
  file.read(npy_data.data(), field_entry->uncompressed_size);

  // Parse NPY header
  size_t data_offset;
  std::vector<size_t> shape;
  bool is_double;
  if (!parseNpyHeader(npy_data, data_offset, shape, is_double)) {
    std::cerr << "MeshManager: Failed to parse NPY header for " << field_key
              << std::endl;
    return false;
  }

  if (!is_double) {
    std::cerr << "MeshManager: Only float64 scalar fields supported"
              << std::endl;
    return false;
  }

  if (shape.empty()) {
    std::cerr << "MeshManager: Invalid shape for " << field_key << std::endl;
    return false;
  }

  size_t n_values      = shape[0];
  const auto& instance = mesh_instances_[mesh_id];

  // Check size compatibility
  // The NPZ might contain data for linearized mesh (fewer nodes than T10)
  if (n_values > static_cast<size_t>(instance.num_nodes)) {
    std::cerr << "MeshManager: Scalar field has " << n_values
              << " values but mesh has " << instance.num_nodes << " nodes"
              << std::endl;
    return false;
  }

  // Read scalar values
  Eigen::VectorXd field(instance.num_nodes);
  field.setZero();  // Initialize to zero

  const double* src =
      reinterpret_cast<const double*>(npy_data.data() + data_offset);

  // If we need to handle original_vertex_ids mapping, we should load that too
  // For now, assume direct mapping for first n_values nodes
  // TODO: Add proper vertex ID remapping if needed
  for (size_t i = 0;
       i < n_values && i < static_cast<size_t>(instance.num_nodes); ++i) {
    field(i) = src[i];
  }

  scalar_field_buffers_[mesh_id] = field;
  has_scalar_fields_             = true;
  RebuildScalarFieldArray();

  std::cout << "MeshManager: Loaded scalar field '" << field_key << "' from "
            << npz_file << " (" << n_values << " values)" << std::endl;

  return true;
}

bool MeshManager::LoadScalarFieldFromBinary(int mesh_id,
                                            const std::string& bin_file,
                                            int n_values) {
  if (mesh_id < 0 || mesh_id >= static_cast<int>(mesh_instances_.size())) {
    std::cerr << "MeshManager: Invalid mesh_id " << mesh_id << std::endl;
    return false;
  }

  std::ifstream file(bin_file, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "MeshManager: Failed to open " << bin_file << std::endl;
    return false;
  }

  const auto& instance = mesh_instances_[mesh_id];
  Eigen::VectorXd field(instance.num_nodes);
  field.setZero();

  int read_count = std::min(n_values, instance.num_nodes);
  file.read(reinterpret_cast<char*>(field.data()), read_count * sizeof(double));

  if (!file.good()) {
    std::cerr << "MeshManager: Error reading binary file " << bin_file
              << std::endl;
    return false;
  }

  scalar_field_buffers_[mesh_id] = field;
  has_scalar_fields_             = true;
  RebuildScalarFieldArray();

  std::cout << "MeshManager: Loaded scalar field from " << bin_file << " ("
            << read_count << " values)" << std::endl;

  return true;
}

bool MeshManager::SetScalarField(int mesh_id, const Eigen::VectorXd& field) {
  if (mesh_id < 0 || mesh_id >= static_cast<int>(mesh_instances_.size())) {
    std::cerr << "MeshManager: Invalid mesh_id " << mesh_id << std::endl;
    return false;
  }

  const auto& instance = mesh_instances_[mesh_id];
  if (field.size() != instance.num_nodes) {
    std::cerr << "MeshManager: Scalar field size (" << field.size()
              << ") does not match mesh node count (" << instance.num_nodes
              << ")" << std::endl;
    return false;
  }

  scalar_field_buffers_[mesh_id] = field;
  has_scalar_fields_             = true;
  RebuildScalarFieldArray();

  return true;
}

void MeshManager::RebuildScalarFieldArray() {
  if (!has_scalar_fields_) {
    all_scalar_fields_.resize(0);
    return;
  }

  all_scalar_fields_.resize(total_nodes_);
  all_scalar_fields_.setZero();

  int node_offset = 0;
  for (size_t i = 0; i < mesh_instances_.size(); ++i) {
    const auto& instance = mesh_instances_[i];
    if (i < scalar_field_buffers_.size() &&
        scalar_field_buffers_[i].size() > 0) {
      all_scalar_fields_.segment(node_offset, instance.num_nodes) =
          scalar_field_buffers_[i];
    }
    node_offset += instance.num_nodes;
  }
}

void MeshManager::TransformMesh(int mesh_id, const Eigen::Matrix4d& transform) {
  if (mesh_id < 0 || mesh_id >= static_cast<int>(node_buffers_.size())) {
    std::cerr << "MeshManager: Invalid mesh_id " << mesh_id << std::endl;
    return;
  }

  Eigen::MatrixXd& nodes = node_buffers_[mesh_id];
  for (int i = 0; i < nodes.rows(); ++i) {
    Eigen::Vector4d p(nodes(i, 0), nodes(i, 1), nodes(i, 2), 1.0);
    Eigen::Vector4d p_transformed = transform * p;
    nodes(i, 0)                   = p_transformed(0);
    nodes(i, 1)                   = p_transformed(1);
    nodes(i, 2)                   = p_transformed(2);
  }

  RebuildUnifiedArrays();
}

void MeshManager::TranslateMesh(int mesh_id, double dx, double dy, double dz) {
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform(0, 3)           = dx;
  transform(1, 3)           = dy;
  transform(2, 3)           = dz;
  TransformMesh(mesh_id, transform);
}

void MeshManager::RebuildUnifiedArrays() {
  if (mesh_instances_.empty()) {
    all_nodes_.resize(0, 3);
    all_elements_.resize(0, 0);
    return;
  }

  // Determine element columns (should be consistent, e.g., 10 for T10 tets)
  int elem_cols = elem_buffers_[0].cols();

  // Allocate unified arrays
  all_nodes_.resize(total_nodes_, 3);
  all_elements_.resize(total_elements_, elem_cols);

  int node_offset = 0;
  int elem_offset = 0;

  for (size_t i = 0; i < mesh_instances_.size(); ++i) {
    const auto& nodes = node_buffers_[i];
    const auto& elems = elem_buffers_[i];

    // Copy nodes
    all_nodes_.block(node_offset, 0, nodes.rows(), 3) = nodes;

    // Copy elements with shifted indices
    for (int e = 0; e < elems.rows(); ++e) {
      for (int n = 0; n < elems.cols(); ++n) {
        all_elements_(elem_offset + e, n) = elems(e, n) + node_offset;
      }
    }

    node_offset += static_cast<int>(nodes.rows());
    elem_offset += static_cast<int>(elems.rows());
  }
}

const MeshInstance& MeshManager::GetMeshInstance(int mesh_id) const {
  if (mesh_id < 0 || mesh_id >= static_cast<int>(mesh_instances_.size())) {
    throw std::out_of_range("MeshManager: Invalid mesh_id " +
                            std::to_string(mesh_id));
  }
  return mesh_instances_[mesh_id];
}

int MeshManager::GetMeshIdFromElement(int global_elem_idx) const {
  for (size_t i = 0; i < mesh_instances_.size(); ++i) {
    const auto& instance = mesh_instances_[i];
    if (global_elem_idx >= instance.element_offset &&
        global_elem_idx < instance.element_offset + instance.num_elements) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

int MeshManager::GetMeshIdFromNode(int global_node_idx) const {
  for (size_t i = 0; i < mesh_instances_.size(); ++i) {
    const auto& instance = mesh_instances_[i];
    if (global_node_idx >= instance.node_offset &&
        global_node_idx < instance.node_offset + instance.num_nodes) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

void MeshManager::Clear() {
  mesh_instances_.clear();
  node_buffers_.clear();
  elem_buffers_.clear();
  scalar_field_buffers_.clear();
  all_nodes_.resize(0, 3);
  all_elements_.resize(0, 0);
  all_scalar_fields_.resize(0);
  total_nodes_       = 0;
  total_elements_    = 0;
  has_scalar_fields_ = false;
}

}  // namespace ANCFCPUUtils
