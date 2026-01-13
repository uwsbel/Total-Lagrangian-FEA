#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../lib_src/collision/HydroelasticNarrowphase.cuh"

namespace ANCFCPUUtils {

/**
 * Utility class for exporting collision detection results for visualization.
 * Supports VTP (ParaView polygons), CSV, JSON, and VTU formats.
 */
class VisualizationUtils {
 public:
  /**
   * Export contact patches to VTP format for ParaView visualization.
   * Each contact patch is exported as a polygon with associated data.
   *
   * @param patches Vector of contact patches to export
   * @param filename Output VTP filename
   * @return true if export successful
   */
  static bool ExportContactPatchesToVTP(
      const std::vector<ContactPatch>& patches, const std::string& filename,
      double normalScale = 0.02) {
    std::ofstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: Cannot open file " << filename << " for writing"
                << std::endl;
      return false;
    }

    // Count total vertices and polygons
    int totalVerts  = 0;
    int numPolygons = 0;
    std::vector<double> forceMag;
    forceMag.reserve(patches.size());
    double maxForceMag = 0.0;
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        totalVerts += patch.numVertices;
        numPolygons++;

        double f = std::abs(patch.p_equilibrium) * patch.area;
        forceMag.push_back(f);
        if (f > maxForceMag) {
          maxForceMag = f;
        }
      }
    }

    std::vector<double> normalLineLength;
    normalLineLength.reserve(forceMag.size());
    if (maxForceMag > 0.0) {
      for (double f : forceMag) {
        normalLineLength.push_back(0.5 * normalScale * (f / maxForceMag));
      }
    } else {
      normalLineLength.assign(forceMag.size(), 0.0);
    }

    int numLines    = numPolygons;
    int totalPoints = totalVerts + numLines * 2;

    file << std::setprecision(15) << std::scientific;

    // VTP XML header
    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"PolyData\" version=\"1.0\" "
            "byte_order=\"LittleEndian\">\n";
    file << "  <PolyData>\n";
    file << "    <Piece NumberOfPoints=\"" << totalPoints
         << "\" NumberOfLines=\"" << numLines << "\" NumberOfPolys=\""
         << numPolygons << "\">\n";

    // Handle empty patches case - write minimal valid VTP structure
    if (numPolygons == 0) {
      file << "      <Points>\n";
      file << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" "
              "format=\"ascii\">\n";
      file << "        </DataArray>\n";
      file << "      </Points>\n";

      file << "      <Lines>\n";
      file << "        <DataArray type=\"Int32\" Name=\"connectivity\" "
              "format=\"ascii\">\n";
      file << "        </DataArray>\n";
      file << "        <DataArray type=\"Int32\" Name=\"offsets\" "
              "format=\"ascii\">\n";
      file << "        </DataArray>\n";
      file << "      </Lines>\n";

      file << "      <Polys>\n";
      file << "        <DataArray type=\"Int32\" Name=\"connectivity\" "
              "format=\"ascii\">\n";
      file << "        </DataArray>\n";
      file << "        <DataArray type=\"Int32\" Name=\"offsets\" "
              "format=\"ascii\">\n";
      file << "        </DataArray>\n";
      file << "      </Polys>\n";
      file << "    </Piece>\n";
      file << "  </PolyData>\n";
      file << "</VTKFile>\n";
      file.close();
      return true;
    }

    // Points (vertices)
    file << "      <Points>\n";
    file << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" "
            "format=\"ascii\">\n";
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        for (int i = 0; i < patch.numVertices; ++i) {
          file << "          " << patch.vertices[i].x << " "
               << patch.vertices[i].y << " " << patch.vertices[i].z << "\n";
        }
      }
    }

    int arrowIdx = 0;
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        double len = normalLineLength[arrowIdx];
        file << "          " << patch.centroid.x - len * patch.normal.x
             << " " << patch.centroid.y - len * patch.normal.y << " "
             << patch.centroid.z - len * patch.normal.z << "\n";
        file << "          " << patch.centroid.x + len * patch.normal.x
             << " " << patch.centroid.y + len * patch.normal.y << " "
             << patch.centroid.z + len * patch.normal.z << "\n";
        arrowIdx++;
      }
    }
    file << "        </DataArray>\n";
    file << "      </Points>\n";

    file << "      <Lines>\n";
    file << "        <DataArray type=\"Int32\" Name=\"connectivity\" "
            "format=\"ascii\">\n";
    int linePointOffset = totalVerts;
    arrowIdx            = 0;
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        int base = linePointOffset + arrowIdx * 2;
        file << "          " << base << " " << base + 1 << "\n";
        arrowIdx++;
      }
    }
    file << "        </DataArray>\n";

    file << "        <DataArray type=\"Int32\" Name=\"offsets\" "
            "format=\"ascii\">\n";
    for (int i = 0; i < numLines; ++i) {
      file << "          " << (i + 1) * 2 << "\n";
    }
    file << "        </DataArray>\n";
    file << "      </Lines>\n";

    // Polygons (connectivity and offsets)
    file << "      <Polys>\n";

    // Connectivity: vertex indices for each polygon
    file << "        <DataArray type=\"Int32\" Name=\"connectivity\" "
            "format=\"ascii\">\n";
    int vertOffset = 0;
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        file << "          ";
        for (int i = 0; i < patch.numVertices; ++i) {
          file << (vertOffset + i) << " ";
        }
        file << "\n";
        vertOffset += patch.numVertices;
      }
    }
    file << "        </DataArray>\n";

    // Offsets: cumulative vertex count
    file << "        <DataArray type=\"Int32\" Name=\"offsets\" "
            "format=\"ascii\">\n";
    int offset = 0;
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        offset += patch.numVertices;
        file << "          " << offset << "\n";
      }
    }
    file << "        </DataArray>\n";
    file << "      </Polys>\n";

    // Cell data (per-polygon attributes)
    file << "      <CellData>\n";

    file << "        <DataArray type=\"Float64\" Name=\"ForceMag\" "
            "format=\"ascii\">\n";
    for (double f : forceMag) {
      file << "          " << f << "\n";
    }
    for (double f : forceMag) {
      file << "          " << f << "\n";
    }
    file << "        </DataArray>\n";

    file << "        <DataArray type=\"Float64\" Name=\"NormalLineLength\" "
            "format=\"ascii\">\n";
    for (double len : normalLineLength) {
      file << "          " << len << "\n";
    }
    for (double len : normalLineLength) {
      file << "          " << len << "\n";
    }
    file << "        </DataArray>\n";

    // Area
    file << "        <DataArray type=\"Float64\" Name=\"Area\" "
            "format=\"ascii\">\n";
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        file << "          " << patch.area << "\n";
      }
    }
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        file << "          " << patch.area << "\n";
      }
    }
    file << "        </DataArray>\n";

    // g_A (gradient in A direction)
    file << "        <DataArray type=\"Float64\" Name=\"g_A\" "
            "format=\"ascii\">\n";
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        file << "          " << patch.g_A << "\n";
      }
    }
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        file << "          " << patch.g_A << "\n";
      }
    }
    file << "        </DataArray>\n";

    // g_B (gradient in B direction)
    file << "        <DataArray type=\"Float64\" Name=\"g_B\" "
            "format=\"ascii\">\n";
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        file << "          " << patch.g_B << "\n";
      }
    }
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        file << "          " << patch.g_B << "\n";
      }
    }
    file << "        </DataArray>\n";

    // Equilibrium pressure
    file << "        <DataArray type=\"Float64\" Name=\"p_equilibrium\" "
            "format=\"ascii\">\n";
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        file << "          " << patch.p_equilibrium << "\n";
      }
    }
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        file << "          " << patch.p_equilibrium << "\n";
      }
    }
    file << "        </DataArray>\n";

    // Tet pair indices
    file << "        <DataArray type=\"Int32\" Name=\"TetA_idx\" "
            "format=\"ascii\">\n";
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        file << "          " << patch.tetA_idx << "\n";
      }
    }
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        file << "          " << patch.tetA_idx << "\n";
      }
    }
    file << "        </DataArray>\n";

    file << "        <DataArray type=\"Int32\" Name=\"TetB_idx\" "
            "format=\"ascii\">\n";
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        file << "          " << patch.tetB_idx << "\n";
      }
    }
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        file << "          " << patch.tetB_idx << "\n";
      }
    }
    file << "        </DataArray>\n";

    // Valid orientation flag
    file << "        <DataArray type=\"Int32\" Name=\"ValidOrientation\" "
            "format=\"ascii\">\n";
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        file << "          " << (patch.validOrientation ? 1 : 0) << "\n";
      }
    }
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        file << "          " << (patch.validOrientation ? 1 : 0) << "\n";
      }
    }
    file << "        </DataArray>\n";

    // Normal vectors
    file << "        <DataArray type=\"Float64\" Name=\"Normal\" "
            "NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        file << "          " << patch.normal.x << " " << patch.normal.y << " "
             << patch.normal.z << "\n";
      }
    }
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        file << "          " << patch.normal.x << " " << patch.normal.y << " "
             << patch.normal.z << "\n";
      }
    }
    file << "        </DataArray>\n";

    // Centroid
    file << "        <DataArray type=\"Float64\" Name=\"Centroid\" "
            "NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        file << "          " << patch.centroid.x << " " << patch.centroid.y
             << " " << patch.centroid.z << "\n";
      }
    }
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        file << "          " << patch.centroid.x << " " << patch.centroid.y
             << " " << patch.centroid.z << "\n";
      }
    }
    file << "        </DataArray>\n";

    file << "      </CellData>\n";
    file << "    </Piece>\n";
    file << "  </PolyData>\n";
    file << "</VTKFile>\n";

    file.close();
    std::cout << "Exported " << numPolygons << " contact patches to "
              << filename << std::endl;
    return true;
  }

  /**
   * Export contact patches to CSV format for analysis in Python/MATLAB.
   * One row per patch with centroid, normal, area, and other attributes.
   *
   * @param patches Vector of contact patches to export
   * @param filename Output CSV filename
   * @return true if export successful
   */
  static bool ExportContactPatchesToCSV(
      const std::vector<ContactPatch>& patches, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: Cannot open file " << filename << " for writing"
                << std::endl;
      return false;
    }

    file << std::setprecision(15) << std::scientific;

    // Header
    file << "patch_idx,tetA_idx,tetB_idx,centroid_x,centroid_y,centroid_z,"
         << "normal_x,normal_y,normal_z,area,g_A,g_B,p_equilibrium,"
         << "num_vertices,valid_orientation\n";

    int patchIdx = 0;
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        file << patchIdx << "," << patch.tetA_idx << "," << patch.tetB_idx
             << "," << patch.centroid.x << "," << patch.centroid.y << ","
             << patch.centroid.z << "," << patch.normal.x << ","
             << patch.normal.y << "," << patch.normal.z << "," << patch.area
             << "," << patch.g_A << "," << patch.g_B << ","
             << patch.p_equilibrium << "," << patch.numVertices << ","
             << (patch.validOrientation ? 1 : 0) << "\n";
        patchIdx++;
      }
    }

    file.close();
    std::cout << "Exported " << patchIdx << " contact patches to " << filename
              << std::endl;
    return true;
  }

  /**
   * Export contact patches to JSON format for debugging and web visualization.
   *
   * @param patches Vector of contact patches to export
   * @param filename Output JSON filename
   * @return true if export successful
   */
  static bool ExportContactPatchesToJSON(
      const std::vector<ContactPatch>& patches, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: Cannot open file " << filename << " for writing"
                << std::endl;
      return false;
    }

    file << std::setprecision(15);

    file << "{\n";
    file << "  \"contact_patches\": [\n";

    bool firstPatch = true;
    int patchCount  = 0;
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        if (!firstPatch) {
          file << ",\n";
        }
        firstPatch = false;

        file << "    {\n";
        file << "      \"tetA_idx\": " << patch.tetA_idx << ",\n";
        file << "      \"tetB_idx\": " << patch.tetB_idx << ",\n";
        file << "      \"centroid\": [" << patch.centroid.x << ", "
             << patch.centroid.y << ", " << patch.centroid.z << "],\n";
        file << "      \"normal\": [" << patch.normal.x << ", "
             << patch.normal.y << ", " << patch.normal.z << "],\n";
        file << "      \"area\": " << patch.area << ",\n";
        file << "      \"g_A\": " << patch.g_A << ",\n";
        file << "      \"g_B\": " << patch.g_B << ",\n";
        file << "      \"p_equilibrium\": " << patch.p_equilibrium << ",\n";
        file << "      \"valid_orientation\": "
             << (patch.validOrientation ? "true" : "false") << ",\n";
        file << "      \"vertices\": [\n";
        for (int i = 0; i < patch.numVertices; ++i) {
          file << "        [" << patch.vertices[i].x << ", "
               << patch.vertices[i].y << ", " << patch.vertices[i].z << "]";
          if (i < patch.numVertices - 1) {
            file << ",";
          }
          file << "\n";
        }
        file << "      ]\n";
        file << "    }";
        patchCount++;
      }
    }

    file << "\n  ]\n";
    file << "}\n";

    file.close();
    std::cout << "Exported " << patchCount << " contact patches to " << filename
              << std::endl;
    return true;
  }

  /**
   * Export mesh to VTU format for volumetric visualization.
   * Includes scalar field values on nodes.
   *
   * @param nodes Node coordinates (N x 3, rows are nodes)
   * @param elements Element connectivity (M x 10 for T10 elements, rows are
   * elements)
   * @param scalarField Scalar field values per node
   * @param filename Output VTU filename
   * @return true if export successful
   */
  static bool ExportMeshToVTU(const Eigen::MatrixXd& nodes,
                              const Eigen::MatrixXi& elements,
                              const Eigen::VectorXd& scalarField,
                              const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: Cannot open file " << filename << " for writing"
                << std::endl;
      return false;
    }

    int numNodes    = nodes.rows();
    int numElements = elements.rows();

    file << std::setprecision(15) << std::scientific;

    // VTU XML header
    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" "
            "byte_order=\"LittleEndian\">\n";
    file << "  <UnstructuredGrid>\n";
    file << "    <Piece NumberOfPoints=\"" << numNodes << "\" NumberOfCells=\""
         << numElements << "\">\n";

    // Points (node coordinates)
    file << "      <Points>\n";
    file << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" "
            "format=\"ascii\">\n";
    for (int i = 0; i < numNodes; ++i) {
      file << "          " << nodes(i, 0) << " " << nodes(i, 1) << " "
           << nodes(i, 2) << "\n";
    }
    file << "        </DataArray>\n";
    file << "      </Points>\n";

    // Point data (scalar field)
    file << "      <PointData Scalars=\"pressure\">\n";
    file << "        <DataArray type=\"Float64\" Name=\"pressure\" "
            "format=\"ascii\">\n";
    for (int i = 0; i < numNodes; ++i) {
      if (i < scalarField.size()) {
        file << "          " << scalarField(i) << "\n";
      } else {
        file << "          0.0\n";
      }
    }
    file << "        </DataArray>\n";
    file << "      </PointData>\n";

    // Cells (element connectivity)
    // For T10 elements, use only first 4 nodes (corners) for visualization
    file << "      <Cells>\n";
    file << "        <DataArray type=\"Int32\" Name=\"connectivity\" "
            "format=\"ascii\">\n";
    for (int i = 0; i < numElements; ++i) {
      file << "          ";
      // Use first 4 nodes (linear tet for visualization)
      for (int j = 0; j < 4; ++j) {
        file << elements(i, j) << " ";
      }
      file << "\n";
    }
    file << "        </DataArray>\n";

    // Offsets
    file << "        <DataArray type=\"Int32\" Name=\"offsets\" "
            "format=\"ascii\">\n";
    for (int i = 0; i < numElements; ++i) {
      file << "          " << (i + 1) * 4 << "\n";
    }
    file << "        </DataArray>\n";

    // Cell types (10 = VTK_TETRA)
    file << "        <DataArray type=\"UInt8\" Name=\"types\" "
            "format=\"ascii\">\n";
    for (int i = 0; i < numElements; ++i) {
      file << "          10\n";
    }
    file << "        </DataArray>\n";

    file << "      </Cells>\n";
    file << "    </Piece>\n";
    file << "  </UnstructuredGrid>\n";
    file << "</VTKFile>\n";

    file.close();
    std::cout << "Exported mesh with " << numNodes << " nodes and "
              << numElements << " elements to " << filename << std::endl;
    return true;
  }

  /**
   * Export normal vectors as VTP line segments for visualization.
   * Each normal is shown as an arrow from patch centroid.
   *
   * @param patches Vector of contact patches
   * @param arrowScale Scale factor for arrow length
   * @param filename Output VTP filename
   * @return true if export successful
   */
  static bool ExportNormalsAsArrows(const std::vector<ContactPatch>& patches,
                                    double arrowScale,
                                    const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: Cannot open file " << filename << " for writing"
                << std::endl;
      return false;
    }

    // Count valid patches
    int numArrows = 0;
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        numArrows++;
      }
    }

    if (numArrows == 0) {
      file << std::setprecision(15) << std::scientific;

      file << "<?xml version=\"1.0\"?>\n";
      file << "<VTKFile type=\"PolyData\" version=\"1.0\" "
              "byte_order=\"LittleEndian\">\n";
      file << "  <PolyData>\n";
      file << "    <Piece NumberOfPoints=\"0\" NumberOfLines=\"0\">\n";

      file << "      <Points>\n";
      file << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" "
              "format=\"ascii\">\n";
      file << "        </DataArray>\n";
      file << "      </Points>\n";

      file << "      <Lines>\n";
      file << "        <DataArray type=\"Int32\" Name=\"connectivity\" "
              "format=\"ascii\">\n";
      file << "        </DataArray>\n";
      file << "        <DataArray type=\"Int32\" Name=\"offsets\" "
              "format=\"ascii\">\n";
      file << "        </DataArray>\n";
      file << "      </Lines>\n";

      file << "    </Piece>\n";
      file << "  </PolyData>\n";
      file << "</VTKFile>\n";
      file.close();
      return true;
    }

    file << std::setprecision(15) << std::scientific;

    // VTP XML header
    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"PolyData\" version=\"1.0\" "
            "byte_order=\"LittleEndian\">\n";
    file << "  <PolyData>\n";
    file << "    <Piece NumberOfPoints=\"" << numArrows * 2
         << "\" NumberOfLines=\"" << numArrows << "\">\n";

    // Points (arrow start and end)
    file << "      <Points>\n";
    file << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" "
            "format=\"ascii\">\n";
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        // Start point (centroid)
        file << "          " << patch.centroid.x << " " << patch.centroid.y
             << " " << patch.centroid.z << "\n";
        // End point (centroid + scaled normal)
        file << "          " << patch.centroid.x + arrowScale * patch.normal.x
             << " " << patch.centroid.y + arrowScale * patch.normal.y << " "
             << patch.centroid.z + arrowScale * patch.normal.z << "\n";
      }
    }
    file << "        </DataArray>\n";
    file << "      </Points>\n";

    // Lines (connectivity and offsets)
    file << "      <Lines>\n";
    file << "        <DataArray type=\"Int32\" Name=\"connectivity\" "
            "format=\"ascii\">\n";
    for (int i = 0; i < numArrows; ++i) {
      file << "          " << i * 2 << " " << i * 2 + 1 << "\n";
    }
    file << "        </DataArray>\n";

    file << "        <DataArray type=\"Int32\" Name=\"offsets\" "
            "format=\"ascii\">\n";
    for (int i = 0; i < numArrows; ++i) {
      file << "          " << (i + 1) * 2 << "\n";
    }
    file << "        </DataArray>\n";
    file << "      </Lines>\n";

    // Cell data for coloring
    file << "      <CellData>\n";
    file << "        <DataArray type=\"Float64\" Name=\"Area\" "
            "format=\"ascii\">\n";
    for (const auto& patch : patches) {
      if (patch.isValid && patch.numVertices >= 3) {
        file << "          " << patch.area << "\n";
      }
    }
    file << "        </DataArray>\n";
    file << "      </CellData>\n";

    file << "    </Piece>\n";
    file << "  </PolyData>\n";
    file << "</VTKFile>\n";

    file.close();
    std::cout << "Exported " << numArrows << " normal arrows to " << filename
              << std::endl;
    return true;
  }

  /**
   * Export mesh with displacement field to VTU format.
   * Includes displacement magnitude (scalar) and displacement vector.
   *
   * @param nodes Node coordinates (n_nodes x 3)
   * @param elements Element connectivity (n_elements x nodes_per_element)
   * @param displacement Displacement vector (3 * n_nodes), layout: [dx0, dy0, dz0, dx1, ...]
   * @param filename Output VTU filename
   * @return true if export successful
   */
  static bool ExportMeshWithDisplacement(const Eigen::MatrixXd& nodes,
                                         const Eigen::MatrixXi& elements,
                                         const Eigen::VectorXd& displacement,
                                         const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: Cannot open file " << filename << " for writing"
                << std::endl;
      return false;
    }

    int numNodes    = nodes.rows();
    int numElements = elements.rows();

    file << std::setprecision(15) << std::scientific;

    // VTU XML header
    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" "
            "byte_order=\"LittleEndian\">\n";
    file << "  <UnstructuredGrid>\n";
    file << "    <Piece NumberOfPoints=\"" << numNodes << "\" NumberOfCells=\""
         << numElements << "\">\n";

    // Points (node coordinates)
    file << "      <Points>\n";
    file << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" "
            "format=\"ascii\">\n";
    for (int i = 0; i < numNodes; ++i) {
      file << "          " << nodes(i, 0) << " " << nodes(i, 1) << " "
           << nodes(i, 2) << "\n";
    }
    file << "        </DataArray>\n";
    file << "      </Points>\n";

    // Point data (displacement magnitude and vector)
    file << "      <PointData Scalars=\"displacement_magnitude\" "
            "Vectors=\"displacement\">\n";

    // Displacement magnitude
    file << "        <DataArray type=\"Float64\" Name=\"displacement_magnitude\" "
            "format=\"ascii\">\n";
    for (int i = 0; i < numNodes; ++i) {
      double dx  = (i * 3 < displacement.size()) ? displacement(i * 3) : 0.0;
      double dy  = (i * 3 + 1 < displacement.size()) ? displacement(i * 3 + 1) : 0.0;
      double dz  = (i * 3 + 2 < displacement.size()) ? displacement(i * 3 + 2) : 0.0;
      double mag = std::sqrt(dx * dx + dy * dy + dz * dz);
      file << "          " << mag << "\n";
    }
    file << "        </DataArray>\n";

    // Displacement vector
    file << "        <DataArray type=\"Float64\" Name=\"displacement\" "
            "NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < numNodes; ++i) {
      double dx = (i * 3 < displacement.size()) ? displacement(i * 3) : 0.0;
      double dy = (i * 3 + 1 < displacement.size()) ? displacement(i * 3 + 1) : 0.0;
      double dz = (i * 3 + 2 < displacement.size()) ? displacement(i * 3 + 2) : 0.0;
      file << "          " << dx << " " << dy << " " << dz << "\n";
    }
    file << "        </DataArray>\n";
    file << "      </PointData>\n";

    // Cells (element connectivity)
    // For T10 elements, use only first 4 nodes (corners) for visualization
    file << "      <Cells>\n";
    file << "        <DataArray type=\"Int32\" Name=\"connectivity\" "
            "format=\"ascii\">\n";
    for (int i = 0; i < numElements; ++i) {
      file << "          ";
      // Use first 4 nodes (linear tet for visualization)
      for (int j = 0; j < 4; ++j) {
        file << elements(i, j) << " ";
      }
      file << "\n";
    }
    file << "        </DataArray>\n";

    // Offsets
    file << "        <DataArray type=\"Int32\" Name=\"offsets\" "
            "format=\"ascii\">\n";
    for (int i = 0; i < numElements; ++i) {
      file << "          " << (i + 1) * 4 << "\n";
    }
    file << "        </DataArray>\n";

    // Cell types (10 = VTK_TETRA)
    file << "        <DataArray type=\"UInt8\" Name=\"types\" "
            "format=\"ascii\">\n";
    for (int i = 0; i < numElements; ++i) {
      file << "          10\n";
    }
    file << "        </DataArray>\n";

    file << "      </Cells>\n";
    file << "    </Piece>\n";
    file << "  </UnstructuredGrid>\n";
    file << "</VTKFile>\n";

    file.close();
    return true;
  }

  /**
   * Export an ANCF3443 mesh as extruded hexahedra (one hex per element).
   * Uses only the nodal position DOF (dof 0) per node; gradient DOFs are ignored.
   *
   * For visualization convenience, this creates 8 points per element (no sharing)
   * by extruding the 4 corner nodes along the element normal by +/- thickness/2.
   *
   * Hex connectivity follows VTK_HEXAHEDRON (type 12) convention:
   * points [0,1,2,3] form the bottom quad, and [4,5,6,7] the top quad with the
   * same local ordering. For correct orientation/volume, each row of
   * element_connectivity should list the 4 corner nodes in a consistent loop
   * (e.g., counter-clockwise on the midsurface when viewed from the "top").
   *
   * @param x12 Global x coefficient vector (size >= 4 * n_nodes)
   * @param y12 Global y coefficient vector
   * @param z12 Global z coefficient vector
   * @param element_connectivity (n_elem x 4) node IDs (not coefficient IDs)
   * @param thickness Extrusion thickness (e.g., H)
   * @param filename Output VTU filename
   */
  static bool ExportANCF3443ToVTU(const Eigen::VectorXd& x12,
                                  const Eigen::VectorXd& y12,
                                  const Eigen::VectorXd& z12,
                                  const Eigen::MatrixXi& element_connectivity,
                                  double thickness,
                                  const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: Cannot open file " << filename << " for writing"
                << std::endl;
      return false;
    }

    const int numElements = element_connectivity.rows();
    const int numPoints   = numElements * 8;

    file << std::setprecision(15) << std::scientific;

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" "
            "byte_order=\"LittleEndian\">\n";
    file << "  <UnstructuredGrid>\n";
    file << "    <Piece NumberOfPoints=\"" << numPoints
         << "\" NumberOfCells=\"" << numElements << "\">\n";

    // Points
    file << "      <Points>\n";
    file << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" "
            "format=\"ascii\">\n";

    const double half = 0.5 * thickness;
    for (int e = 0; e < numElements; ++e) {
      Eigen::Vector3d p[4];
      for (int n = 0; n < 4; ++n) {
        const int node = element_connectivity(e, n);
        const int idx  = node * 4;
        p[n] = Eigen::Vector3d(x12(idx), y12(idx), z12(idx));
      }

      // Compute an element normal using a right-hand rule:
      //   normal ∝ (p[1] - p[0]) × (p[3] - p[0]).
      // This assumes the 4 corner nodes are provided in a consistent loop
      // (see the function docstring). If the node order is reversed, the normal
      // (and thus the meaning of "top" (+off) vs "bottom" (-off)) flips.
      Eigen::Vector3d normal = (p[1] - p[0]).cross(p[3] - p[0]);
      const double nrm = normal.norm();
      if (nrm < 1e-12) {
        normal = Eigen::Vector3d(0.0, 0.0, 1.0);
      } else {
        normal /= nrm;
      }

      const Eigen::Vector3d off = half * normal;
      for (int n = 0; n < 4; ++n) {
        const Eigen::Vector3d b = p[n] - off;
        file << "          " << b.x() << " " << b.y() << " " << b.z() << "\n";
      }
      for (int n = 0; n < 4; ++n) {
        const Eigen::Vector3d t = p[n] + off;
        file << "          " << t.x() << " " << t.y() << " " << t.z() << "\n";
      }
    }

    file << "        </DataArray>\n";
    file << "      </Points>\n";

    // Cells
    file << "      <Cells>\n";
    file << "        <DataArray type=\"Int32\" Name=\"connectivity\" "
            "format=\"ascii\">\n";
    for (int e = 0; e < numElements; ++e) {
      const int base = e * 8;
      file << "          " << base + 0 << " " << base + 1 << " " << base + 2
           << " " << base + 3 << " " << base + 4 << " " << base + 5 << " "
           << base + 6 << " " << base + 7 << "\n";
    }
    file << "        </DataArray>\n";

    file << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (int e = 0; e < numElements; ++e) {
      file << "          " << (e + 1) * 8 << "\n";
    }
    file << "        </DataArray>\n";

    // VTK_HEXAHEDRON = 12
    file << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (int e = 0; e < numElements; ++e) {
      file << "          12\n";
    }
    file << "        </DataArray>\n";
    file << "      </Cells>\n";

    file << "    </Piece>\n";
    file << "  </UnstructuredGrid>\n";
    file << "</VTKFile>\n";

    file.close();
    return true;
  }

  /**
   * Export an ANCF3243 beam mesh as hexahedra (one hex per element).
   * Uses only the nodal position DOF (dof 0) per node; gradient DOFs are ignored.
   *
   * For visualization convenience, this creates 8 points per element (no sharing)
   * by building a rectangular cross-section (width x height) at each end node and
   * connecting them into a hexahedron along the element tangent.
   *
   * Hex connectivity follows VTK_HEXAHEDRON (type 12) convention:
   * points [0,1,2,3] form the cross-section at node0, and [4,5,6,7] the
   * corresponding cross-section at node1. The local frame is constructed from the
   * chord tangent and an arbitrary reference vector (fallback if near-parallel);
   * this is intended for quick visualization, not exact ANCF geometry.
   *
   * @param x12 Global x coefficient vector (size >= 4 * n_nodes)
   * @param y12 Global y coefficient vector
   * @param z12 Global z coefficient vector
   * @param element_connectivity (n_elem x 2) node IDs (not coefficient IDs)
   * @param width Cross-section width (e.g., W)
   * @param height Cross-section height (e.g., H)
   * @param filename Output VTU filename
   */
  static bool ExportANCF3243ToVTU(const Eigen::VectorXd& x12,
                                  const Eigen::VectorXd& y12,
                                  const Eigen::VectorXd& z12,
                                  const Eigen::MatrixXi& element_connectivity,
                                  double width, double height,
                                  const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: Cannot open file " << filename << " for writing"
                << std::endl;
      return false;
    }

    const int numElements = element_connectivity.rows();
    const int numPoints   = numElements * 8;

    file << std::setprecision(15) << std::scientific;
    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" "
            "byte_order=\"LittleEndian\">\n";
    file << "  <UnstructuredGrid>\n";
    file << "    <Piece NumberOfPoints=\"" << numPoints
         << "\" NumberOfCells=\"" << numElements << "\">\n";

    const double halfW = 0.5 * width;
    const double halfH = 0.5 * height;

    file << "      <Points>\n";
    file << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" "
            "format=\"ascii\">\n";
    for (int e = 0; e < numElements; ++e) {
      const int n0 = element_connectivity(e, 0);
      const int n1 = element_connectivity(e, 1);
      const int i0 = n0 * 4;
      const int i1 = n1 * 4;

      const Eigen::Vector3d p0(x12(i0), y12(i0), z12(i0));
      const Eigen::Vector3d p1(x12(i1), y12(i1), z12(i1));

      Eigen::Vector3d t = p1 - p0;
      double tnorm      = t.norm();
      if (tnorm < 1e-12) {
        t = Eigen::Vector3d(1.0, 0.0, 0.0);
        tnorm = 1.0;
      }
      t /= tnorm;

      Eigen::Vector3d ref(0.0, 0.0, 1.0);
      if (std::abs(t.dot(ref)) > 0.9) {
        ref = Eigen::Vector3d(0.0, 1.0, 0.0);
      }

      Eigen::Vector3d n = t.cross(ref);
      double nnorm      = n.norm();
      if (nnorm < 1e-12) {
        n = Eigen::Vector3d(0.0, 1.0, 0.0);
        nnorm = 1.0;
      }
      n /= nnorm;

      Eigen::Vector3d b = t.cross(n);
      const double bnorm = b.norm();
      if (bnorm > 1e-12) {
        b /= bnorm;
      } else {
        b = Eigen::Vector3d(0.0, 0.0, 1.0);
      }

      const Eigen::Vector3d oW = halfW * n;
      const Eigen::Vector3d oH = halfH * b;

      const Eigen::Vector3d p0_0 = p0 - oW - oH;
      const Eigen::Vector3d p0_1 = p0 + oW - oH;
      const Eigen::Vector3d p0_2 = p0 + oW + oH;
      const Eigen::Vector3d p0_3 = p0 - oW + oH;
      const Eigen::Vector3d p1_0 = p1 - oW - oH;
      const Eigen::Vector3d p1_1 = p1 + oW - oH;
      const Eigen::Vector3d p1_2 = p1 + oW + oH;
      const Eigen::Vector3d p1_3 = p1 - oW + oH;

      for (const auto& p : {p0_0, p0_1, p0_2, p0_3, p1_0, p1_1, p1_2, p1_3}) {
        file << "          " << p.x() << " " << p.y() << " " << p.z() << "\n";
      }
    }
    file << "        </DataArray>\n";
    file << "      </Points>\n";

    file << "      <Cells>\n";
    file << "        <DataArray type=\"Int32\" Name=\"connectivity\" "
            "format=\"ascii\">\n";
    for (int e = 0; e < numElements; ++e) {
      const int base = e * 8;
      file << "          " << base + 0 << " " << base + 1 << " " << base + 2
           << " " << base + 3 << " " << base + 4 << " " << base + 5 << " "
           << base + 6 << " " << base + 7 << "\n";
    }
    file << "        </DataArray>\n";

    file << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (int e = 0; e < numElements; ++e) {
      file << "          " << (e + 1) * 8 << "\n";
    }
    file << "        </DataArray>\n";

    // VTK_HEXAHEDRON = 12
    file << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (int e = 0; e < numElements; ++e) {
      file << "          12\n";
    }
    file << "        </DataArray>\n";
    file << "      </Cells>\n";

    file << "    </Piece>\n";
    file << "  </UnstructuredGrid>\n";
    file << "</VTKFile>\n";

    file.close();
    return true;
  }
};

}  // namespace ANCFCPUUtils
