#include "csv_utils.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace CSVUtils {

bool loadMatrixCSV(Eigen::MatrixXd &matrix, const std::string &filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filepath << " for reading"
              << std::endl;
    return false;
  }

  std::vector<std::vector<double>> data;
  std::string line;

  while (std::getline(file, line)) {
    if (line.empty())
      continue;  // Skip empty lines

    std::vector<double> row;
    std::stringstream ss(line);
    std::string cell;

    while (std::getline(ss, cell, ',')) {
      try {
        row.push_back(std::stod(cell));
      } catch (const std::exception &e) {
        std::cerr << "Error parsing number: " << cell << std::endl;
        return false;
      }
    }

    if (!row.empty()) {
      data.push_back(row);
    }
  }

  if (data.empty()) {
    std::cerr << "Error: No data found in file " << filepath << std::endl;
    return false;
  }

  int rows = data.size();
  int cols = data[0].size();

  // Check that all rows have the same number of columns
  for (size_t i = 1; i < data.size(); ++i) {
    if (data[i].size() != static_cast<size_t>(cols)) {
      std::cerr << "Error: Inconsistent number of columns in CSV file"
                << std::endl;
      return false;
    }
  }

  matrix.resize(rows, cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix(i, j) = data[i][j];
    }
  }

  file.close();
  return true;
}

bool saveMatrixCSV(const Eigen::MatrixXd &matrix, const std::string &filepath) {
  std::ofstream file(filepath);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filepath << " for writing"
              << std::endl;
    return false;
  }

  file << std::scientific << std::setprecision(17);
  for (int i = 0; i < matrix.rows(); ++i) {
    for (int j = 0; j < matrix.cols(); ++j) {
      file << matrix(i, j);
      if (j < matrix.cols() - 1)
        file << ",";
    }
    file << "\n";
  }

  file.close();
  return true;
}

}  // namespace CSVUtils