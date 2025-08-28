#pragma once

#include <Eigen/Dense>
#include <string>

namespace CSVUtils {

/**
 * Load an Eigen matrix from a CSV file
 */
bool loadMatrixCSV(Eigen::MatrixXd &matrix, const std::string &filepath);

/**
 * Save an Eigen matrix to a CSV file
 */
bool saveMatrixCSV(const Eigen::MatrixXd &matrix, const std::string &filepath);

} // namespace CSVUtils