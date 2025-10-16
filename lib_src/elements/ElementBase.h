#pragma once

#include <Eigen/Dense>
#include <vector>

enum ElementType { TYPE_3243, TYPE_3443, TYPE_T10 };

class ElementBase {
 public:
  ElementType type;

  virtual ~ElementBase() {}

  ElementBase *d_data;

  // Do not use virtual function in solver class
  // CUDA cannot use virtual function
  virtual __host__ __device__ int get_n_beam() const = 0;
  virtual __host__ __device__ int get_n_coef() const = 0;

  // Core computation functions (actually implemented and used)
  virtual void CalcMassMatrix()     = 0;
  virtual void CalcInternalForce()  = 0;
  virtual void CalcConstraintData() = 0;
  virtual void CalcP()              = 0;

  virtual void RetrieveMassMatrixToCPU(Eigen::MatrixXd &mass_matrix)       = 0;
  virtual void RetrieveInternalForceToCPU(Eigen::VectorXd &internal_force) = 0;
  virtual void RetrieveConstraintDataToCPU(Eigen::VectorXd &constraint)    = 0;
  virtual void RetrieveConstraintJacobianToCPU(
      Eigen::MatrixXd &constraint_jac)                     = 0;
  virtual void RetrievePositionToCPU(Eigen::VectorXd &x12, Eigen::VectorXd &y12,
                                     Eigen::VectorXd &z12) = 0;
  virtual void RetrieveDeformationGradientToCPU(
      std::vector<std::vector<Eigen::MatrixXd>> &deformation_gradient) = 0;
  virtual void RetrievePFromFToCPU(
      std::vector<std::vector<Eigen::MatrixXd>> &p_from_F) = 0;
};