/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * File:    HolisticNewton.cuh
 * Brief:   Monolithic Newton solver for multi-element FE problems with
 *          global mixed constraints spanning multiple element blocks.
 *==============================================================
 *==============================================================*/

#pragma once

#include <cublas_v2.h>

#include <memory>
#include <vector>

#include "lib_utils/cuda_utils.h"
#include "../constraints/MixedConstraintSystem.h"
#include "FEMultiElementProblem.h"
#include "SolverBase.h"

struct HolisticNewtonParams {
  double inner_atol         = 1e-8;
  double inner_rtol         = 1e-6;
  double outer_tol          = 1e-6;
  double rho                = 1000.0;
  int max_outer             = 10;
  int max_inner             = 100;
  double time_step          = 1e-4;
  bool enable_line_search   = false;
};

class HolisticNewtonSolver : public SolverBase {
 public:
  HolisticNewtonSolver(FEMultiElementProblem* problem,
                       MixedConstraintSystem* mixed_constraints);
  ~HolisticNewtonSolver() override;

  HolisticNewtonSolver(const HolisticNewtonSolver&)            = delete;
  HolisticNewtonSolver& operator=(const HolisticNewtonSolver&) = delete;
  HolisticNewtonSolver(HolisticNewtonSolver&&)                 = delete;
  HolisticNewtonSolver& operator=(HolisticNewtonSolver&&)      = delete;

  void SetParameters(void* params) override;
  void Setup();
  void Solve() override;

  double* GetExternalForceDevicePtr() const;
  int GetTotalDofs() const { return n_dofs_; }
  int GetNumConstraints() const { return n_constraints_; }

 private:
  struct BlockLaunchInfo {
    ElementBase* element = nullptr;
    ElementType type     = TYPE_T10;
    int coef_offset      = 0;
    int coef_count       = 0;
    int n_elem           = 0;
    int n_qp             = 0;
  };

  FEMultiElementProblem* problem_                 = nullptr;
  MixedConstraintSystem* mixed_constraints_       = nullptr;
  HolisticNewtonParams params_;
  std::vector<BlockLaunchInfo> block_infos_;

  int n_coef_        = 0;
  int n_dofs_        = 0;
  int n_constraints_ = 0;
  int h_nnz_         = 0;
  bool use_symmetric_constraint_hessian_ = false;

  bool setup_done_                   = false;
  bool sparse_hessian_initialized_   = false;
  bool analysis_done_                = false;
  bool factorization_done_           = false;

  double* d_v_guess_        = nullptr;
  double* d_v_prev_         = nullptr;
  double* d_lambda_guess_   = nullptr;
  double* d_g_              = nullptr;
  double* d_delta_v_        = nullptr;
  double* d_r_              = nullptr;
  double* d_x12_prev_       = nullptr;
  double* d_y12_prev_       = nullptr;
  double* d_z12_prev_       = nullptr;
  double* d_v_trial_backup_ = nullptr;
  double* d_norm_temp_      = nullptr;

  int* d_csr_row_offsets_ = nullptr;
  int* d_csr_col_indices_ = nullptr;
  double* d_csr_values_   = nullptr;

  cublasHandle_t cublas_handle_ = nullptr;
  cudssHandle_t cudss_handle_   = nullptr;
  cudssConfig_t cudss_config_   = nullptr;
  cudssData_t cudss_data_       = nullptr;

  void InitializeBlockInfos();
  void ValidateProblemConfiguration() const;
  void AnalyzeHessianSparsity();
  void BuildConstraintAwareSparsityPattern(std::vector<int>& row_offsets,
                                           std::vector<int>& col_indices);
  void DistributeExternalForces();
  void BackupCurrentPositions();
  void UpdateProblemStateFromVelocity() const;
  void EvaluateState();
  void AssembleGradient();
  void AssembleHessian();
  double ComputeL2Norm(double* d_vec, int n_entries) const;
  void UpdateProblemVelocity() const;
};
