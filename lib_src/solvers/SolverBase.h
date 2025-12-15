/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    SolverBase.h
 * Brief:   Declares the abstract SolverBase interface for RoboDyna solvers.
 *          Provides a minimal polymorphic API (Solve and SetParameters) that
 *          is implemented by GPU-synchronized optimizers and Newton solvers
 *          such as SyncedAdamW, SyncedNesterov, and SyncedNewton.
 *==============================================================
 *==============================================================*/

#pragma once

class SolverBase {
 public:
  virtual ~SolverBase() = default;
  virtual void Solve()  = 0;

  // Generic parameter setter
  virtual void SetParameters(void *params) = 0;
};