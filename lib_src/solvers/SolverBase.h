#pragma once

class SolverBase {
 public:
  virtual ~SolverBase() = default;
  virtual void Solve()  = 0;

  // Generic parameter setter
  virtual void SetParameters(void *params) = 0;
};