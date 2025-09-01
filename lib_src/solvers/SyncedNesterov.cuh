#pragma once
#include "SolverBase.h"
#include "../GPUMemoryManager.cuh"

struct NesterovParams
{
    double alpha, rho, inner_tol, outer_tol;
    int max_outer, max_inner;
    double penalty;
};

class NesterovSolver : public SolverBase
{
public:
    NesterovSolver(GPU_ANCF3243_Data *data) : data_(data) {}

    void SetParameters(void *params) override
    {
        NesterovParams *p = static_cast<NesterovParams *>(params);
        data_->SetNesterovParameters(p->alpha, p->rho, p->inner_tol, p->outer_tol, p->max_outer, p->max_inner, p->penalty);
    }

    void Solve() override
    {
        data_->OneStepNesterov();
    }

private:
    GPU_ANCF3243_Data *data_;
};