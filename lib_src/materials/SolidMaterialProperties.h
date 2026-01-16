/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    SolidMaterialProperties.h
 * Brief:   Defines material properties for deformable solid bodies.
 *          Encapsulates Young's modulus, Poisson's ratio, density,
 *          damping parameters, and material model selection for use
 *          with T10, ANCF3243, and ANCF3443 elements.
 *==============================================================
 *==============================================================*/

#pragma once

#include "MaterialModel.cuh"

/**
 * Encapsulates physical/material properties for a deformable solid body.
 * This struct holds all parameters needed to define the constitutive
 * behavior, density, and damping of a finite element body.
 */
struct SolidMaterialProperties {
  // Elastic properties (SVK model)
  double E   = 1e7;     // Young's modulus (Pa)
  double nu  = 0.3;     // Poisson's ratio (dimensionless)

  // Density
  double rho0 = 1000.0; // Reference density (kg/m³)

  // Damping parameters (Kelvin-Voigt model)
  double eta_damp    = 0.0;  // Shear-like damping coefficient
  double lambda_damp = 0.0;  // Volumetric-like damping coefficient

  // Material model selection
  int material_model = MATERIAL_MODEL_SVK;

  // Mooney-Rivlin parameters (used when material_model == MATERIAL_MODEL_MOONEY_RIVLIN)
  double mu10  = 0.0;  // First Mooney-Rivlin coefficient
  double mu01  = 0.0;  // Second Mooney-Rivlin coefficient
  double kappa = 0.0;  // Bulk modulus (volumetric penalty)

  // Computed Lamé parameters for SVK model
  double mu() const { return E / (2.0 * (1.0 + nu)); }
  double lambda() const { return (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu)); }

  // Default constructor
  SolidMaterialProperties() = default;

  // Constructor for SVK material with basic properties
  SolidMaterialProperties(double E_, double nu_, double rho0_)
      : E(E_), nu(nu_), rho0(rho0_), material_model(MATERIAL_MODEL_SVK) {}

  // Full constructor for SVK with damping
  SolidMaterialProperties(double E_, double nu_, double rho0_,
                          double eta_damp_, double lambda_damp_)
      : E(E_), nu(nu_), rho0(rho0_),
        eta_damp(eta_damp_), lambda_damp(lambda_damp_),
        material_model(MATERIAL_MODEL_SVK) {}

  // Static factory: Create SVK material
  static SolidMaterialProperties SVK(double E, double nu, double rho0,
                                     double eta_damp = 0.0,
                                     double lambda_damp = 0.0) {
    SolidMaterialProperties props;
    props.E              = E;
    props.nu             = nu;
    props.rho0           = rho0;
    props.eta_damp       = eta_damp;
    props.lambda_damp    = lambda_damp;
    props.material_model = MATERIAL_MODEL_SVK;
    return props;
  }

  // Static factory: Create Mooney-Rivlin material
  static SolidMaterialProperties MooneyRivlin(double mu10, double mu01,
                                              double kappa, double rho0,
                                              double eta_damp = 0.0,
                                              double lambda_damp = 0.0) {
    SolidMaterialProperties props;
    props.rho0           = rho0;
    props.eta_damp       = eta_damp;
    props.lambda_damp    = lambda_damp;
    props.material_model = MATERIAL_MODEL_MOONEY_RIVLIN;
    props.mu10           = mu10;
    props.mu01           = mu01;
    props.kappa          = kappa;
    // E and nu are not used for Mooney-Rivlin but set reasonable defaults
    props.E              = 0.0;
    props.nu             = 0.0;
    return props;
  }
};
