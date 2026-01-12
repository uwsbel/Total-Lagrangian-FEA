/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    MaterialModel.cuh
 * Brief:   Defines material model identifiers for constitutive laws used by
 *          element kernels (e.g., SVK, Mooney-Rivlin).
 *==============================================================
 *==============================================================*/

#pragma once

enum MaterialModel : int {
  MATERIAL_MODEL_SVK           = 0,
  MATERIAL_MODEL_MOONEY_RIVLIN = 1,
};
