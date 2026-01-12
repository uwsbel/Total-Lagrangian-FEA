#pragma once

constexpr int totalN_Elements = 1000;  // Total number of elements in the mesh; used as a guard to avoid out-of-bounds access.

constexpr float minJthreshold = 1e-6f; // minimum value of the determinant of the deformation gradient F to avoid division by zero.

constexpr float mu10 = 1.0f; // first Lame parameter
constexpr float mu01 = 1.0f; // second Lame parameter
constexpr float bulkK = 1.0f; // bulk modulus