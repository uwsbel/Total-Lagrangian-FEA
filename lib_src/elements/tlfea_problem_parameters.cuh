#pragma once

const int totalN_Elements = 1000;  // Total number of elements in the mesh; used as a guard to avoid out-of-bounds access.

const float minJthreshold = 1e-6f; // minimum value of the determinant of the deformation gradient F to avoid division by zero.

const float mu10 = 1.0f; // first Lame parameter
const float mu01 = 1.0f; // second Lame parameter
const float bulkK = 1.0f; // bulk modulus