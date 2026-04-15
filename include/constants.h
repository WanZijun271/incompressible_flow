#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "config.h"
#include "types.h"

constexpr int east   = 0;
constexpr int west   = 1;
constexpr int north  = 2;
constexpr int south  = 3;
constexpr int top    = 4;
constexpr int bottom = 5;

constexpr int id_b  = 0;
constexpr int id_aP = 1;
constexpr int id_aE = 2;
constexpr int id_aW = 3;
constexpr int id_aN = 4;
constexpr int id_aS = 5;
constexpr int id_aT = 6;
constexpr int id_aB = 7;

const scalar dx = (xmax - xmin) / (scalar)nx;
const scalar dy = (ymax - ymin) / (scalar)ny;
const scalar dz = (zmax - zmin) / (scalar)nz;

#endif