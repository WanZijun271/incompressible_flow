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
constexpr int id_aC = 1;
constexpr int id_aE = 2;
constexpr int id_aW = 3;
constexpr int id_aN = 4;
constexpr int id_aS = 5;
constexpr int id_aT = 6;
constexpr int id_aB = 7;

constexpr scalar dx = (xmax - xmin) / (scalar)nx;
constexpr scalar dy = (ymax - ymin) / (scalar)ny;
constexpr scalar dz = (zmax - zmin) / (scalar)nz;

constexpr scalar areaE = dy * dz;
constexpr scalar areaW = dy * dz;
constexpr scalar areaN = dz * dx;
constexpr scalar areaS = dz * dx;
constexpr scalar areaT = dx * dy;
constexpr scalar areaB = dx * dy;

#endif