#ifndef KERNELS_H
#define KERNELS_H

#include "types.h"
#include <vector>

void pointJacobiIterate(std::vector<scalar>& tempField, const std::vector<scalar>& coef);

void GaussSeidelIterate(std::vector<scalar>& tempField, const std::vector<scalar>& coef);

#endif