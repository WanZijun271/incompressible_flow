#ifndef KERNELS_H
#define KERNELS_H

#include "types.h"
#include <vector>

void initFaceVel(scalar *u_dev, scalar *v_dev, scalar *w_dev, scalar *uf_dev, scalar *vf_dev, scalar *wf_dev);

void applyFaceVelBCs(scalar *uf_dev, scalar *vf_dev, scalar *wf_dev, scalar *u_dev, scalar *v_dev, scalar *w_dev);

void calcMomLinkCoef(scalar *coef_dev, scalar *uf_dev, scalar *vf_dev, scalar *wf_dev);

void calcMomSrcTerm(scalar *uSrcTerm_dev, scalar *vSrcTerm_dev, scalar *wSrcTerm_dev, scalar *p_dev);

void pointJacobiIterate(std::vector<scalar> &tempField, const std::vector<scalar> &coef);

void GaussSeidelIterate(std::vector<scalar> &tempField, const std::vector<scalar> &coef);

#endif