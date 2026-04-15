#include "kernels.cuh"
#include "config.h"
#include "constants.h"
#include <cmath>
#include <algorithm>

using namespace std;

__global__ void initUfKernel(scalar *u, scalar *uf) {

    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int id_c = i + (nx+1) * (j + ny * k);
        int id_W = i-1 + nx * (j + ny * k);
        int id_E = i + nx * (j + ny * k);
        uf[id_c] = (u[id_W] + u[id_E]) / 2;
    }
}

__global__ void initVfKernel(scalar *v, scalar *vf) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int id_c = j + (ny+1) * (k + nz * i);
        int id_S = j-1 + ny * (k + nz * i);
        int id_N = j + ny * (k + nz * i);
        vf[id_c] = (v[id_S] + v[id_N]) / 2;
    }
}

__global__ void initWfKernel(scalar *w, scalar *wf) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i < nx && j < ny && k < nz) {
        int id_c = k + (nz+1) * (i + nx * j);
        int id_B = k-1 + nz * (i + nx * j);
        int id_T = k + nz * (i + nx * j);
        wf[id_c] = (w[id_B] + w[id_T]) / 2;
    }
}

void initFaceVel(scalar *u_dev, scalar *v_dev, scalar *w_dev, scalar *uf_dev, scalar *vf_dev, scalar *wf_dev) {

    cudaStream_t stream[dim];
    for (int i = 0; i < dim; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    dim3 threadsPerBlock;
    dim3 numBlocks;
    if (dim == 2) {
        threadsPerBlock = dim3(32, 32, 1);
    } else if (dim == 3) {
        threadsPerBlock = dim3(16, 16, 8);
    }

    numBlocks.x = (nx - 1 + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;
    initUfKernel<<<numBlocks, threadsPerBlock, 0, stream[0]>>>(u_dev, uf_dev);

    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny - 1 + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;
    initVfKernel<<<numBlocks, threadsPerBlock, 0, stream[1]>>>(v_dev, vf_dev);

    if (dim == 3) {
        numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = (nz - 1 + threadsPerBlock.z - 1) / threadsPerBlock.z;
        initWfKernel<<<numBlocks, threadsPerBlock, 0, stream[2]>>>(w_dev, wf_dev);
    }

    for (int i = 0; i < dim; ++i) {
        cudaStreamSynchronize(stream[i]);
    }

    for (int i = 0; i < dim; ++i) {
        cudaStreamDestroy(stream[i]);
    }
}

__global__ void applyUfBCsKernel(scalar *uf, scalar *u, int i, int type, scalar val) {

    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (j < ny && k < nz) {
        int id = i + (nx+1) * (j + ny * k);
        if (type == 0) { // "wall"
            uf[id] = 0;
        } else if (type == 1) { // "inlet"
            uf[id] = val;
        } else if (type == 2) { // "outlet"
            if (i == 0) {
                uf[id] = u[0 + nx * (j + ny * k)];
            } else if (i == nx) {
                uf[id] = u[nx-1 + nx * (j + ny * k)];
            }
        }
    }
}

__global__ void applyVfBCsKernel(scalar *vf, scalar *v, int j, int type, scalar val) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && k < nz) {
        int id = j + (ny+1) * (k + nz * i);
        if (type == 0) { // "wall"
            vf[id] = 0;
        } else if (type == 1) { // "inlet"
            vf[id] = val;
        } else if (type == 2) { // "outlet"
            if (j == 0) {
                vf[id] = v[0 + ny * (k + nz * i)];
            } else if (j == ny) {
                vf[id] = v[ny-1 + ny * (k + nz * i)];
            }
        }
    }
}

__global__ void applyWfBCsKernel(scalar *wf, scalar *w, int k, int type, scalar val) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx && j < ny) {
        int id = k + (nz+1) * (i + nx * j);
        if (type == 0) { // "wall"
            wf[id] = 0;
        } else if (type == 1) { // "inlet"
            wf[id] = val;
        } else if (type == 2) { // "outlet"
            if (k == 0) {
                wf[id] = w[0 + nz * (i + nx * j)];
            } else if (k == nz) {
                wf[id] = w[nz-1 + nz * (i + nx * j)];
            }
        }
    }
}

void applyFaceVelBCs(scalar *uf_dev, scalar *vf_dev, scalar *wf_dev, scalar *u_dev, scalar *v_dev, scalar *w_dev) {

    cudaStream_t stream[2*dim];
    for (int i = 0; i < 2*dim; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    dim3 threadsPerBlock;
    dim3 numBlocks;

    if (dim == 2) {
        threadsPerBlock = dim3(1, 1024, 1);
    } else if (dim == 3) {
        threadsPerBlock = dim3(1, 32, 32);
    }
    numBlocks.x = 1;
    numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;
    applyUfBCsKernel<<<numBlocks, threadsPerBlock, 0, stream[0]>>>(uf_dev, u_dev, 0, velBCs::type[west], velBCs::val[west][0]);
    applyUfBCsKernel<<<numBlocks, threadsPerBlock, 0, stream[1]>>>(uf_dev, u_dev, nx, velBCs::type[east], velBCs::val[east][0]);

    if (dim == 2) {
        threadsPerBlock = dim3(1024, 1, 1);
    } else if (dim == 3) {
        threadsPerBlock = dim3(32, 1, 32);
    }
    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = 1;
    numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;
    applyVfBCsKernel<<<numBlocks, threadsPerBlock, 0, stream[2]>>>(vf_dev, v_dev, 0, velBCs::type[south], velBCs::val[south][1]);
    applyVfBCsKernel<<<numBlocks, threadsPerBlock, 0, stream[3]>>>(vf_dev, v_dev, ny, velBCs::type[north], velBCs::val[north][1]);

    if (dim == 3) {
        threadsPerBlock = dim3(32, 32, 1);
        numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = 1;
        applyWfBCsKernel<<<numBlocks, threadsPerBlock, 0, stream[4]>>>(wf_dev, w_dev, 0, velBCs::type[bottom], velBCs::val[bottom][2]);
        applyWfBCsKernel<<<numBlocks, threadsPerBlock, 0, stream[5]>>>(wf_dev, w_dev, nz, velBCs::type[top], velBCs::val[top][2]);
    }

    for (int i = 0; i < 2*dim; ++i) {
        cudaStreamSynchronize(stream[i]);
    }

    for (int i = 0; i < 2*dim; ++i) {
        cudaStreamDestroy(stream[i]);
    }
}

__global__ void calcMomentumLinkCoefKernel(scalar *coef, scalar *uf, scalar *vf, scalar *wf) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int nTerms = 2 + 2 * dim;
        int id = nTerms * (i + nx * (j + ny * k));

        scalar ue = uf[i+1 + (nx+1) * (j + ny * k)];
        scalar uw = uf[i + (nx+1) * (j + ny * k)];
        scalar vn = vf[j+1 + (ny+1) * (k + nz * i)];
        scalar vs = vf[j + (ny+1) * (k + nz * i)];

        coef[id+id_aE] = (density * (ue - abs(ue)) * areaE) / 2 - dynamicViscosity * areaE / dx;
        coef[id+id_aW] = (density * (-uw - abs(uw)) * areaW) / 2 - dynamicViscosity * areaW / dx;
        coef[id+id_aN] = (density * (vn - abs(vn)) * areaN) / 2 - dynamicViscosity * areaN / dy;
        coef[id+id_aS] = (density * (-vs - abs(vs)) * areaS) / 2 - dynamicViscosity * areaS / dy;
        coef[id+id_aC] = -(coef[id+id_aE] + coef[id+id_aW] + coef[id+id_aN] + coef[id+id_aS]);

        if (dim == 3) {
            scalar wt = wf[k+1 + (nz+1) * (i + nx * j)];
            scalar wb = wf[k + (nz+1) * (i + nx * j)];

            coef[id+id_aT] = (density * (wt - abs(wt)) * areaT) / 2 - dynamicViscosity * areaT / dz;
            coef[id+id_aB] = (density * (-wb - abs(wb)) * areaB) / 2 - dynamicViscosity * areaB / dz;
            coef[id+id_aC] += -(coef[id+id_aT] + coef[id+id_aB]);
        }
    }
}

void calcMomentumLinkCoef(scalar *coef_dev, scalar *uf_dev, scalar *vf_dev, scalar *wf_dev) {

    dim3 threadsPerBlock;
    dim3 numBlocks; 

    if (dim == 2) {
        threadsPerBlock = dim3(32, 32, 1);
    } else if (dim == 3) {
        threadsPerBlock = dim3(16, 8, 8);
    }
    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;

    calcMomentumLinkCoefKernel<<<numBlocks, threadsPerBlock>>>(coef_dev, uf_dev, vf_dev, wf_dev);

    cudaDeviceSynchronize();
}

__global__ void pointJacobiIterateKernel(scalar *tempField, scalar* tempField0, scalar *coef, scalar *norm) {

    extern __shared__ scalar sharedNorm[];

    int threadId = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int idP = i + j * nx + k * nx * ny;
        scalar tempP = tempField0[idP];
        // tE
        scalar tempE = 0.0;
        if ( i != nx - 1 ) {
            int idE = (i+1) + j * nx + k * nx * ny;
            tempE = tempField0[idE];
        }
        // tW
        scalar tempW = 0.0;
        if ( i != 0 ) {
            int idW = (i-1) + j * nx + k * nx * ny;
            tempW = tempField0[idW];
        }
        // tN
        scalar tempN = 0.0;
        if ( j != ny - 1 ) {
            int idN = i + (j+1) * nx + k * nx * ny;
            tempN = tempField0[idN];
        }
        // tS
        scalar tempS = 0.0;
        if ( j != 0 ) {
            int idS = i + (j-1) * nx + k * nx * ny;
            tempS = tempField0[idS];
        }
        // tT
        scalar tempT = 0.0;
        if ( k != nz - 1 ) {
            int idT = i + j * nx + (k+1) * nx * ny;
            tempT = tempField0[idT];
        }
        // tB
        scalar tempB = 0.0;
        if ( k != 0 ) {
            int idB = i + j * nx + (k-1) * nx * ny;
            tempB = tempField0[idB];
        }
        
        scalar newTemp = tempP;
        if (dim == 2) {
            int id = i * 6 + j * nx * 6;
            newTemp = coef[id+id_b]
                - coef[id+id_aE] * tempE
                - coef[id+id_aW] * tempW
                - coef[id+id_aN] * tempN
                - coef[id+id_aS] * tempS;
            newTemp /= coef[id+id_aC];
        } else if (dim == 3) {
            int id = i * 8 + j * nx * 8 + k * nx * ny * 8;
            newTemp = coef[id+id_b]
                - coef[id+id_aE] * tempE
                - coef[id+id_aW] * tempW
                - coef[id+id_aN] * tempN
                - coef[id+id_aS] * tempS
                - coef[id+id_aT] * tempT
                - coef[id+id_aB] * tempB;
            newTemp /= coef[id+id_aC];
        }

        scalar dT = relax * (newTemp - tempP);

        tempField[idP] = tempP + dT;

        sharedNorm[threadId] = dT*dT;
        __syncthreads();

        int stride = blockDim.x * blockDim.y * blockDim.z;
        for (int s = stride / 2; s > 0; s >>= 1) {
            if (threadId < s) {
                sharedNorm[threadId] += sharedNorm[threadId + s];
            }
            __syncthreads();
        }

        if (threadId == 0) {
            atomicAdd(norm, sharedNorm[0]);
        }
    }
}

void pointJacobiIterate(vector<scalar> &tempField, const vector<scalar> &coef) {

    size_t tempFieldSize = tempField.size() * sizeof(scalar);
    size_t coefSize = coef.size() * sizeof(scalar);

    scalar *devTempField, *devTempField0, *devCoef, *devNorm;
    cudaMalloc(&devTempField, tempFieldSize);
    cudaMalloc(&devTempField0, tempFieldSize);
    cudaMalloc(&devCoef, coefSize);
    cudaMalloc(&devNorm, sizeof(scalar));

    cudaMemcpy(devTempField, tempField.data(), tempFieldSize, cudaMemcpyHostToDevice);
    cudaMemset(devTempField0, 0.0, tempFieldSize);
    cudaMemcpy(devCoef, coef.data(), coefSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock;
    dim3 numBlocks;
    if (dim == 2) {
        threadsPerBlock = dim3(32, 32, 1);

        numBlocks.x = (nx + 31) / 32;
        numBlocks.y = (ny + 31) / 32;
        numBlocks.z = 1;
    } else if (dim == 3) {
        threadsPerBlock = dim3(16, 16, 8);

        numBlocks.x = (nx + 15) / 16;
        numBlocks.y = (ny + 7) / 8;
        numBlocks.z = (nz + 7) / 8;
    }

    int blockSize = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z;

    scalar maxNorm = -1e20;

    for (int it = 0; it < niter; ++it) {
        
        scalar *tmp = devTempField;
        devTempField = devTempField0;
        devTempField0 = tmp;

        scalar norm = 0.0;
        cudaMemcpy(devNorm, &norm, sizeof(scalar), cudaMemcpyHostToDevice);

        pointJacobiIterateKernel<<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize>>>(devTempField, devTempField0, devCoef, devNorm);
        
        cudaMemcpy(&norm, devNorm, sizeof(scalar), cudaMemcpyDeviceToHost);

        norm = sqrt(norm / (nx * ny * nz));

        maxNorm = max(norm, maxNorm);

        scalar relNorm = norm / (maxNorm + 1e-20);    // relative residual
        if (relNorm < tol) {
            break;
        }
    }

    cudaMemcpy(tempField.data(), devTempField, tempFieldSize, cudaMemcpyDeviceToHost);

    cudaFree(devTempField);
    cudaFree(devTempField0);
    cudaFree(devCoef);
    cudaFree(devNorm);
}

__global__ void GaussSeidelIterateKernel(scalar *tempField, scalar *coef, scalar *norm) {

    extern __shared__ scalar sharedNorm[];

    int threadId = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int idP = i + j * nx + k * nx * ny;
        scalar tempP = tempField[idP];
        // tE
        scalar tempE = 0.0;
        if ( i != nx - 1 ) {
            int idE = (i+1) + j * nx + k * nx * ny;
            tempE = tempField[idE];
        }
        // tW
        scalar tempW = 0.0;
        if ( i != 0 ) {
            int idW = (i-1) + j * nx + k * nx * ny;
            tempW = tempField[idW];
        }
        // tN
        scalar tempN = 0.0;
        if ( j != ny - 1 ) {
            int idN = i + (j+1) * nx + k * nx * ny;
            tempN = tempField[idN];
        }
        // tS
        scalar tempS = 0.0;
        if ( j != 0 ) {
            int idS = i + (j-1) * nx + k * nx * ny;
            tempS = tempField[idS];
        }
        // tT
        scalar tempT = 0.0;
        if ( k != nz - 1 ) {
            int idT = i + j * nx + (k+1) * nx * ny;
            tempT = tempField[idT];
        }
        // tB
        scalar tempB = 0.0;
        if ( k != 0 ) {
            int idB = i + j * nx + (k-1) * nx * ny;
            tempB = tempField[idB];
        }
        
        scalar newTemp = tempP;
        if (dim == 2) {
            int id = i * 6 + j * nx * 6;
            newTemp = coef[id+id_b]
                - coef[id+id_aE] * tempE
                - coef[id+id_aW] * tempW
                - coef[id+id_aN] * tempN
                - coef[id+id_aS] * tempS;
            newTemp /= coef[id+id_aC];
        } else if (dim == 3) {
            int id = i * 8 + j * nx * 8 + k * nx * ny * 8;
            newTemp = coef[id+id_b]
                - coef[id+id_aE] * tempE
                - coef[id+id_aW] * tempW
                - coef[id+id_aN] * tempN
                - coef[id+id_aS] * tempS
                - coef[id+id_aT] * tempT
                - coef[id+id_aB] * tempB;
            newTemp /= coef[id+id_aC];
        }

        scalar dT = relax * (newTemp - tempP);

        tempField[idP] = tempP + dT;

        sharedNorm[threadId] = dT*dT;
        __syncthreads();

        int stride = blockDim.x * blockDim.y * blockDim.z;
        for (int s = stride / 2; s > 0; s >>= 1) {
            if (threadId < s) {
                sharedNorm[threadId] += sharedNorm[threadId + s];
            }
            __syncthreads();
        }

        if (threadId == 0) {
            atomicAdd(norm, sharedNorm[0]);
        }
    }
}

void GaussSeidelIterate(vector<scalar> &tempField, const vector<scalar> &coef) {

    size_t tempFieldSize = tempField.size() * sizeof(scalar);
    size_t coefSize = coef.size() * sizeof(scalar);

    scalar *devTempField, *devCoef, *devNorm;
    cudaMalloc(&devTempField, tempFieldSize);
    cudaMalloc(&devCoef, coefSize);
    cudaMalloc(&devNorm, sizeof(scalar));

    cudaMemcpy(devTempField, tempField.data(), tempFieldSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devCoef, coef.data(), coefSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock;
    dim3 numBlocks;
    if (dim == 2) {
        threadsPerBlock = dim3(32, 32, 1);

        numBlocks.x = (nx + 31) / 32;
        numBlocks.y = (ny + 31) / 32;
        numBlocks.z = 1;
    } else if (dim == 3) {
        threadsPerBlock = dim3(16, 16, 8);

        numBlocks.x = (nx + 15) / 16;
        numBlocks.y = (ny + 7) / 8;
        numBlocks.z = (nz + 7) / 8;
    }

    int blockSize = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z;

    scalar maxNorm = -1e20;

    for (int it = 0; it < niter; ++it) {

        scalar norm = 0.0;
        cudaMemcpy(devNorm, &norm, sizeof(scalar), cudaMemcpyHostToDevice);

        GaussSeidelIterateKernel<<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize>>>(devTempField, devCoef, devNorm);
        
        cudaMemcpy(&norm, devNorm, sizeof(scalar), cudaMemcpyDeviceToHost);

        norm = sqrt(norm / (nx * ny * nz));

        maxNorm = max(norm, maxNorm);

        scalar relNorm = norm / (maxNorm + 1e-20);    // relative residual
        if (relNorm < tol) {
            break;
        }
    }

    cudaMemcpy(tempField.data(), devTempField, tempFieldSize, cudaMemcpyDeviceToHost);

    cudaFree(devTempField);
    cudaFree(devCoef);
    cudaFree(devNorm);
}