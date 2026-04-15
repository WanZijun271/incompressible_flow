#include "kernels.cuh"
#include "config.h"
#include "constants.h"
#include <cmath>
#include <algorithm>

using namespace std;

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
            newTemp /= coef[id+id_aP];
        } else if (dim == 3) {
            int id = i * 8 + j * nx * 8 + k * nx * ny * 8;
            newTemp = coef[id+id_b]
                - coef[id+id_aE] * tempE
                - coef[id+id_aW] * tempW
                - coef[id+id_aN] * tempN
                - coef[id+id_aS] * tempS
                - coef[id+id_aT] * tempT
                - coef[id+id_aB] * tempB;
            newTemp /= coef[id+id_aP];
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

void pointJacobiIterate(vector<scalar>& tempField, const vector<scalar>& coef) {

    size_t tempFieldSize = tempField.size() * sizeof(scalar);
    size_t coefSize = coef.size() * sizeof(scalar);

    scalar *devTempField, *devTempField0, *devCoef, *devNorm;
    cudaMalloc(&devTempField, tempFieldSize);
    cudaMalloc(&devTempField0, tempFieldSize);
    cudaMalloc(&devCoef, coefSize);
    cudaMalloc(&devNorm, sizeof(scalar));

    cudaMemcpy(devTempField, tempField.data(), tempFieldSize, cudaMemcpyHostToDevice);
    cudaMemset(devTempField0, 0, tempFieldSize);
    cudaMemcpy(devCoef, coef.data(), coefSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock;
    dim3 numBlocks;
    if (dim == 2) {
        threadsPerBlock.x = 32;
        threadsPerBlock.y = 32;
        threadsPerBlock.z = 1;

        numBlocks.x = (nx + 31) / 32;
        numBlocks.y = (ny + 31) / 32;
        numBlocks.z = 1;
    } else if (dim == 3) {
        threadsPerBlock.x = 16;
        threadsPerBlock.y = 8;
        threadsPerBlock.z = 8;

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
            newTemp /= coef[id+id_aP];
        } else if (dim == 3) {
            int id = i * 8 + j * nx * 8 + k * nx * ny * 8;
            newTemp = coef[id+id_b]
                - coef[id+id_aE] * tempE
                - coef[id+id_aW] * tempW
                - coef[id+id_aN] * tempN
                - coef[id+id_aS] * tempS
                - coef[id+id_aT] * tempT
                - coef[id+id_aB] * tempB;
            newTemp /= coef[id+id_aP];
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

void GaussSeidelIterate(vector<scalar>& tempField, const vector<scalar>& coef) {

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
        threadsPerBlock.x = 32;
        threadsPerBlock.y = 32;
        threadsPerBlock.z = 1;

        numBlocks.x = (nx + 31) / 32;
        numBlocks.y = (ny + 31) / 32;
        numBlocks.z = 1;
    } else if (dim == 3) {
        threadsPerBlock.x = 16;
        threadsPerBlock.y = 8;
        threadsPerBlock.z = 8;

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