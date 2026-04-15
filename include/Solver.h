#ifndef SOLVER_H
#define SOLVER_H

#include <vector>
#include <string>
#include "types.h"

class Solver {
private:
    std::vector<scalar> _tempField;        // temperature field

    std::vector<scalar> _coef;             // coefficient

public:
    Solver();
    void initTempField(scalar temp);                     // initialize temperature field
    void pointJacobiSolver();                            // solve the equation using the Jacobi interative method
    void GaussSeidelSolver();                            // solve the equation using the Gauss-Seidel interative method
    void writeVTK(const std::string& filepath) const;    // write the result to .vtk file
private:
    void calcCoef();                                     // calculate the coefficients of the equation
};

#endif