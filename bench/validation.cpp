#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "utils/options.hpp"

void poiseuille(float* ux)
{
    int xCheck = options::Nx / 2;
    std::vector<float> uProfile(options::Ny);
    for (int y = 0; y < options::Ny; ++y) {
        int i = y * options::Nx + xCheck;
        uProfile[y] = ux[i];
    }

    float uMax = *std::max_element(uProfile.begin(), uProfile.end());

    double errSquaredSum = 0.0;
    for (int y = 0; y < options::Ny; ++y) {
        float yNorm = (y - options::Ny / 2.0f) / (options::Ny / 2.0f);
        float uAnalytic = uMax * (1.0f - yNorm * yNorm);
        float diff = uProfile[y] - uAnalytic;
        errSquaredSum += diff * diff;
    }
    double rmse = std::sqrt(errSquaredSum / options::Ny);
    std::cout << "RMSE vs Poiseuille analitico: " << rmse << std::endl;

    double relError = rmse / uMax;
    std::cout << "Errore relativo: " << (relError * 100) << "%\n";
}