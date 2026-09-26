#include <chrono>
#ifndef LBM
#include "lib/core/lbm.cpp"
#endif

void runHeadless(sycl::queue& q, float* f, float* f_new, float* rho, float* ux,
                 float* uy, const uint8_t* solidPtr, int Nx, int Ny, float tau,
                 float U0, int nSteps)
{
    constexpr int warmupSteps = 500;
    for (int step = 0; step < warmupSteps; ++step) {
        lbm::computeMacroscopics(q, f, rho, ux, uy, solidPtr, Nx, Ny);
        lbm::collision(q, f, rho, ux, uy, solidPtr, tau, Nx, Ny);
        lbm::streaming(q, f, f_new, solidPtr, Nx, Ny);
        lbm::applyBoundaries(q, f_new, U0, Nx, Ny);
        std::swap(f, f_new);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < nSteps; ++step) {
        lbm::computeMacroscopics(q, f, rho, ux, uy, solidPtr, Nx, Ny);
        lbm::collision(q, f, rho, ux, uy, solidPtr, tau, Nx, Ny);
        lbm::streaming(q, f, f_new, solidPtr, Nx, Ny);
        lbm::applyBoundaries(q, f_new, U0, Nx, Ny);
        std::swap(f, f_new);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();
    double mlups = (double(Nx) * Ny * nSteps) / (seconds * 1e6);

    std::cout << "Steps: " << nSteps << "\n";
    std::cout << "Time: " << seconds << " s\n";
    std::cout << "MLUPS: " << mlups << "\n";
}
