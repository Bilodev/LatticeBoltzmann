#include <sycl/sycl.hpp>

#include "core/LBM_DATA.hpp"
#define LBM

namespace lbm {

void computeMacroscopics(sycl::queue& q, float* f, float* rho, float* ux,
                         float* uy, const uint8_t* solid, int Nx, int Ny)
{
    q.submit([&](sycl::handler& h) {
         h.parallel_for(sycl::range<1>(Nx * Ny), [=](sycl::id<1> idx) {
             int i = idx[0];
             if (solid[i]) {
                 rho[i] = 1.0f;
                 ux[i] = 0.0f;
                 uy[i] = 0.0f;
                 return;
             }

             float r = 0.0f, u = 0.0f, v = 0.0f;
             for (int k = 0; k < lbm::Q; ++k) {
                 float fk = f[i * lbm::Q + k];
                 r += fk;
                 u += fk * lbm::cx[k];
                 v += fk * lbm::cy[k];
             }
             rho[i] = r;
             ux[i] = u / r;
             uy[i] = v / r;
         });
     }).wait();
}

void collision(sycl::queue& q, float* f, const float* rho, const float* ux,
               const float* uy, const uint8_t* solid, float tau, int Nx, int Ny)
{
    float invTau = 1.0f / tau;
    q.submit([&](sycl::handler& h) {
         h.parallel_for(sycl::range<1>(Nx * Ny), [=](sycl::id<1> idx) {
             int i = idx[0];
             if (solid[i]) return;  // niente collision nelle celle solide

             float r = rho[i], u = ux[i], v = uy[i];
             for (int k = 0; k < lbm::Q; ++k) {
                 float eq = lbm::feq(k, r, u, v);
                 f[i * lbm::Q + k] -= invTau * (f[i * lbm::Q + k] - eq);
             }
         });
     }).wait();
}

void streaming(sycl::queue& q, const float* f, float* f_new,
               const uint8_t* solid, int Nx, int Ny)
{
    q.submit([&](sycl::handler& h) {
         h.parallel_for(sycl::range<2>(Nx, Ny), [=](sycl::id<2> idx) {
             int x = idx[0];
             int y = idx[1];
             int i = y * Nx + x;

             for (int k = 0; k < lbm::Q; ++k) {
                 int xs = x - lbm::cx[k];
                 int ys = y - lbm::cy[k];

                 // fuori dal dominio su un qualsiasi lato -> gestito dopo dai
                 // BC
                 if (xs < 0 || xs >= Nx || ys < 0 || ys >= Ny) {
                     continue;
                 }

                 int si = ys * Nx + xs;

                 if (solid[si]) {
                     f_new[i * lbm::Q + k] = f[i * lbm::Q + lbm::opposite[k]];
                 }
                 else {
                     f_new[i * lbm::Q + k] = f[si * lbm::Q + k];
                 }
             }
         });
     }).wait();
}

void applyBoundaries(sycl::queue& q, float* f, float U0, int Nx, int Ny)
{
    // INLET a sinistra: velocità fissa
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(Ny), [=](sycl::id<1> idx) {
            int y = idx[0];
            int i0 = y * Nx + 0;
            for (int k = 0; k < lbm::Q; ++k)
                f[i0 * lbm::Q + k] = lbm::feq(k, 1.0f, U0, 0.0f);
        });
    });

    // OUTLET a destra: zero-gradient
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(Ny), [=](sycl::id<1> idx) {
            int y = idx[0];
            int iN = y * Nx + (Nx - 1);
            int iPrev = y * Nx + (Nx - 2);
            for (int k = 0; k < lbm::Q; ++k)
                f[iN * lbm::Q + k] = f[iPrev * lbm::Q + k];
        });
    });

    // OUTLET in alto (y=0): zero-gradient, copia dalla riga sotto
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(Nx), [=](sycl::id<1> idx) {
            int x = idx[0];
            int i0 = 0 * Nx + x;
            int iNext = 1 * Nx + x;
            for (int k = 0; k < lbm::Q; ++k)
                f[i0 * lbm::Q + k] = f[iNext * lbm::Q + k];
        });
    });

    // OUTLET in basso (y=Ny-1): zero-gradient, copia dalla riga sopra
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(Nx), [=](sycl::id<1> idx) {
            int x = idx[0];
            int iN = (Ny - 1) * Nx + x;
            int iPrev = (Ny - 2) * Nx + x;
            for (int k = 0; k < lbm::Q; ++k)
                f[iN * lbm::Q + k] = f[iPrev * lbm::Q + k];
        });
    });

    q.wait();  // un solo wait finale per tutti e 4 i submit
}
}  // namespace lbm
