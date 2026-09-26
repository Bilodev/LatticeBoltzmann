#include <iostream>
//
#include "bench/bench.cpp"
#include "view/view.cpp"

enum MODES { STD, HEADLESS };

int main(int argc, char** argv)
{
    MODES mode = STD;
    options::load("config");

    int nSteps = 1000;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--headless")
            mode = HEADLESS;
        else if (arg == "--steps" && i + 1 < argc)
            nSteps = std::atoi(argv[++i]);
    }

    // SYCL INIT
    sycl::queue q{sycl::default_selector_v};
    std::cout << "Device: "
              << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    float* f =
        sycl::malloc_shared<float>(options::Nx * options::Ny * lbm::Q, q);
    float* f_new =
        sycl::malloc_shared<float>(options::Nx * options::Ny * lbm::Q, q);
    float* rho = sycl::malloc_shared<float>(options::Nx * options::Ny, q);
    float* ux = sycl::malloc_shared<float>(options::Nx * options::Ny, q);
    float* uy = sycl::malloc_shared<float>(options::Nx * options::Ny, q);

    // init dei nodi (rho=1, u=0, f=equilibrio)
    q.submit([&](sycl::handler& h) {
         h.parallel_for(sycl::range<1>(options::Nx * options::Ny),
                        [=](sycl::id<1> idx) {
                            int i = idx[0];
                            rho[i] = 1.0f;
                            ux[i] = 0.0f;
                            uy[i] = 0.0f;
                            for (int k = 0; k < lbm::Q; ++k) {
                                f[i * lbm::Q + k] = lbm::w[k];
                            }
                        });
     }).wait();

    // SHAPE INIT
    std::vector<uint8_t> solid(options::Nx * options::Ny, 0);
    Circle c(20);
    Shape* shape = &c;
    shape->createMask(solid);

    uint8_t* solidPtr =
        sycl::malloc_shared<uint8_t>(options::Nx * options::Ny, q);
    std::copy(solid.begin(), solid.end(), solidPtr);

    // RUN
    if (mode == HEADLESS) {
        runHeadless(q, f, f_new, rho, ux, uy, solidPtr, options::Nx,
                    options::Ny, options::tau, options::U0, nSteps);
    }
    else {
        loop(q, f, f_new, rho, ux, uy, solidPtr, shape, options::Nx,
             options::Ny, options::tau, options::U0);
    }

    // CLEANUP
    sycl::free(f, q);
    sycl::free(f_new, q);
    sycl::free(rho, q);
    sycl::free(ux, q);
    sycl::free(uy, q);
    sycl::free(solidPtr, q);

    return 0;
}