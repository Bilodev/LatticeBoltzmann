#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <algorithm>

#ifndef LBM
#include "lib/core/lbm.cpp"
#endif

#include "shape.hpp"
#include "utils/GUIUtils.hpp"

void loop(sycl::queue& q, float* f, float* f_new, float* rho, float* ux,
          float* uy, const uint8_t* solidPtr, Shape* shape, int Nx, int Ny,
          float tau, float U0)
{
    // SFML INIT
    sf::RenderWindow renderWindow(
        sf::VideoMode(options::width, options::height), "Simulation");
    sf::Event event;

    // FLUID INIT (solo roba di rendering, la fisica è già inizializzata fuori)
    sf::Texture fluidTexture;
    fluidTexture.create(Nx, Ny);
    sf::Sprite fluidSprite(fluidTexture);
    fluidSprite.setScale(float(options::width) / Nx,
                         float(options::height) / Ny);
    std::vector<sf::Uint8> pixels(Nx * Ny * 4);

    //  RAMP CONFIG
    constexpr int rampSteps = 100;  // durata della rampa di velocità inlet

    long step = 0;

    // LOOP
    while (renderWindow.isOpen()) {
        while (renderWindow.pollEvent(event)) {
            if (event.type == sf::Event::EventType::Closed) {
                renderWindow.close();
            }
        }

        // rampa: U0 cresce gradualmente nei primi rampSteps, poi resta fisso
        float U_current = std::min(U0, U0 * float(step) / float(rampSteps));

        lbm::computeMacroscopics(q, f, rho, ux, uy, solidPtr, Nx, Ny);
        lbm::collision(q, f, rho, ux, uy, solidPtr, tau, Nx, Ny);
        lbm::streaming(q, f, f_new, solidPtr, Nx, Ny);
        lbm::applyBoundaries(q, f_new, U_current, Nx, Ny);
        std::swap(f, f_new);

        GUI::updateTexture(ux, uy, pixels, Nx, Ny);
        fluidTexture.update(pixels.data());

        if (step < rampSteps) step++;

        renderWindow.clear();
        renderWindow.draw(fluidSprite);
        shape->render(renderWindow);
        renderWindow.display();
    }
}