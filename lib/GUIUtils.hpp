#include <iostream>
//
#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include "options.hpp"

class GUI {
    static void densityToColor(float t, sf::Uint8& r, sf::Uint8& g,
                               sf::Uint8& b)
    {
        t = std::clamp(t, 0.0f, 1.0f);

        if (t < 0.5f) {
            // Blu -> Verde
            float local = t / 0.5f;  // rimappa [0,0.5] -> [0,1]
            r = 0;
            g = static_cast<sf::Uint8>(255 * local);
            b = static_cast<sf::Uint8>(255 * (1.0f - local));
        }
        else {
            // Verde -> Rosso
            float local = (t - 0.5f) / 0.5f;  // rimappa [0.5,1] -> [0,1]
            r = static_cast<sf::Uint8>(255 * local);
            g = static_cast<sf::Uint8>(255 * (1.0f - local));
            b = 0;
        }
    }

   public:
    static void updateTexture(const float* ux, const float* uy,
                       std::vector<sf::Uint8>& pixels, int Nx, int Ny)
    {
        for (int i = 0; i < Nx * Ny; ++i) {
            float speed = std::sqrt(ux[i] * ux[i] + uy[i] * uy[i]);
            float t = std::clamp(speed / options::U0, 0.0f,
                                 1.0f);  // normalizza rispetto a U0

            sf::Uint8 r, g, b;
            densityToColor(t, r, g,
                           b);  // stessa funzione blu->verde->rosso di prima
            pixels[i * 4 + 0] = r;
            pixels[i * 4 + 1] = g;
            pixels[i * 4 + 2] = b;
            pixels[i * 4 + 3] = 255;
        }
    }
};