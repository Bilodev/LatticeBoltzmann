#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
//
#include "options.hpp"
#include "shape.hpp"

Circle::Circle(double r) : radius(r)
{
    // r è in NODI griglia — unità della fisica
    gridCenterX = options::Nx / 4;
    gridCenterY = options::Ny / 2;

    // fattori di conversione griglia -> pixel schermo
    float scaleX = static_cast<float>(options::width) / options::Nx;   // = 2
    float scaleY = static_cast<float>(options::height) / options::Ny;  // = 2

    // il raggio a schermo deve scalare insieme alla posizione
    float screenRadius =
        static_cast<float>(r) * scaleX;  // assumendo scaleX == scaleY

    circle = sf::CircleShape(screenRadius);
    circle.setFillColor(sf::Color::White);
    circle.setOrigin({screenRadius, screenRadius});
    circle.setPosition({gridCenterX * scaleX, gridCenterY * scaleY});
}

void Circle::createMask(std::vector<uint8_t>& solid)
{
    for (int y = 0; y < options::Ny; y++) {
        for (int x = 0; x < options::Nx; x++) {
            int dx = x - gridCenterX;
            int dy = y - gridCenterY;
            if (dx * dx + dy * dy <=
                radius * radius) {  // radius in nodi, coerente
                solid[y * options::Nx + x] = 1;
            }
        }
    }
}

double Circle::getRadius() { return radius; }

void Circle::render(sf::RenderWindow& w) { w.draw(circle); }
