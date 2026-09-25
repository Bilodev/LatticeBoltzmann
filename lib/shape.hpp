#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>

class Shape {
   public:
    virtual void render(sf::RenderWindow&) = 0;
    virtual void createMask(std::vector<uint8_t>& solid) = 0;
};

class Circle : public Shape {
   private:
    double radius;
    int gridCenterX, gridCenterY;
    sf::CircleShape circle;

   public:
    Circle(double);
    double getRadius();
    void render(sf::RenderWindow& w) override;
    void createMask(std::vector<uint8_t>& solid) override;
};
