CXX = acpp
CXXFLAGS = -O3 -I. -Ilib --acpp-targets=omp	
LDFLAGS = -lsfml-graphics -lsfml-window -lsfml-system

SRCS =  main.cpp src/shape.cpp
TARGET = dist/main

all:
	mkdir -p dist
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

clean:
	rm -rf dist