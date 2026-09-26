#pragma once

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace options {

inline unsigned int width = 800;
inline unsigned int height = 400;

inline int Nx = 800;
inline int Ny = 400;
inline float tau = 0.6f;  // -> ν ≈ 0.0333
inline float U0 = 0.04f;

// Legge un file di testo con righe tipo "chiave = valore" (o "chiave valore")
// e sovrascrive i default sopra. Righe vuote o che iniziano con '#' vengono
// ignorate.
inline void load(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("options::load: impossibile aprire il file " +
                                 path);
    }

    std::unordered_map<std::string, std::string> values;
    std::string line;

    while (std::getline(file, line)) {
        // rimuove eventuali commenti dopo '#'
        auto hash_pos = line.find('#');
        if (hash_pos != std::string::npos) line = line.substr(0, hash_pos);

        std::istringstream iss(line);
        std::string key, eq, value;

        if (!(iss >> key)) continue;    // riga vuota
        if (!(iss >> value)) continue;  // manca il valore

        // supporta sia "chiave = valore" che "chiave valore"
        if (value == "=") {
            if (!(iss >> value)) continue;
        }

        values[key] = value;
    }

    auto get_uint = [&](const std::string& k, unsigned int& dst) {
        if (auto it = values.find(k); it != values.end())
            dst = static_cast<unsigned int>(std::stoul(it->second));
    };
    auto get_int = [&](const std::string& k, int& dst) {
        if (auto it = values.find(k); it != values.end())
            dst = std::stoi(it->second);
    };
    auto get_float = [&](const std::string& k, float& dst) {
        if (auto it = values.find(k); it != values.end())
            dst = std::stof(it->second);
    };

    get_uint("width", width);
    get_uint("height", height);
    get_int("Nx", Nx);
    get_int("Ny", Ny);
    get_float("tau", tau);
    get_float("U0", U0);
}

}  // namespace options