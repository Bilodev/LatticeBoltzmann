// lbm_constants.hpp
namespace lbm {
inline constexpr int Q = 9;
inline constexpr int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
inline constexpr int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
inline constexpr float w[9] = {4.f / 9,  1.f / 9,  1.f / 9,  1.f / 9, 1.f / 9,
                               1.f / 36, 1.f / 36, 1.f / 36, 1.f / 36};
inline constexpr int opposite[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

inline float feq(int k, float rho, float ux, float uy)
{
    float cu = cx[k] * ux + cy[k] * uy;
    float u2 = ux * ux + uy * uy;
    return w[k] * rho * (1.f + 3.f * cu + 4.5f * cu * cu - 1.5f * u2);
}
}  // namespace lbm