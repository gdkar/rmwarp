#pragma once

#include <type_traits>
#include <initializer_list>
#include <complex>
#include <utility>
#include <functional>
#include <numeric>
#include <iterator>
#include <ctgmath>
#include <cmath>
#include <algorithm>
#include <memory>
#include "Simd.hpp"

namespace RMWarp {
template<class T> constexpr const T T_PI = T{M_PI};
template<class T> constexpr const T T_TWO_PI = T{2*M_PI};
template<class T> constexpr const T T_PI_2 = T{0.5*M_PI};

template<class T>
constexpr T align_up(T x, T a) {
    using U = std::make_unsigned_t<T>;
    auto m = U(x) % U(a);
    return m ? T(x + a - m) : x;
}
template<class T>
constexpr T align_down(T x, T a) {
    using U = std::make_unsigned_t<T>;
    return T(U(x) - (U(x) % U(a)));
}
template<class T, class Compare>
constexpr const T& clamp( const T& v, const T& lo, const T& hi, Compare && comp)
{
    return comp(v, lo) ? lo : comp(hi, v) ? hi : v;
}
template<class T>
constexpr const T &clamp( const T& v, const T& lo, const T& hi)
{
    return clamp( v, lo, hi, std::less<>{});
}
}
