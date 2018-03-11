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

//template<class T>
//constexpr decltype(auto) princarg(T a) { return bs::pedantic_(bs::rem)(a,bs::Twopi<T>());}

/*
template<>
constexpr int popcount(uint32_t t) { return __builtin_popcount(t); }
template<>
constexpr int popcount(uint64_t t) { return __builtin_popcountl(t); }
template<>
constexpr int popcount(uint16_t t) { return popcount(uint32_t(t)); }
template<>
constexpr int popcount(uint8_t t)  { return popcount(uint32_t(t)); }
template<>
constexpr int popcount(int32_t t)  { return __builtin_popcount(t); }
template<>
constexpr int popcount(int64_t t)  { return __builtin_popcountl(t); }
template<>
constexpr int popcount(int16_t t)  { return popcount(uint32_t(t)); }
template<>
constexpr int popcount(int8_t t)   { return popcount(uint32_t(t)); }
*/
template<typename T>
constexpr std::enable_if_t<std::is_integral<T>::value,T>
roundup ( T x)
{
    using U = typename std::make_unsigned<T>::type;
    auto u = U{x} - U{1};
    for(auto j = 1; j < std::numeric_limits<U>::digits(); j <<= 1)
        u |= (u >> j);
    return T{u + U{1}};
}

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
template<class T, class R>
constexpr std::enable_if_t<std::is_floating_point<R>::value,T> lerp(T lo, T hi, R _frac, R _range)
{
    return lo + (hi - lo) * (_frac * (R{1}/_range));
}
template<class T, class R>
constexpr std::enable_if_t<!std::is_floating_point<R>::value,T> lerp(T lo, T hi, R _frac, R _range)
{
    return lo + (((hi - lo) * _frac)/_range);
}
template<class T, class R, R _range = R{1}>
constexpr std::enable_if_t<std::is_floating_point<R>::value, T> lerp(T lo, T hi, R _frac)
{
    constexpr auto _factor = R{1}/_range;
    return lo + (hi - lo) * (_frac*_factor);
}
}
