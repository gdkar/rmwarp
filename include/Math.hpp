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
#include <valarray>
#include <algorithm>
#include <memory>
#include "Simd.hpp"

namespace RMWarp {
template<class T> constexpr const T T_PI = T{M_PI};
template<class T> constexpr const T T_TWO_PI = T{2*M_PI};
template<class T> constexpr const T T_PI_2 = T{0.5*M_PI};

template<class T>
constexpr T princarg(T a) { return bs::rem(a, T_TWO_PI<T>);}

template<class T>
constexpr int ilog2(T t)
{
    return 64 - __builtin_clzl(uint64_t(t) - 1);
}
template<>
constexpr int ilog2(uint32_t t) { return 32 - __builtin_clz(t-1u); }
template<>
constexpr int ilog2(uint64_t t) { return 64 - __builtin_clzl(t-1u); }
template<>
constexpr int ilog2(uint16_t t) { return ilog2(uint32_t(t)); }
template<>
constexpr int ilog2(uint8_t t)  { return ilog2(uint32_t(t)); }
template<>
constexpr int ilog2(int32_t t)  { return 32 - __builtin_clz(t-1u); }
template<>
constexpr int ilog2(int64_t t)  { return 64 - __builtin_clzl(t-1u); }
template<>
constexpr int ilog2(int16_t t)  { return ilog2(uint32_t(t)); }
template<>
constexpr int ilog2(int8_t t)   { return ilog2(int32_t(t)); }

template<typename T>
T roundup ( T x) { return T{1} << ilog2(x); }

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
