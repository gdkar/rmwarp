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

template<typename T>
constexpr T roundup ( T x )
{
  x--;x|=x>>1;x|=x>>2;x|=x>>4;
  x|=(x>>((sizeof(x)>1)?8:0));
  x|=(x>>((sizeof(x)>2)?16:0));
  x|=(x>>((sizeof(x)>4)?32:0));
  return x+1;
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
