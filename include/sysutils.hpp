/* -*- c-basic-offset: 4 indent-tabs-mode: nil -*-  vi:set ts=8 sts=4 sw=4: */

/*
    Rubber Band Library
    An audio time-stretching and pitch-shifting library.
    Copyright 2007-2015 Particular Programs Ltd.

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation; either version 2 of the
    License, or (at your option) any later version.  See the file
    COPYING included with this distribution for more information.

    Alternatively, if you have a valid commercial licence for the
    Rubber Band Library obtained by agreement with the copyright
    holders, you may redistribute and/or modify it under the terms
    described in that licence.

    If you wish to distribute code using the Rubber Band Library
    under terms other than those of the GNU General Public License,
    you must obtain a valid commercial licence before doing so.
*/

#ifndef _RUBBERBAND_SYSUTILS_H_
#define _RUBBERBAND_SYSUTILS_H_

#include <alloca.h>
#include <cstdint>
#include <cstddef>
#include <numeric>
#include <algorithm>
#include <iterator>
#include <functional>
#include <initializer_list>
#include <array>
#include <utility>
#include <limits>
#include <type_traits>
#include <tuple>
#include <cmath>
#include <vector>
#include <deque>
#include <complex>
#include <set>
#include <unordered_set>
#include <map>
#include <unordered_map>
#include "Simd.hpp"
#include "Math.hpp"
namespace RMWarp {

template<class T>
constexpr T princarg(T a) { return std::remainder(a, bs::Twopi<T>());}

template<class F, class... T>
constexpr void cexpr_for_each(F && f, T &&... t)
{
    using absorb = int[];
    void(absorb { (std::forward<F>(f)(std::forward<T>(t)),void(),0)...});
}
template<class F, class Tup, size_t... I>
constexpr void tuple_for_each_helper(F && f, Tup && tup, std::index_sequence<I...>)
{
    cexpr_for_each(std::forward<F>(f),std::get<I>(std::forward<Tup>(tup))...);
}
template<class F, class Tup>
constexpr void tuple_for_each(F && f, Tup && tup)
{
    tuple_for_each_helper(std::forward<F>(f), std::forward<Tup>(tup),std::make_index_sequence<std::tuple_size<Tup>::value>{});
}

template<class F, class... T>
constexpr decltype(auto) cexpr_fmap(F && f, T &&... t)
{
    using std::forward;
    using result_type = std::tuple<decltype(forward<F>(f)(forward<T>(t)))...>;
    return result_type{std::forward<F>(f)(std::forward<T>(t))...};
}
template<class F, class Tup, size_t... I>
constexpr decltype(auto) tuple_fmap_helper(F && f, Tup && tup, std::index_sequence<I...>)
{
    return cexpr_fmap(std::forward<F>(f),std::get<I>(std::forward<Tup>(tup))...);
}
template<class F, class Tup, size_t... I>
constexpr decltype(auto) tuple_fmap(F && f, Tup && tup)
{
    return tuple_fmap_helper(std::forward<F>(f),std::forward<Tup>(tup),
        std::make_index_sequence<std::tuple_size<Tup>::value>{});
}
} // end namespace
#endif
