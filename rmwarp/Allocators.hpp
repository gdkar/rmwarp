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

#pragma once

#include <numeric>
#include <algorithm>
#include <initializer_list>
#include <tuple>
#include <array>
#include <vector>
#include <utility>
#include <numeric>
#include <functional>
#include <memory>
#include <iterator>
#include <cstdlib>
#include <alloca.h>
#include <type_traits>
#include <limits>
#include <new> // for std::bad_alloc

#include <boost/simd/pack.hpp>
#include <boost/simd/memory/allocator.hpp>
#include <boost/align/aligned_allocator.hpp>
#include <boost/align/aligned_delete.hpp>

#include <boost/simd/constant/twopi.hpp>
#include <boost/simd/decorator.hpp>
#include <boost/simd/range.hpp>
#include <boost/simd/bitwise.hpp>
#include <boost/simd/ieee.hpp>
#include <boost/simd/memory.hpp>
#include <boost/simd/algorithm.hpp>
#include <boost/simd/boolean.hpp>
#include <boost/simd/eulerian.hpp>
#include <boost/simd/operator.hpp>
#include <boost/simd/reduction.hpp>
#include <boost/simd/forward.hpp>
#include <boost/simd/logical.hpp>
#include <boost/simd/trigonometric.hpp>
#include <boost/simd/arithmetic.hpp>
#include <boost/simd/mask.hpp>
#include <boost/simd/predicates.hpp>
#include <boost/simd/hyperbolic.hpp>
#include <boost/simd/as.hpp>

namespace RMWarp {

namespace bs = boost::simd;
namespace ba = boost::alignment;

using pts_type = std::int64_t;
using sample_type = float;
using time_type   = double;

template<class T>
using aligned_ptr = std::unique_ptr<T,bs::aligned_delete>;

template<class T = sample_type>
using simd_reg = bs::pack<T>;

template<class T = sample_type>
using simd_vec = std::vector<T , bs::allocator<T> >;

template<class T = sample_type>
constexpr auto simd_alignment = simd_reg<T>::alignment;

constexpr auto default_alignment = simd_alignment<sample_type>;

namespace detail {
template<class T>
struct up_if_object { using type = aligned_ptr<T>;};
template<class T>
struct up_if_object<T[]> { };
template<class T, std::size_t N>
struct up_if_object<T[N]> { };
template<class T>
using up_if_object_t = typename up_if_object<T>::type;

template<class T>
struct up_if_array { };
template<class T>
struct up_if_array<T[]> { using type = aligned_ptr<T[]>; };

template<class T>
using up_if_array_t = typename up_if_array<T>::type;

template<class T>
struct up_element { };
template<class T>
struct up_element<T[]>  { using type = T; };
template<class T>
using up_element_t = typename up_element<T>::type;

template<class T>
struct simd_new_align : std::integral_constant<
    std::size_t
  , std::max<std::size_t>({
        default_alignment
      , simd_alignment<T>
      , ba::alignment_of<T>::value
        })
    > { };

template<class T>
constexpr auto simd_new_align_v = simd_new_align<T>::value;
};

template<class T>
detail::up_if_object_t<T> make_aligned()
{
    auto p = ba::aligned_alloc(detail::simd_new_align_v<T>, sizeof(T));
    if (!p) {
        throw std::bad_alloc();
    }
    try {
        auto q = ::new(p) T();
        return detail::up_if_object_t<T>(q);
    } catch (...) {
        ba::aligned_free(p);
        throw;
    }
}
template<class T, class... Args>
detail::up_if_object_t<T> make_aligned(Args&&... args)
{
    auto p = ba::aligned_alloc(detail::simd_new_align_v<T>, sizeof(T));
    if (!p) {
        throw std::bad_alloc();
    }
    try {
        auto q = ::new(p) T(std::forward<Args>(args)...);
        return detail::up_if_object_t<T>(q);
    } catch (...) {
        ba::aligned_free(p);
        throw;
    }
}
template<class T, class... Args>
detail::up_if_object_t<T> make_aligned_noinit()
{
    auto p = ba::aligned_alloc(detail::simd_new_align_v<T>, sizeof(T));
    if (!p) {
        throw std::bad_alloc();
    }
    try {
        auto q = ::new(p) T;
        return detail::up_if_object_t<T>(q);
    } catch (...) {
        ba::aligned_free(p);
        throw;
    }
}
template<class T>
detail::up_if_array_t<T> make_aligned(std::size_t n)
{
    using E = detail::up_element_t<T>;
    if(auto p = ba::aligned_alloc(detail::simd_new_align_v<E>, sizeof(E) * n)) {
        try {
            auto q = ::new(p) E[n]();
            return detail::up_if_array_t<T>(q);
        } catch (...) {
            ba::aligned_free(p);
            throw;
        }
    }else{
        throw std::bad_alloc();
    }
}
template<class T>
typename detail::up_if_array<T>::type make_aligned_noinit(std::size_t n)
{
    using E = detail::up_element_t<T>;
    if(auto p = ba::aligned_alloc(detail::simd_new_align_v<E>, sizeof(E) * n)) {
        try {
            auto q = ::new(p) E[n]; return detail::up_if_array_t<T>(q);
        } catch (...) {
            ba::aligned_free(p);
            throw;
        }
    }else{
        throw std::bad_alloc();
    }
}
}
