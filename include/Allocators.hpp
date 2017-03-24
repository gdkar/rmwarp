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
        auto q = ::new(p) T(); return detail::up_if_object_t<T>(q);
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
            auto q = ::new(p) E[n](); return detail::up_if_array_t<T>(q);
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
template <typename T>
T *allocate(size_t count)
{
    return static_cast<T*>(ba::aligned_alloc(detail::simd_new_align_v<T>, sizeof(T) * count));
}
template <typename T>
T *allocate_and_zero(size_t count)
{
    auto it = allocate<T>(count);
    std::fill_n(it, count, T{});
    return it;
}
template <typename T>
void deallocate(T *ptr)
{
    ba::aligned_free(ptr);
}
/// Reallocate preserving contents but leaving additional memory uninitialised
template <typename T>
T *reallocate(T *ptr, size_t oldcount, size_t count)
{
    if(!count) {
        deallocate(ptr);
        return nullptr;
    } else if(!oldcount) {
        return allocate<T>(count);
    }else{
        auto it = allocate<T>(count);
        std::copy_n(ptr, std::min(count,oldcount), it);
        deallocate(ptr);
        return it;
    }
}
/// Reallocate, zeroing all contents
template <typename T>
T *reallocate_and_zero(T *ptr, size_t oldcount, size_t count)
{
    if(count == oldcount)
        return ptr;
    deallocate(ptr);
    return count ? allocate_and_zero<T>(count) : nullptr;
}
template<class T>
void reallocate_and_zero(aligned_ptr<T[]> & ptr,size_t oldcount, size_t newcount)
{
    if(oldcount != newcount) {
        ptr = make_aligned<T[]>(newcount);
    }
    std::fill_n(ptr.get(), newcount, T{});
}
template<class T>
void reallocate_and_zero_extension(aligned_ptr<T[]> & ptr,size_t oldcount, size_t newcount)
{
    if(oldcount != newcount) {
        auto tmp = make_aligned<T[]>(newcount);
        if(auto copy_count = std::min(oldcount,newcount)) {
            std::fill(std::move(ptr.get(), ptr.get() + oldcount, tmp.get()),tmp.get() + newcount, T{});
        }else if(newcount) {
            std::fill_n(ptr.get(), newcount, T{});
        }
        ptr.swap(tmp);
    }
}
/// Reallocate preserving contents and zeroing any additional memory

template <typename T>
T *reallocate_and_zero_extension(T *ptr, size_t oldcount, size_t count)
{
    if(count == oldcount)
        return ptr;
    auto it = ptr;
    if(count) {
        it = allocate<T>(count);
        auto mid = oldcount ? std::copy_n(ptr, std::min(oldcount,count), it) : it;
        std::fill(mid, it + count, T{0});
    }
    if(ptr)
        deallocate(ptr);
    return it;
}
}
