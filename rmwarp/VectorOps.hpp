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

#ifndef _RUBBERBAND_VECTOR_OPS_H_
#define _RUBBERBAND_VECTOR_OPS_H_

#include <cstring>
#include "Simd.hpp"
#include "Allocators.hpp"
#include "Math.hpp"
#include "sysutils.hpp"

namespace RMWarp {

// Note that all functions with a "target" vector have their arguments
// in the same order as memcpy and friends, i.e. target vector first.
// This is the reverse order from the IPP functions.

// The ideal here is to write the basic loops in such a way as to be
// auto-vectorizable by a sensible compiler (definitely gcc-4.3 on
// Linux, ideally also gcc-4.0 on OS/X).

template<typename T>
void v_zero(T *const ptr,int count)
{
    std::fill_n(ptr,count, T{});
}
template<typename T>
void v_set(T *const ptr,const T value,int count)
{
    std::fill_n(ptr,count,value);
}
template<typename T>
void v_copy(T *const dst,
            const T *const src,
            int count)
{
    std::copy_n(src,count,dst);
}
// src and dst alias by definition, so not restricted
template<typename T>
void v_move(T *const dst,
                   const T *const src,
                   int count)
{
    ::memmove(dst, src, count * sizeof(T));
}
template<typename T>
void v_shift(T *const dst, int count , int by)
{
    auto beg = dst + by;
    auto mid = dst + (count-by);
    auto end = dst + count;
    std::move(beg,end,dst);
    std::fill(mid,end, T{});
}
template<typename T>
void v_add(T *const dst,
                  const T *const src,
                  int count)
{
    bs::transform(src,src+count,dst,dst,bs::plus);
}
template<typename T>
void v_add(T *const dst,
                  const T value,
                  int count)
{
//    auto v = simd_reg<T>(value);
    bs::transform(dst,dst+count,dst, [v=value](auto x){return x + v;});
}
template<typename T, typename G>
void v_add_with_gain(T *const dst,
                            const T *const src,
                            G gain,
                            int count)
{
    bs::transform(src,src+count,dst,dst,[gain](auto x, auto y) { return y + x* gain; });
}
template<typename T>
void v_subtract(T *const dst,const T *const src,int count)
{
    bs::transform(dst,dst+count,src,dst,bs::minus);
}
template<typename T, typename G>
void v_scale(
    T *const dst,
    T const *const src,
    const G gain,
    int count)
{
    bs::transform(src,src+count,dst,[gain](auto x){ return x* gain;});
}

template<typename T, typename G>
void v_scale(T *const dst,
                    const G gain,
                    int count)
{
    bs::transform(dst,dst+count,dst,[gain](auto x){ return x* gain;});
}
template<typename T>
void v_multiply(T *const dst,
                       const T *const src,
                       int count)
{
    bs::transform(src,src+count,dst,dst,bs::multiplies);
}
template<typename T>
void v_multiply(T *const dst,
                       const T *const src1,
                       const T *const src2,
                       int count)
{
    bs::transform(src1,src1+count,src2,dst,bs::multiplies);
}
template<typename T>
void v_divide(T *const dst,const T *const src,int count)
{
    bs::transform(dst,dst+count,src,dst,bs::divides);
}
template<typename T>
void v_multiply_and_add(T *const dst,
                            const T *const src,
                            T gain,
                            int count)
{
    bs::transform(src,src+count,dst,dst,[gain](auto x, auto y) { return y + x* gain; });
}
template<typename T>
void v_multiply_and_add(
    T *const dst,
    const T *const src1,
    const T *const src2,
    int count)
{
    int i = 0;
    constexpr auto w = int(simd_width<T>);
    using reg = simd_reg<T>;
    for ( ; i + w < count; i += w ) {
        auto d = reg(dst + i);
        auto s1= reg(src1+ i);
        auto s2= reg(src2+ i);
        bs::store(bs::fma(s1,s2,d),dst+i);
    }
    for ( ; i < count; ++i )
        dst[i] += src1[i] * src2[i];
}
template<typename T>
T v_sum(const T *const src, int count)
{
    return bs::reduce(src,src+count,T{});
}
template<typename T>
void v_log(T *const dst, int count)
{
    bs::transform(dst,dst+count,dst,bs::log);
}
template<typename T>
void v_exp(T *const dst, int count)
{
    bs::transform(dst,dst+count,dst,bs::exp);
}
template<typename T>
void v_sqrt(T *const dst,int count)
{
    bs::transform(dst,dst+count,dst,bs::sqrt);
}
template<typename T>
void v_square(T *const dst,int count)
{
    bs::transform(dst,dst+count,dst,bs::sqr);
}
template<typename T>
void v_abs(T *const dst, int count)
{
    bs::transform(dst,dst+count,dst,bs::abs);
}
template<typename T>
void v_interleave(T *const dst,const T *const *const src,int channels,int count)
{
    auto idx = 0;
    switch (channels) {
    case 2: {
        // common case, may be vectorized by compiler if hardcoded
        int i = 0;
        constexpr auto w = int(simd_width<T>);
        using reg = simd_reg<T>;
        for ( ; i + w < count; i += w ) {
            auto lo = reg(src[0] + i);
            auto hi = reg(src[1] + i);
//            auto in = bs::interleave(lo, hi);
            bs::aligned_store(bs::interleave_first(lo,hi), dst + i * 2);
            bs::aligned_store(bs::interleave_second(lo,hi), dst + i * 2 + w);
//            bs::aligned_store(std::get<1>(in), dst + i * 2 + w);
        }
        for (i = 0; i < count; ++i) {
            for (int j = 0; j < 2; ++j)
                dst[2 * i + j] = src[j][i];
        }
        return;
    }
    case 1:
        v_copy(dst, src[0], count);
        return;
    default:
        for (int i = 0; i < count; ++i) {
            for (int j = 0; j < channels; ++j) {
                dst[idx++] = src[j][i];
            }
        }
    }
}
template<typename T>
void v_deinterleave(T *const *const dst,
                           const T *const src,
                           int channels,
                           int count)
{
    int idx = 0;
    switch (channels) {
    case 2: {
        // common case, may be vectorized by compiler if hardcoded
        int i = 0;
        constexpr auto w = int(simd_width<T>);
        using reg = simd_reg<T>;
        for ( ; i + w < count; i += w ) {
            auto lo = reg(src + 2 * i);
            auto hi = reg(src + 2 * i + w);
            bs::aligned_store(bs::deinterleave_first(lo, hi), dst[0] + i);
            bs::aligned_store(bs::deinterleave_second(lo, hi), dst[1] + i);
//            auto de = bs::deinterleave(lo, hi);
//            bs::aligned_store(std::get<0>(de), dst[0] + i);
//            bs::aligned_store(std::get<1>(de), dst[1] + i);
        }
        for (i = 0; i < count; ++i) {
            for (int j = 0; j < 2; ++j)
                dst[j][i] = src[2 * i + j];
        }
        return;
    }
    case 1:
        v_copy(dst[0], src, count);
        return;
    default:
        for (int i = 0; i < count; ++i) {
            for (int j = 0; j < channels; ++j)
                dst[j][i] = src[idx++];
        }
    }
}
}
#endif
