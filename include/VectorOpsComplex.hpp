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

#ifndef _RUBBERBAND_VECTOR_OPS_COMPLEX_H_
#define _RUBBERBAND_VECTOR_OPS_COMPLEX_H_

#include "VectorOps.hpp"


namespace RMWarp {

template<typename T> // S source, T target
void v_polar_to_cartesian(T *const  real,
                          T *const  imag,
                          const T *const  mag,
                          const T *const  phase,
                          const int count)
{
    constexpr auto w = int(simd_width<T>);
    using reg = simd_reg<T>;
    auto i = 0;
    for ( ; i + w < count; i += w ) {
        auto _i_r = bs::sincos(reg(phase + i));
        auto _mag = reg(mag + i);
        bs::store(_i_r.first * _mag, imag + i);
        bs::store(_i_r.second* _mag, real + i);
    }
    for (; i < count; ++i) {
        auto _i_r = bs::sincos(*(phase+i));
        auto _mag = mag[i];
        imag[i] = _i_r.first * _mag;
        real[i] = _i_r.second* _mag;
    }
}
template< typename T> // S source, T target
void v_cartesian_to_magnitude(T *const  mag,
                          const T *const  real,
                          const T *const  imag,
                          int count)
{
    bs::transform(real,real+count,imag,mag,bs::hypot);
}


template< typename T> // S source, T target
void v_cartesian_to_polar(T *const  mag,
                          T *const  phase,
                          const T *const  real,
                          const T *const  imag,
                          const int count)
{
    constexpr auto w = int(simd_width<T>);
    using reg = simd_reg<T>;
    auto i = 0;
    for ( ; i + w < count; i += w ) {
        auto _r = reg(real + i);
        auto _i = reg(imag + i);
        bs::store(bs::hypot(_i,_r), mag + i);
        bs::store(bs::atan2(_i,_r), phase+i);
    }
    for (; i < count; ++i) {
        auto _r = *(real + i);
        auto _i = *(imag + i);
        mag[i]   = bs::hypot(_i,_r);
        phase[i] = bs::atan2(_i,_r);
    }
}
}
#endif
