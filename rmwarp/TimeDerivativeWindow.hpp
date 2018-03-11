#pragma once

#include <cstdio>
#include <iostream>
#include <fftw3.h>
#include "Math.hpp"
#include "Simd.hpp"
#include "Plan.hpp"
#include "Allocators.hpp"
#include "TimeAlias.hpp"

namespace RMWarp {
template<class InputIt, class OutputIt>
OutputIt time_derivative_window(InputIt wbeg, InputIt wend, OutputIt dst, float Fs = 1.f)
{
    auto win_size = std::distance(wbeg,wend);
    std::cerr << "win_size = " << win_size << std::endl;
    if(!win_size)
        return dst;
    /*  fft(ifft(x) = n * x, under fftw3 rules, so scale is 1 / n
     *  framp should be the supplied range ( n peak to peak ) * ( Fs / n )
     *  in this case we choose to time-normalize such that
     *  Fs ( in cycles / time ) is just 1.f
     *  this yields a cummulative scale factor of ( 1 / n ) * ( 1 / n )
     *  or ( 2 / ( n * n ) )
     *
     *  the rest of this mess is doing two things at once.
     *
     *  1.  we actually want to compute -imag(ifft(framp * fft(win)))
     *      that negative is just applied to mul, and passed along
     *
     *  2.  since we want an ifft, but only have a plan for an fft,
     *      we do imag( swap (fft( swap ( -framp * fft( win ) ) ) ) )
     *
     *      the inner swap is done explicitly in the loop. the outer swap is
     *      done implicitly by taking
     *
     *      real( fft( swap( -framp * fft( win ) ) ) )
     *
     *      which is basically verbatim what we compute below.
     */
    auto _buff= simd_vec<float>();
    _buff.resize(win_size * 4);

    const auto _time_r = &_buff[0];
    const auto _time_i = _time_r + win_size;
    const auto _freq_r = _time_i + win_size;
    const auto _freq_i = _freq_r + win_size;

    {
        auto plan = FFTPlan::dft_1d_c2c(win_size, _time_r,_time_i,_freq_r,_freq_i);
        std::copy(wbeg,wend,_time_r);
        std::fill_n(_time_i,win_size,0.f);
        plan.execute(_time_r,_freq_r);
        {
            auto Wm = (win_size+1)/2u;
            auto c  = (win_size%2)?0.0f : 0.5f;
            auto norm_mul = -Fs / (float(win_size));
            std::cerr << "norm_mul = " << norm_mul << std::endl;
            auto i  = decltype(win_size){0};
            for(; i < Wm; ++i) {
                auto mul = float(c + i) * norm_mul;
                _time_r[i] = _freq_i[i] * mul;
                _time_i[i] = _freq_r[i] * mul;
            }
            c -= win_size;
            for(; i < win_size; ++i) {
                auto mul = float(c + i) * norm_mul;
                _time_r[i] = _freq_i[i] * mul;
                _time_i[i] = _freq_r[i] * mul;
            }
        }
        plan.execute(_time_r,_freq_r);
    }
    return std::copy_n(_freq_i,win_size,dst);
}
}
