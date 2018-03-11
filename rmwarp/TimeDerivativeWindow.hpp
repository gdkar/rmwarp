#pragma once

#include <fftw3.h>
#include "Math.hpp"
#include "Simd.hpp"
#include "Plan.hpp"
#include "TimeAlias.hpp"
#include "Allocators.hpp"

namespace RMWarp {
template<class InputIt, class OutputIt>
OutputIt time_derivative_window(InputIt wbeg, InputIt wend, OutputIt dst, float Fs = 1.f)
{
    const auto win_size = std::distance(wbeg,wend);
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
    const auto padded_size = align_up(win_size * sizeof(float), 64ul) / sizeof(float);
    auto _data = simd_vec<float>{};//wbegin,wend);
    _data.resize(padded_size * 4);
    const auto time_r = &_data[0];
    const auto time_i = time_r + padded_size;
    const auto freq_r = time_i + padded_size;
    const auto freq_i = freq_r + padded_size;
    {
        auto plan = FFTPlan::dft_1d_c2c(win_size, time_r,time_i,freq_r,freq_i);
        std::copy_n(wbeg,win_size,time_r);
        std::fill_n(time_i,win_size,0.f );
        plan.execute();
        {
            const auto Mw         = (win_size+1)/2;

            const auto norm_mul = Fs * bs::Twopi<float>() / win_size;
            auto i = decltype(win_size){};
            for(; i < Mw; ++i) {
                auto m = -i * norm_mul;
//                auto idx = (i) % n;
                time_r[i] = freq_i[i]*m;
                time_i[i] = freq_r[i]*m;
            }
            for(; i < win_size; ++i) {
                auto m = -(i - win_size) * norm_mul;
                time_r[i] = freq_i[i] * m;
                time_i[i] = freq_r[i] * m;
            }
        }
        plan.execute();
        std::transform(freq_r,freq_r + win_size, freq_r, [ws=(1.f/win_size)](auto x) {return x * ws;});
    }
    return std::copy_n(freq_r,win_size,dst);
}
}
