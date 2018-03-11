#pragma once

#include <fftw3.h>
#include "rmwarp/Math.hpp"
#include "rmwarp/Simd.hpp"
#include "rmwarp/Plan.hpp"
#include "rmwarp/Allocators.hpp"

namespace RMWarp {
template<class InputIt, class OutputIt>
OutputIt time_derivative_window(InputIt wbegin, InputIt wend, OutputIt dst, float Fs = 1.f)
{
    auto win_size = int(std::distance(wbegin,wend));
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
    auto time_r = simd_vec<float>(wbegin,wend);
    auto time_i = simd_vec<float>(win_size,0.f);
    {
        auto plan = FFTPlan::dft_1d_c2c(win_size, &time_r[0],&time_i[0],&time_r[0],&time_i[0]);
        plan.execute();
        {
            auto Mw         = (win_size-1)/2;
            auto first_half = (win_size+1)/2;
            auto c          = (win_size % 2) ? 0.f : 0.5f;

            auto norm_mul = Fs / (win_size*win_size);
            using std::swap;
            auto i = 0;
            for(; i < first_half; ++i) {
                auto mul = - c * norm_mul;
//                auto idx = (i) % n;
                time_r[i] *= mul;
                time_i[i] *= mul;
                swap(time_r[i],time_i[i]);
                c += 1.f;
            }
            c -= win_size;
            for(; i < win_size; ++i) {
                auto mul = - c * norm_mul;
                time_r[i] *= mul;
                time_i[i] *= mul;
                swap(time_r[i],time_i[i]);
                c += 1.f;
            }
        }
        plan.execute();
    }
    return std::copy(time_r.cbegin(),time_r.cend(),dst);
}
}
