#pragma once

#include "FFT.hpp"

namespace RMWarp {
template<class InputIt, class OutputIt>
OutputIt time_derivative_window(InputIt wbegin, InputIt wend, OutputIt dst)
{
    auto n = int(std::distance(wbegin,wend));
    auto time_r = simd_vec<float>(wbegin,wend), time_i = simd_vec<float>(n)
      ,  freq_r = simd_vec<float>(n), freq_i = simd_vec<float>(n);
    auto fft = FFT{n};
    fft.forward(freq_r.begin(),freq_i.begin(),time_r.begin(),time_i.begin());
    {
        auto base_idx = n/2 + 1;
        auto base_mul = - ( n - 1 ) * 0.5f;
        auto norm_mul = float(2 * M_PI) / n;
        for(auto i = 0; i < n; ++i) {
            auto idx = ( i + base_idx ) % n;
            auto mul = (i + base_mul) * norm_mul;
            freq_r[idx] *= -mul;
            freq_i[idx] *= -mul;
        }
    }
    fft.inverse(time_r.begin(),time_i.begin(),freq_r.cbegin(),freq_i.cbegin());
    return std::copy(time_i.cbegin(),time_i.cend(),dst);
}
}
