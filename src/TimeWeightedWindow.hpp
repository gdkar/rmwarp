#pragma once

#include "FFT.hpp"

namespace RMWarp {

template<class InputIt, class OutputIt>
OutputIt time_weighted_window(InputIt wbegin, InputIt wend, OutputIt dst)
{
    auto n = int(std::distance(wbegin,wend));
    auto c = -0.5f * (n - 1);
    for(; wbegin != wend; (c += 1.0f))
        *dst++ = *wbegin++ * c;
    return dst;
}
template<class T>
T * time_weighted_window( const T* wbegin, const T* wend, T* dst)
{
    using reg = simd_reg<T>;
    constexpr auto w = int(simd_width<T>);
    auto n = wend - wbegin;
    auto i = 0;
    auto c = bs::enumerate<reg>(-T(0.5) * (n-1));
    for(; i + w < n; i += w ) {
        bs::aligned_store(reg(wbegin + i) * c, dst + i);
        c += reg(1);
    }
    auto sc = c[0];
    for(; i < n; ++i, sc += T{1})
        dst[i] = wbegin[i] * c;
}
}
