#pragma once

#include "sysutils.hpp"

#include <string>
#include <set>

namespace RMWarp {

template<class InputIt, class OutputIt>
OutputIt time_weighted_window(InputIt wbeg, InputIt wend, OutputIt dst, float Fs = 1.f)
{
    auto win_size = std::distance(wbeg,wend);
    auto c = -0.5f * (win_size - 1);
    auto Fs_inv = 1.f / Fs;
    for(; wbeg != wend; c++) {
        *(dst++) = *(wbeg++) * c * Fs_inv;
    }
    return dst;
}
/*template<class T>
T * time_weighted_window( const T* wbegin, const T* wend, T* dst, float Fs = 1.f)
{
    using reg = simd_reg<T>;
    constexpr auto w = int(simd_width<T>);
    auto win_size = int(std::distance(wbegin,wend));
    auto i = 0;
    auto Fs_inv = 1 / Fs;
    auto c = bs::enumerate<reg>(-T(0.5) * (win_size-1));
    for(; i + w <= win_size; i += w ) {
        bs::aligned_store(reg(wbegin + i) * (c + i) * Fs_inv, dst + i);
    }
    auto sc = c[0];
    for(; i < win_size; ++i)
        bs::aligned_store(*(wbegin + i) * (sc + i) * Fs_inv, dst +i);
    return dst;
}*/
}
