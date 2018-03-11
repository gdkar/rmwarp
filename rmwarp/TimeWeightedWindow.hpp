#pragma once

#include "sysutils.hpp"

#include <string>
#include <set>

namespace RMWarp {

template<class InputIt, class OutputIt>
OutputIt time_weighted_window(InputIt wbegin, InputIt wend, OutputIt dst, float Fs = 1.f)
{
    auto win_size = int(std::distance(wbegin,wend));
    const auto Fs_inv = 1 / Fs;
    auto Wm = (win_size+1)/2;
    auto c = (win_size % 2) ? 0.0f : 0.5f;
    auto i = 0;
    for(;i < Wm; ++i) {
        *dst++ = *wbegin++ * (i + c) * Fs_inv;
    }
    c -= win_size;
    for(;i < win_size; ++i) {
        *dst++ = *wbegin++ * (i + c) * Fs_inv;
    }
    return dst;
}
}
