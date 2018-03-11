#pragma once

#include "Math.hpp"
#include "Simd.hpp"
#include "Allocators.hpp"

namespace RMWarp {

template<typename T, typename S, typename W>
void cutShift(const S *sbeg, T *dbeg, W wbeg, W wend)
{
    auto win_size = wend - wbeg;
    auto Wm = (win_size+1)/2;
    auto Ws = win_size - Wm;

    auto smid = sbeg + Wm;
    auto wmid = wend - Wm;
    auto dmid = dbeg + Ws;

    bs::transform(wmid,wend, sbeg,dmid,bs::multiplies);
    bs::transform(wbeg,wmid, smid,dbeg,bs::multiplies);
}
template<typename T, typename S, typename W>
void cutShift(const  S *sbeg,const S *send, T* dbeg, W wbeg, W wend)
{
    auto win_size = send - sbeg;
    auto src_size = send - sbeg;
    auto Wm = (win_size +1)/2;
    auto Ws = win_size - Wm;

    auto smid = sbeg + Wm;
    auto wmid = wend - Wm;
    auto dmid = dbeg + Ws;

    if(src_size <= Wm) {
        std::fill(dbeg, dmid, 0.f);
        bs::transform(wbeg, wbeg + src_size, sbeg, dmid, bs::multiplies);
        std::fill(dmid + src_size, dmid + win_size, 0.f);
    } else if(src_size < win_size) {
        auto Wc = src_size - Wm;
        bs::transform(wmid,wend,sbeg,dmid,bs::multiplies);
        bs::transform(wbeg,wbeg + Wc, smid, dbeg, bs::multiplies);
        std::fill(dbeg + Wc, dbeg + Wm, 0.f);
    } else {
        bs::transform(wmid,wend, sbeg,dmid,bs::multiplies);
        bs::transform(wbeg,wmid, smid,dbeg,bs::multiplies);
    }
}
template<class It, class Ot>
Ot fftshift(It ibeg, It iend, Ot obeg) {
    auto win_size = iend - ibeg;
    auto Wm = (win_size+1)/2;
    return std::rotate_copy(ibeg, ibeg + Wm, iend, obeg);
}
template<class It>
void fftshift(It ibeg, It iend) {
    auto win_size = iend - ibeg;
    auto Wm = (win_size+1)/2;
    std::rotate(ibeg, ibeg + Wm, iend);
}
template<class It, class Ot>
Ot ifftshift(It ibeg, It iend, Ot obeg) {
    auto win_size = iend - ibeg;
    auto Wm = (win_size+1)/2;
    return std::rotate_copy(ibeg, iend - Wm, iend, obeg);
}
template<class It>
void ifftshift(It ibeg, It iend) {
    auto win_size = iend - ibeg;
    auto Wm = (win_size+1)/2;
    std::rotate(ibeg, iend - Wm, iend);
}

}
