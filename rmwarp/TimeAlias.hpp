#pragma once

#include "Math.hpp"
#include "Simd.hpp"
#include "Allocators.hpp"
#include "VectorOps.hpp"

namespace RMWarp {
template<typename T, typename S>
void shiftAndFold(T *target, int targetSize,
                        S *src,
                    int srcSize)
{
    if(targetSize <= srcSize) {
        auto hws = srcSize / 2;
        auto smid = src + hws;
        v_copy(target, smid, hws);
        v_zero(target + hws, targetSize - srcSize);
        v_copy(target + targetSize - hws, src, hws);
        return;
    }else{
        auto hts = targetSize / 2;
        auto win_off = 0;
        auto tmid = target + hts;
        v_zero(target, hts);
        v_copy(tmid, src, hts);
        win_off = hts;
        while(win_off + targetSize < srcSize) {
            v_add(target, src + win_off, targetSize);
            win_off += targetSize;
        }
        v_add(target, src + win_off, srcSize - win_off);
    }
}
template<typename T, typename S, typename W>
void cutShiftAndFold(T *target, int targetSize,
                        S *src,
                        const W &window)
{
    if(targetSize <= window.size()) {
        auto hws = window.size() / 2;
        auto wmid = window.data() + hws;
        auto smid = src + hws;
        v_multiply(target, wmid, smid, hws);
        v_zero(target + hws, targetSize - window.size());
        v_multiply(target + targetSize - hws, src, window.data(), hws);
        return;
    }
    auto hts = targetSize / 2;
    auto win = window.data();
    auto win_off = 0;
    auto tmid = target + hts;
    v_multiply(tmid, src, window.data(), hts);
    win_off = hts;
    while(win_off + targetSize < window.size()) {
        v_multiply_and_add(target, src + win_off, win + win_off, targetSize);
        win_off += targetSize;
    }
    v_multiply_and_add(target, src + win_off, win + win_off, window.size()- win_off);
}

template<typename T, typename S, typename W>
void cutShift(T *target,S *src,const W &window)
{
    auto targetSize = window.size();
    auto hws = targetSize / 2;
    auto wbeg = window.data();
    auto wmid = wbeg + hws;
    auto wend = wbeg + targetSize;
    auto smid = src + hws;
    auto tmid = target + (targetSize-hws);
    bs::transform(wmid,wend,smid,target,bs::multiplies);
    bs::transform(wbeg,wmid,src, tmid, bs::multiplies  );
}
template<typename T, typename S, typename W>
void cutShift(T *target,S *src,S *send, const W &window)
{
    auto targetSize = window.size();
    auto srcSize = send - src;
    auto hws = targetSize / 2;
    auto wbeg = window.data();
    auto wmid = wbeg + hws;
    auto wend = wbeg + targetSize;
    auto smid = src + hws;
    auto tmid = target + (targetSize-hws);
    auto tend = target + targetSize;
    if(srcSize > targetSize)
        send = src + targetSize;

    if(srcSize < hws) {
        bs::fill(target,tmid, 0);
        bs::fill(
            bs::transform(
                src
              , send
              , wbeg
              , tmid
              , bs::multiplies)
          , tend, 0);
    }else if( srcSize < targetSize) {
        bs::fill(
            bs::transform(
                smid
              , send
              , wmid
              , target
              , bs::multiplies
                )
          , tend
          , 0);
        bs::transform(src,smid,wbeg, tmid, bs::multiplies  );
    }else{
        bs::transform(smid,send,wmid,target,bs::multiplies);
        bs::transform(src,smid,wbeg, tmid, bs::multiplies  );
    }
}
}
