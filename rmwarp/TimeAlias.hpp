#pragma once

#include "Math.hpp"
#include "Simd.hpp"
#include "Allocators.hpp"

namespace RMWarp {

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
