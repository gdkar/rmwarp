#pragma once


#include <boost/simd/pack.hpp>
#include <boost/simd/meta/cardinal_of.hpp>

#include <boost/simd/algorithm.hpp>


#include <boost/simd/memory/allocator.hpp>
#include <boost/align/aligned_allocator.hpp>
#include <boost/align/aligned_delete.hpp>

#include <boost/simd/constant/twopi.hpp>
#include <boost/simd/decorator.hpp>
#include <boost/simd/trigonometric.hpp>
#include <boost/simd/hyperbolic.hpp>
#include <boost/simd/exponential.hpp>
#include <boost/simd/memory.hpp>
#include <boost/simd/arithmetic.hpp>
#include <boost/simd/ieee.hpp>
#include <boost/simd/range.hpp>
#include <boost/simd/reduction.hpp>
#include <boost/simd/algorithm.hpp>
#include <boost/simd/boolean.hpp>
#include <boost/simd/bitwise.hpp>
#include <boost/simd/operator.hpp>
#include <boost/simd/eulerian.hpp>
#include <boost/simd/predicates.hpp>

#include <boost/simd/function/load.hpp>
#include <boost/simd/function/aligned_load.hpp>
#include <boost/simd/function/store.hpp>
#include <boost/simd/function/aligned_store.hpp>

#include <boost/simd/function/group.hpp>
#include <boost/simd/function/slice.hpp>
#include <boost/simd/function/sum.hpp>
#include <boost/simd/function/plus.hpp>
#include <boost/simd/function/sqr.hpp>
#include <boost/simd/function/cos.hpp>
#include <boost/simd/function/sin.hpp>
#include <boost/simd/function/log.hpp>
#include <boost/simd/function/exp.hpp>
#include <boost/simd/function/sqrt.hpp>
#include <boost/simd/function/minus.hpp>
#include <boost/simd/function/remquo.hpp>
#include <boost/simd/function/rec.hpp>
#include <boost/simd/function/modf.hpp>

#include <boost/simd/function/sincos.hpp>
#include <boost/simd/function/hypot.hpp>
#include <boost/simd/function/atan2.hpp>

#include <boost/simd/function/shuffle.hpp>
#include <boost/simd/function/enumerate.hpp>

#include <boost/simd/function/if_inc.hpp>
#include <boost/simd/function/if_else.hpp>

#include <boost/simd/function/interleave.hpp>
#include <boost/simd/function/interleave_first.hpp>
#include <boost/simd/function/interleave_second.hpp>
#include <boost/simd/function/deinterleave.hpp>
#include <boost/simd/function/deinterleave_first.hpp>
#include <boost/simd/function/deinterleave_second.hpp>

#include "Allocators.hpp"
namespace RMWarp {

namespace bs = boost::simd;
namespace ba = boost::alignment;

template<class T = float>
using simd_reg = bs::pack<T>;


template<class T = float>
constexpr size_t simd_width = bs::cardinal_of<simd_reg<T> >();

template<class T = float>
constexpr auto simd_align = simd_reg<T>::alignment;

constexpr auto default_align = simd_align<float>;

}
