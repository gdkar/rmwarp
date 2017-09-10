#pragma once

#include <type_traits>

#include <boost/simd/meta/cardinal_of.hpp>

#include <boost/align/aligned_allocator.hpp>
#include <boost/align/aligned_delete.hpp>

#include <boost/simd/algorithm.hpp>
#include <boost/simd/bitwise.hpp>
#include <boost/simd/boolean.hpp>
#include <boost/simd/constant.hpp>
#include <boost/simd/decorator.hpp>
#include <boost/simd/eulerian.hpp>
#include <boost/simd/forward.hpp>
#include <boost/simd/hyperbolic.hpp>
#include <boost/simd/ieee.hpp>
#include <boost/simd/iterate.hpp>
#include <boost/simd/literal.hpp>
#include <boost/simd/logical.hpp>
#include <boost/simd/mask.hpp>
#include <boost/simd/memory.hpp>
#include <boost/simd/operator.hpp>
#include <boost/simd/pack.hpp>
#include <boost/simd/predicates.hpp>
#include <boost/simd/range.hpp>
#include <boost/simd/reduction.hpp>
#include <boost/simd/swar.hpp>
#include <boost/simd/trigonometric.hpp>

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
