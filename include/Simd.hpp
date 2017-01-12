#pragma once

#include <alloca.h>

#include <cstdint>
#include <cstddef>
#include <cstdarg>
#include <cstdio>
#include <cassert>
#include <cerrno>
#include <csignal>
#include <climits>
#include <cfloat>
#include <cstring>
#include <cmath>
#include <ctime>

#include <sys/time.h>
#include <chrono>

#include <numeric>
#include <limits>
#include <type_traits>
#include <algorithm>
#include <utility>
#include <functional>
#include <iterator>
#include <initializer_list>
#include <tuple>
#include <vector>
#include <deque>
#include <memory>
#include <valarray>
#include <new>


#include <atomic>
#include <thread>
#include <mutex>
#include <future>
#include <shared_mutex>
#include <condition_variable>
#include <semaphore.h>


#include <boost/simd/pack.hpp>
#include <boost/simd/meta/cardinal_of.hpp>

#include <boost/simd/algorithm.hpp>

#include <boost/simd/memory/allocator.hpp>
#include <boost/align/aligned_allocator.hpp>
#include <boost/align/aligned_delete.hpp>

#include <boost/simd/function/fast.hpp>

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

namespace RMWarp {

namespace bs = boost::simd;
namespace ba = boost::alignment;

template<class T>
using align_alloc = bs::allocator<T>;

template<class T = float>
using simd_reg = bs::pack<T>;

template<class T = float>
using simd_vec = std::vector<T , align_alloc<T> >;
//        T , simd_reg<T>::alignment >
//    >;

template<class T = float>
constexpr size_t simd_width = bs::cardinal_of<simd_reg<T> >();

template<class T = float>
constexpr auto simd_alignment = simd_reg<T>::alignment;

constexpr auto default_alignment = simd_alignment<float>;


//template<class T>
//using simd_vec = std::vector<T, align_alloc<T> >;

}
