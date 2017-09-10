#pragma once

#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <initializer_list>
#include <limits>
#include "rmwarp/Simd.hpp"

namespace RMWarp {

template<int k, class T>
constexpr std::enable_if_t<std::is_unsigned<T>,T> rotl(T t)
{
//    using U = std::make_unsigned_t<T>;
    return T((t<<k)|(U(t)>>(std::numeric_limits<T>::digits-k)));
}
constexpr uint64_t splitmix64(uint64_t &xr) {
    auto z = (xr += 0x9E3779B97F4A7C15UL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    return z ^ (z >> 31);
}
template<size_t N = 1ul>
constexpr bs::pack<uint64_t,N> splitmix64(bs::pack<uint64_t,N> &xr) {
    auto z = (xr += 0x9E3779B97F4A7C15UL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    return z ^ (z >> 31);
}

template<size_t N>
struct xoroshiro128plus {
    using value_type = typename std::conditional<
        N>1ul,
        bs::pack<uint64_t, N>;
        uint64_t>::type;
    value_type s0{bs::enumerate<value_type>(1,2)};
    value_type s1{bs::enumerate<value_type>(16,32)};
//    uint64_t    s0{1};
//    uint64_t    s1{2};
    constexpr xoroshiro128plus() = default;
    constexpr xoroshiro128plus(const xoroshiro128plus&) = default;
    constexpr xoroshiro128plus(xoroshiro128plus&&) noexcept = default;
    xoroshiro128plus&operator=(const xoroshiro128plus&) = default;
    xoroshiro128plus&operator=(xoroshiro128plus&&) noexcept = default;
    constexpr xoroshiro128plus(uint64_t _s0, uint64_t _s1)
    : s0(_s0),s1(_s1){}
    constexpr xoroshiro128plus(value_type _s0, value_type _s1)
    : s0(_s0),s1(_s1){}

    xoroshiro128plus(value_type _seed)
    : xoroshiro128plus{splitmix64(_seed),splitmix64(_seed)}{}

    constexpr xoroshiro128plus next() const
    {
        auto _s1 = s0^s1;
        return xoroshiro128plus(bs::rol(s0,55) ^ _s1 ^(_s1<<14),bs::rol(_s1,36));
    }
    constexpr xoroshiro128plus operator ()() const { return next(); }
    constexpr operator value_type() const
    {
        return s0 + s1;
    }
    static constexpr const size_t digits = std::numeric_limits<uint64_t>::digits;
    constexpr xoroshiro128plus jump() const
    {
        auto res = xoroshiro128plus(0,0);

        for(auto && word : { 0xbeac0467eba5facbul, 0xd86b048b86aa9922ul } ){
            for(auto bit = 1ul; bit; bit <<= 1) {
                if(word & bit) {
                    res.s0 ^= s0;
                    res.s1 ^= s1;
                }
                res = res();
            }
        }
        return res;
    }
};
template<class Iter>
auto xsr128_fill(Iter _bit, Iter _eit, uint64_t _s0, uint64_t _s1)
{
    auto xsr = xoroshiro128plus{_s0,_s1};
    auto gen = [xsr]() mutable { return uint64_t(xsr = xsr());};
    return std::generate(_bit,_eit,gen);
}
template<class Iter>
auto xsr128_fill(Iter _bit, Iter _eit, xoroshiro128plus& xsr)
{
    auto gen = [xsr]() mutable { return uint64_t(xsr = xsr());};
    return std::generate(_bit,_eit,gen);
}
template<class Iter, class... Args>
auto xsr128_fill(Iter _bit, Iter _eit, Args &&  ...args)
{

    auto xsr = xoroshiro128plus(std::forward<Args>(args)...);
    auto gen = [xsr]() mutable { return uint64_t(xsr = xsr());};
    return std::generate(_bit,_eit,gen);
}
}
