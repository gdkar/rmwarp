_Pragma("once")

#include <memory>
#include <climits>
#include <cfloat>
#include <exception>
#include <stdexcept>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <cmath>
#include <cstring>
#include <atomic>
#include <utility>
#include <algorithm>
#include <limits>
#include <array>
#include <numeric>

namespace RMWarp {
template<class Iter>
struct Range {
    using size_type       = size_t;//typename std::iterator_traits<Iter>::size_type;
    using difference_type = typename std::iterator_traits<Iter>::difference_type;
    using value_type      = typename std::iterator_traits<Iter>::value_type;
    using reference       = typename std::iterator_traits<Iter>::reference;
    using pointer         = typename std::iterator_traits<Iter>::pointer;
    Iter _begin{};
    Iter _end  {};
    constexpr Range() = default;
    constexpr Range(const Iter &_b, const Iter &_e) : _begin(_b),_end(_e){}
    constexpr Range(const Range &o)        = default;
    constexpr Range(Range &&o) noexcept    = default;
    Range &operator = (const Range &o)     = default;
    Range &operator = (Range &&o) noexcept = default;
    constexpr Iter begin() const { return _begin; }
    constexpr Iter end()   const { return _end; }
    constexpr Iter cbegin()const { return _begin; }
    constexpr Iter cend()  const { return _end; }
    constexpr bool empty() const { return _begin == _end;}
    constexpr operator bool() const { return !empty();}
    constexpr bool operator !() const { return empty();}
    constexpr size_type size() const { return std::distance(begin(),end());}
    constexpr reference operator[](difference_type idx) { return *std::next(begin(),idx);}
    constexpr const reference operator[](difference_type idx) const { return *std::next(begin(),idx);}
    constexpr std::array<Range,2> split(size_type _size)
    {
        if(_size < size()) {
            auto _mid = _begin + _size;
            return { Range(_begin,_mid),Range(_mid,_end)};
        }else{
            return { Range(_begin,_end),Range(_end,_end)};
        }
    }
    constexpr Range split_first(size_type _size)
    {
        return _size <= size() ? { _begin, _begin + _size} : *this;
    }
    constexpr Range split_second(size_type _size)
    {
        return _size <= size() ? { _begin + _size, _end} : { _end,_end};
    }
};

template<class Iter>
Range<Iter> make_range(Iter _begin, Iter _end) { return {_begin,_end}; }

template<class Container>
Range<typename Container::iterator> iter_range(Container && c)
{
    using std::begin;
    using std::end;
    using std::forward;
    return {begin(forward<Container>(c)),end(forward<Container>(c))};
}
template<class I0, class I1>
std::array<std::pair< Range<I0>,Range<I1> >, 3>
split_contig(const std::array<I0,2> c0, const std::array<I1,2> & c1)
{
    auto && f0 = std::get<0>(c0);
    auto && l0 = std::get<1>(c0);
    auto && f1 = std::get<0>(c1);
    auto && l1 = std::get<1>(c1);
    auto s0 = f0.size();
    auto s1 = f1.size();
    auto res = std::array<std::pair< Range<I0>,Range<I1> >, 3>{};

    if(s0 < s1) {
        auto sp0 = f1.split(s0);
        std::get<0>(res) = std::make_pair(f0,std::get<0>(sp0));
        auto m0 = std::get<1>(sp0);
        auto ms0 = m0.size();
        auto ls0 = l0.size();
        if(ls0 <= ms0) {
            std::get<1>(res) = std::make_pair(l0,m0.split_first(ls0));
        }else{
            auto sp1 = l0.split(ms0);
            std::get<1>(res) = std::make_pair(std::get<0>(sp1), m0);
            auto m1 = std::get<1>(sp1);
            auto seg = std::min(m1.size(),l1.size());
            std::get<2>(res) = std::make_pair(m1.split_first(seg),l1.split_first(seg));
        }
    }else{
        auto sp0 = f0.split(s1);
        std::get<0>(res) = std::make_pair(std::get<0>(sp0),f1);
        auto m0 = std::get<1>(sp0);
        auto ms0 = m0.size();
        auto ls0 = l0.size();
        if(ls0 <= ms0) {
            std::get<1>(res) = std::make_pair(m0.split_first(ls0),l1);
        }else{
            auto sp1 = l1.split(ms0);
            std::get<1>(res) = std::make_pair(m0,std::get<0>(sp1));
            auto m1 = std::get<1>(sp1);
            auto seg = std::min(m1.size(),l0.size());
            std::get<2>(res) = std::make_pair(l1.split_first(seg),m1.split_first(seg));
        }
        return res;
    }
}
}
