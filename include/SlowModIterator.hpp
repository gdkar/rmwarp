_Pragma("once")

#include <cstdint>
#include <algorithm>
#include <iterator>
#include <utility>

#include "Range.hpp"
#include "Math.hpp"

namespace RMWarp {
template<class T, class R = T&, class P = T*>
struct slow_mod_iterator {
    using iterator_category = std::random_access_iterator_tag;
    using value_type      = T;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference       = R;
    using pointer         = P;

    pointer         m_ptr {nullptr};
    difference_type m_idx {0};
    size_type       m_size{0};
    constexpr slow_mod_iterator() noexcept = default;
    constexpr slow_mod_iterator(pointer ptr, difference_type idx, size_type _size) noexcept
    : m_ptr(ptr) , m_idx(idx) , m_size(_size){}
    constexpr slow_mod_iterator(const slow_mod_iterator &o)       = default;
    constexpr slow_mod_iterator(slow_mod_iterator &&o) noexcept   = default;
    slow_mod_iterator &operator=(const slow_mod_iterator &o)      = default;
    slow_mod_iterator &operator=(slow_mod_iterator &&) noexcept   = default;
    template<class Y>
    constexpr slow_mod_iterator(const slow_mod_iterator<Y> &o) noexcept
    : m_ptr(o.m_ptr) , m_idx(o.m_idx) , m_size(o.m_size) { }
    template<class Y>
    constexpr slow_mod_iterator(slow_mod_iterator<Y> &&o) noexcept
    : m_ptr(o.m_ptr) , m_idx(o.m_idx) , m_size(o.m_size) { }
    template<class Y>
    slow_mod_iterator &operator =(const slow_mod_iterator<Y> &o) noexcept { m_ptr = o.m_ptr;m_idx = o.m_idx;m_size= o.m_size; return *this; }
    template<class Y>
    slow_mod_iterator &operator =(slow_mod_iterator<Y> &&o) noexcept { m_ptr = o.m_ptr;m_idx = o.m_idx;m_size= o.m_size; return *this;}
    void swap(slow_mod_iterator &o)  noexcept { using std::swap; swap(m_ptr,o.m_ptr); swap(m_idx,o.m_idx); }
    friend void swap(slow_mod_iterator &lhs, slow_mod_iterator &rhs) noexcept { lhs.swap(rhs);}
    constexpr size_type size() const   { return m_size ;}
    constexpr size_type offset() const { return m_idx % size();}
    constexpr pointer   data() const   { return m_ptr;}
    constexpr pointer   get()  const   { return data() + offset();}

    constexpr bool operator ==(const slow_mod_iterator& o) const { return m_ptr == o.m_ptr && m_idx == o.m_idx;}
    constexpr bool operator !=(const slow_mod_iterator& o) const { return !(*this == o);}
    constexpr bool operator  <(const slow_mod_iterator& o) const { return m_ptr == o.m_ptr && (m_idx < o.m_idx);}
    constexpr bool operator  >(const slow_mod_iterator& o) const { return m_ptr == o.m_ptr && (m_idx > o.m_idx);}
    constexpr bool operator <=(const slow_mod_iterator& o) const { return m_ptr == o.m_ptr && (m_idx <= o.m_idx);}
    constexpr bool operator >=(const slow_mod_iterator& o) const { return m_ptr == o.m_ptr && (m_idx >= o.m_idx);}

    constexpr difference_type operator - (const slow_mod_iterator &o) { return (m_ptr == o.m_ptr) ? m_idx - o.m_idx : throw std::invalid_argument("slow_mod_iterators not to same object.");}

    constexpr slow_mod_iterator &operator ++(){m_idx++;return *this;}
    constexpr slow_mod_iterator  operator ++(int){auto ret = *this;++*this;return ret;}
    constexpr slow_mod_iterator &operator --(){m_idx--;return *this;}
    constexpr slow_mod_iterator  operator --(int){auto ret = *this;--*this;return ret;}
    constexpr slow_mod_iterator &operator +=(difference_type diff) { m_idx += diff; return *this;}
    constexpr slow_mod_iterator &operator -=(difference_type diff) { m_idx -= diff; return *this;}

    friend constexpr slow_mod_iterator operator + (
        const slow_mod_iterator & lhs
      , difference_type rhs
        )
    { return slow_mod_iterator(lhs.m_ptr, lhs.m_idx+rhs, lhs.m_size); }
    friend constexpr slow_mod_iterator operator + (
        difference_type lhs
      , const slow_mod_iterator & rhs
        )
    { return slow_mod_iterator(rhs.m_ptr, rhs.m_idx+lhs, rhs.m_size); }
    friend constexpr slow_mod_iterator operator - (
        const slow_mod_iterator & lhs
      , difference_type rhs
        )
    { return slow_mod_iterator(lhs.m_ptr, lhs.m_idx-rhs, lhs.m_size); }
    constexpr pointer   operator ->() const { return m_ptr + offset();}
    constexpr reference operator *() const { return m_ptr[offset()];}
    constexpr reference operator[](difference_type diff) const { return m_ptr[(m_idx + diff) % size()];}
};
}
