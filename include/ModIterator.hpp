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
#include <numeric>

#include "Range.hpp"
#include "Math.h"

namespace RMWarp {
template<class T>
struct mod_iterator {
    using iterator_category = std::random_access_iterator_tag;
    using value_type      = T;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference       = T&;
    using pointer         = T*;

    pointer         m_ptr{nullptr};
    difference_type m_idx{0};
    size_type       m_mask{0};
    constexpr mod_iterator() noexcept = default;
    constexpr mod_iterator(pointer ptr, difference_type idx, size_type _mask) noexcept
    : m_ptr(ptr) , m_idx(idx) , m_mask(_mask){}
    constexpr mod_iterator(const mod_iterator &o)       = default;
    constexpr mod_iterator(mod_iterator &&o) noexcept   = default;
    mod_iterator &operator=(const mod_iterator &o)      = default;
    mod_iterator &operator=(mod_iterator &&) noexcept   = default;
    template<class Y>
    constexpr mod_iterator(const mod_iterator<Y> &o) noexcept
    : m_ptr(o.m_ptr) , m_idx(o.m_idx) , m_mask(o.m_mask) { }
    template<class Y>
    constexpr mod_iterator(mod_iterator<Y> &&o) noexcept
    : m_ptr(o.m_ptr) , m_idx(o.m_idx) , m_mask(o.m_mask) { }
    template<class Y>
    mod_iterator &operator =(const mod_iterator<Y> &o) noexcept { m_ptr = o.m_ptr;m_idx = o.m_idx;m_mask = o.m_mask; return *this; }
    template<class Y>
    mod_iterator &operator =(mod_iterator<Y> &&o) noexcept { m_ptr = o.m_ptr;m_idx = o.m_idx;m_mask = o.m_mask; return *this;}
    void swap(mod_iterator &o)  noexcept { using std::swap; swap(m_ptr,o.m_ptr); swap(m_idx,o.m_idx); }
    friend void swap(mod_iterator &lhs, mod_iterator &rhs) noexcept { lhs.swap(rhs);}
    constexpr size_type mask() const   { return m_mask;}
    constexpr size_type size() const   { return mask() + 1ul;}
    constexpr size_type offset() const { return m_idx & mask();}
    constexpr pointer   data() const   { return m_ptr;}
    constexpr pointer   get()  const   { return data() + offset();}

    constexpr bool operator ==(const mod_iterator& o) const { return m_ptr == o.m_ptr && m_idx == o.m_idx;}
    constexpr bool operator !=(const mod_iterator& o) const { return !(*this == o);}
    constexpr bool operator  <(const mod_iterator& o) const { return m_ptr == o.m_ptr && (m_idx < o.m_idx);}
    constexpr bool operator  >(const mod_iterator& o) const { return m_ptr == o.m_ptr && (m_idx > o.m_idx);}
    constexpr bool operator <=(const mod_iterator& o) const { return m_ptr == o.m_ptr && (m_idx <= o.m_idx);}
    constexpr bool operator >=(const mod_iterator& o) const { return m_ptr == o.m_ptr && (m_idx >= o.m_idx);}

    constexpr mod_iterator operator +(difference_type diff) { return mod_iterator(m_ptr, m_idx+diff, m_mask);}
    constexpr mod_iterator operator -(difference_type diff) { return mod_iterator(m_ptr, m_idx-diff, m_mask);}
    constexpr difference_type operator - (const mod_iterator &o) { m_idx - o.m_idx; }

    constexpr mod_iterator &operator ++(){m_idx++;return *this;}
    constexpr mod_iterator &operator ++(int){auto ret = *this;++*this;return ret;}
    constexpr mod_iterator &operator --(){m_idx--;return *this;}
    constexpr mod_iterator &operator --(int){auto ret = *this;--*this;return ret;}
    constexpr mod_iterator &operator +=(difference_type diff) { m_idx += diff; return *this;}
    constexpr mod_iterator &operator -=(difference_type diff) { m_idx -= diff; return *this;}

    constexpr pointer   operator ->() const { return m_ptr + offset();}
    constexpr operator pointer() const      { return m_ptr + offset();}
    constexpr reference operator *() const  { return m_ptr[offset()];}
    constexpr reference operator[](difference_type diff) const { return m_ptr[(m_idx + diff) & mask()];}
    constexpr Range<pointer> contig() const
    {
        return { m_ptr + offset(), m_ptr + m_mask + 1 };
    }
};
}
