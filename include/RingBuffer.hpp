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
#include "ModIterator.hpp"
namespace RMWarp {
template<class T>
class RingBuffer {
protected:
    size_t                 m_capacity{0};
    size_t                 m_mask    {0};
    std::atomic<int64_t>   m_ridx{0};
    std::atomic<int64_t>   m_widx{0};
    std::unique_ptr<T[]>   m_data{};
public:
    using value_type = T;
    using size_type       = size_t;
    using difference_type = int64_t;
    using reference       = T&;
    using const_reference = const T&;
    using pointer         = T*;
    using const_pointer   = const T*;
    using iterator        = mod_iterator<T>;
    using const_iterator  = mod_iterator<const T>;
    using range_type      = Range<iterator>;
    constexpr RingBuffer() = default;
    explicit RingBuffer(size_type cap)
    : m_capacity(roundUpToPowerOf2(cap))
    , m_mask(m_capacity - 1)
    , m_data(std::make_unique<T[]>(m_capacity)){}
    RingBuffer(RingBuffer &&o) noexcept
        : m_capacity(std::exchange(o.m_capacity,0))
        ,m_mask(std::exchange(o.m_mask,0))
        ,m_ridx(o.m_ridx.exchange(0))
        ,m_widx(o.m_widx.exchange(0))
    { m_data.swap(o.m_data);}
    RingBuffer &operator =(RingBuffer &&o) noexcept
    {
        using std::swap;
        m_ridx.exchange(o.m_ridx.exchange(0));
        m_widx.exchange(o.m_widx.exchange(0));
        swap(m_capacity, o.m_capacity);
        swap(m_mask,     o.m_mask    );
        swap(m_data, o.m_data);
    }
   ~RingBuffer() = default;
    constexpr size_type capacity() const { return m_capacity; }
    constexpr size_type mask()     const { return m_mask;}
    difference_type read_index()   const { return m_ridx.load(std::memory_order_acquire);}
    difference_type write_index()  const { return m_widx.load(std::memory_order_acquire);}
    difference_type last_index()   const { return m_widx.load(std::memory_order_acquire) - 1;}
    difference_type next_index()   const { return m_widx.load(std::memory_order_acquire) + 1;}
    size_type       read_offset()  const { return read_index() & mask();}
    size_type       write_offset() const { return write_index() & mask();}
    size_type       last_offset()  const { return last_index() & mask();}
    size_type       next_offset()  const { return next_index() & mask();}
    pointer data()   const { return m_data.get();}
    size_type size() const { return write_index() - read_index();}
    size_type space()const { return (read_index() + capacity()) - write_index();}
    bool      full() const { return (read_index() + capacity()) == write_index();}
    bool      empty()const { return read_index() == write_index();}

    range_type read_range()
    {
        return make_range(
            iterator(data(),read_index(),mask())
          , iterator(data(),write_index(),mask()));
    }
    range_type read_range(size_type _size, difference_type _diff = 0)
    {
        _size = std::min<size_type>(_size,std::max<difference_type>(size() - _diff,0));
        return make_range(
            iterator(data(),read_index() + _size,mask())
          , iterator(data(),read_index() + _size + _diff,mask()));
    }
    range_type write_range()
    {
        return make_range(
            iterator(data(),mask(),write_index())
          , iterator(data(),mask(),read_index() + capacity()));
    }
    range_type write_range(size_type _size)
    {
        _size = std::min<size_type>(_size,space());
        return make_range(
            iterator(data(),write_index(),mask())
          , iterator(data(),write_index() + _size,mask()));
    }
    size_type write_advance(size_type count)
    {
        count = std::min(count,space());
        m_widx.fetch_add(count,std::memory_order_release);
        return count;
    }
    size_type read_advance(size_type count)
    {
        count = std::min(count,size());
        m_ridx.fetch_add(count,std::memory_order_release);
        return count;
    }
    template<class Iter>
    size_type peek_n(Iter start, size_type count, difference_type offset = 0)
    {
        auto r = read_range((count = std::min(count, size()-offset)),offset);
        auto stop = std::copy_n(r.cbegin(), r.size(), start);
        return std::distance(start,stop);
    }
    template<class Iter>
    Iter peek(Iter start, Iter stop, difference_type offset = 0)
    {
        auto r = read_range(std::distance(start,stop),offset);
        return std::copy(r.cbegin(), r.cend(), start);
    }
    template<class Iter>
    size_type read_n(Iter start, size_type count)
    {
        auto r = read_range(count = std::min(count, size()));
        std::copy_n(r.begin(), r.size(), start);
        read_advance(r.size());
        return r.size();
    }
    template<class Iter>
    Iter read(Iter start, Iter stop)
    {
        auto r = read_range(std::distance(start,stop));
        stop = std::copy(r.cbegin(), r.cend(), start);
        m_ridx.fetch_add(r.size(), std::memory_order_release);
        return stop;
    }
    template<class Iter>
    size_type write_n(Iter start, size_type count)
    {
        auto r = write_range(count = std::min(count,space()));
        std::copy_n(start,r.size(),r.begin());
        write_advance(r.size());
        return r.size();
    }
    template<class Iter>
    Iter write(Iter start, Iter stop)
    {
        auto r = write_range(std::distance(start,stop));
        std::copy_n(start,r.size(),r.begin());
        m_widx.fetch_add(r.size(),std::memory_order_release);
        return std::next(start,r.size());
    }
    iterator begin()              { return iterator(data(),read_index(),mask());}
    const_iterator begin()  const { return const_iterator(data(),read_index(),mask());}
    const_iterator cbegin() const { return begin();}

    iterator end()                { return iterator(data(),write_index(),mask());}
    const_iterator end()    const { return const_iterator(data(),write_index(),mask());}
    const_iterator cend()   const { return end();}

    reference front()             { return m_data[read_offset()];}
    const_reference front() const { return m_data[read_offset()];}

    reference back ()             { return m_data[last_offset()];}
    const_reference back () const { return m_data[last_offset()];}

    void push_back(const_reference item)
    {
        if(!full()) { *end() = item; m_widx.fetch_add(1);}
        else { throw std::invalid_argument("push onto full fifo.");}
    }
    void push_back(T && item)
    {
        if(!full()) { *end() = std::forward<T>(item); m_widx.fetch_add(1);}
        else { throw std::invalid_argument("push onto full fifo.");}
    }
    template<class... Args>
    void emplace_back(Args &&...args)
    {
        if(!full()) {
            auto & x = *end();
            x.~T();
            ::new (&x) T (std::forward<Args>(args)...);
            m_widx.fetch_add(1);
        }else {
            throw std::invalid_argument("push onto full fifo.");
        }
    }
    void pop_front()
    {
        if(!empty()) {m_ridx.fetch_add(1);}
        else { throw std::invalid_argument("pop from empty fifo.");}
    }
    bool try_push_back(const_reference item)
    {
        if(full())
            return false;
        *end() = item;
        m_widx.fetch_add(1);
        return true;
    }
    bool try_push_back(T && item)
    {
        if(full())
            return false;
        *end() = std::forward<T>(item);
        m_widx.fetch_add(1);
        return true;
    }
    template<class... Args>
    bool try_emplace_back(Args &&...args)
    {
        if(!full()) {
            auto &x = *end();
            x.~T();
            ::new (&x) T (std::forward<Args>(args)...);
            m_widx.fetch_add(1);
            return true;
        }else {
            return false;
        }
    }

    bool try_pop_front(reference item)
    {
        if(empty())
            return false;
        item = front();
        m_ridx.fetch_add(1);
        return true;
    }
};
}
