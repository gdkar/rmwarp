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
#include "SlowModIterator.hpp"
namespace RMWarp {
template<class T>
class MiniRing {
protected:
    size_t                 m_size{0};
    int64_t                m_ridx{0};
    int64_t                m_widx{0};
    std::vector<T>         m_data{m_size};
public:
    using value_type = T;
    using size_type       = size_t;
    using difference_type = int64_t;
    using reference       = T&;
    using const_reference = const T&;
    using pointer         = T*;
    using const_pointer   = const T*;
    using iterator        = slow_mod_iterator<T>;
    using const_iterator  = slow_mod_iterator<const T>;
    using range_type      = Range<iterator>;
    MiniRing() = default;
    explicit MiniRing(size_type cap, const T & item = T{})
    : m_size(cap)
    , m_data(cap, item){}
    MiniRing(MiniRing &&o) noexcept
    {
        using std::swap;
        swap(m_size,o.m_size);
        swap(m_data,o.m_data);
        swap(m_widx,o.m_widx);
        swap(m_ridx,o.m_ridx);
    }
    MiniRing &operator =(MiniRing &&o) noexcept
    {
        using std::swap;
        swap(m_size,o.m_size);
        swap(m_data,o.m_data);
        swap(m_widx,o.m_widx);
        swap(m_ridx,o.m_ridx);
        return *this;
    }
   ~MiniRing() = default;
    size_type capacity()           const { return m_size; }
    difference_type read_index()   const { return m_ridx;}
    difference_type write_index()  const { return m_widx;}
    difference_type last_index()  const { return m_widx - 1;}
    size_type       read_offset()  const { return read_index() % size();}
    size_type       write_offset() const { return write_index() % size();}
    size_type       last_offset() const { return last_index() % size();}
    pointer data()   const { return m_data.get();}
    size_type size() const { return write_index() - read_index();}
    size_type space()const { return (read_index() + capacity()) - write_index();}
    bool      full() const { return (read_index() + capacity()) == write_index();}
    bool      empty()const { return read_index() == write_index();}

    iterator begin()              { return iterator(data(),read_index(),size());}
    const_iterator begin()  const { return const_iterator(data(),read_index(),size());}
    const_iterator cbegin() const { return begin();}

    iterator end()                { return iterator(data(),write_index(),size());}
    const_iterator end()    const { return const_iterator(data(),write_index(),size());}
    const_iterator cend()   const { return end();}

    reference front()             { return *begin();}
    const_reference front() const { return *begin();}

    reference back ()             { return m_data[last_offset()];}
    const_reference back () const { return m_data[last_offset()];}

    reference operator[](difference_type idx)
    {
        return begin()[idx];
    }
    const_reference operator[](difference_type idx) const
    {
        return cbegin()[idx];
    }
    const_reference at (difference_type idx) const
    {
        if(idx < 0 || size_type(idx) >= size())
            throw std::out_of_range();
        return cbegin()[idx];
    }
    void push_back(const_reference item)
    {
        if(full())
            pop_front();
        back() = item; m_widx ++;
    }
    void push_back(T && item)
    {
        if(full())
            pop_front();
        back() = std::forward<T>(item); m_widx.fetch_add(1);
    }
    template<class... Args>
    void emplace_back(Args &&...args)
    {
        if(full())
            pop_front();
        auto &x = *end();
        x.~T();a
        ::new (&x) T (std::forward<Args>(args)...);
        m_widx.fetch_add(1);
    }
    void pop_front()
    {
        if(!empty()) {
            m_ridx++;
        } else {
            throw std::invalid_argument("pop from empty fifo.");
        }
    }
    bool try_push_back(const_reference item)
    {
        if(full())
            return false;
        *end = item;
        m_widx ++;
        return true;
    }
    bool try_push_back(T && item)
    {
        if(full())
            return false;
        *end() = std::forward<T>(item);
        m_widx ++;
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
        m_ridx ++;
        return true;
    }
};
}
