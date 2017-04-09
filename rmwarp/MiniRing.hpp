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

#include "Math.hpp"
#include "Range.hpp"
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
    using container_type = std::vector<T>;
    using size_type       = typename container_type::size_type;
    using difference_type = int64_t;
    using reference       = typename container_type::reference;
    using const_reference = typename container_type::const_reference;
    using pointer         = typename container_type::pointer;
    using const_pointer   = typename container_type::const_pointer;
    using iterator        = slow_mod_iterator<value_type, reference,pointer>;
    using const_iterator  = slow_mod_iterator<value_type, const_reference,const_pointer>;
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
    difference_type last_index()   const { return m_widx - 1;}
    size_type       read_offset()  const { return read_index() % size();}
    size_type       write_offset() const { return write_index() % size();}
    size_type       last_offset() const { return last_index() % size();}
    const_pointer data()   const { return m_data.data();}
    pointer data()   { return m_data.data();}
    size_type size() const { return write_index() - read_index();}
    size_type space()const { return (read_index() + capacity()) - write_index();}
    bool      full() const { return (read_index() + capacity()) == write_index();}
    bool      empty()const { return read_index() == write_index();}

    void clear()
    {
        m_ridx = m_widx;
        std::fill(std::begin(m_data),std::end(m_data),value_type{});
    }
    void reset(int64_t idx)
    {
        m_ridx = m_widx = idx;
    }
    void resize(size_t new_size)
    {
        if(new_size == capacity())
            return;
        if(new_size < size()) {
            m_ridx = m_widx - new_size;
        }
        std::rotate(std::begin(m_data),std::begin(m_data) + read_offset(),std::end(m_data));
        m_data.resize(new_size);
        m_size = new_size;
        std::rotate(std::begin(m_data),std::end(m_data) - read_offset(),std::end(m_data));
    }
    iterator begin()              { return iterator(data(),read_index(),capacity());}
    const_iterator begin()  const { return const_iterator(data(),read_index(),capacity());}
    const_iterator cbegin() const { return begin();}

    iterator end()                { return iterator(data(),write_index(),capacity());}
    const_iterator end()    const { return const_iterator(data(),write_index(),capacity());}
    const_iterator cend()   const { return end();}

    reference front()             { return m_data[read_offset()];}
    const_reference front() const { return m_data[read_offset()];}

    reference back ()             { return m_data[last_offset()];}
    const_reference back () const { return m_data[last_offset()];}

    reference operator[](difference_type idx)
    {
        return m_data[((idx < 0 ? write_index() : read_index()) + idx) % capacity()];
    }
    const_reference operator[](difference_type idx) const
    {
        return m_data[((idx < 0 ? write_index() : read_index()) + idx) % capacity()];
    }
    const_reference at (difference_type idx) const
    {
        if(idx <= -size() || idx >= size())
            throw std::out_of_range();
        return m_data[((idx < 0 ? write_index() : read_index()) + idx) % capacity()];
    }
    void push_back_expanding()
    {
        if(full())
            resize(std::max(capacity(), 8ul)* 3 / 2);
        m_widx++;
    }
    void push_back_expanding(const_reference item)
    {
        push_back_expanding();
        back() = item;
    }
    void push_back_expanding(value_type&& item)
    {
        push_back_expanding();
        back() = std::forward<value_type>(item);
    }
    template<class... Args>
    void emplace_back_expanding(Args &&... args)
    {
        push_back_expanding();
        m_widx++;
        auto &x = back();
        x.~T();
        ::new (&x) T (std::forward<Args>(args)...);
    }
    void push_back()
    {
        if(full())
            pop_front();
        m_widx++;
    }
    void push_back(const_reference item)
    {
        if(full())
            pop_front();
        m_widx++;
        back() = item;
    }
    void push_back(T && item)
    {
        if(full())
            pop_front();
        m_widx++;
        back() = std::forward<T>(item);
    }
    template<class... Args>
    void emplace_back(Args &&...args)
    {
        if(full())
            pop_front();
        m_widx ++;
        auto &x = back();
        x.~T();
        ::new (&x) T (std::forward<Args>(args)...);
    }
    void pop_front()
    {
        if(!empty()) {
            m_ridx++;
        } else {
            throw std::runtime_error("pop from empty fifo.");
        }
    }
};
}
