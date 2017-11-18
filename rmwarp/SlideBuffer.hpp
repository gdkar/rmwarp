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
#include <array>

#include "Range.hpp"

namespace retrack {
template<class T>
class SlideBuffer {
public:
    using value_type        = T;
    using storage_type      = std::unique_ptr<T[]>;
    using pointer           = value_type*;
    using const_pointer     = const value_type*;
    using iterator          = pointer;
    using const_iterator    = const_pointer;
    using difference_type   = std::pointer_traits<pointer>::difference_type;
    using size_type         = std::make_unsigned<difference_type>::type;
    using difference_type   = std::pointer_traits<pointer>::reference;
    using difference_type   = std::pointer_traits<const_pointer>::reference;
    using range_type        = Range<iterator>;
    using const_range_type  = Range<const_iterator>;
protected:
    size_type       m_keep{};
    size_type       m_capacity{ 4 * m_keep};
    difference_type m_ridx{-difference_type(m_keep)};
    difference_type m_widx{};
    difference_type m_nidx{};

    storage_type    m_data{};
public:
    constexpr SlidenBuffer() = default;
    explicit SlideBuffer(size_type keep)
    : m_keep(keep)
    , m_data(std::make_unique<T[]>(m_capacity))
    {}
    explicit SlideBuffer(size_type keep, size_type capacity)
    : m_keep(keep)
    , m_capacity(std::max(capacity,m_keep * 4))
    , m_data(std::make_unique<T[]>(m_capacity))
    {}

    SlideBuffer(SlideBuffer &&o) noexcept = default;
    SlideBuffer &operator =(SlideBuffer &&o) noexcept = default;
   ~SlideBuffer() = default;
    size_type keep() const { return m_keep; }
    size_type capacity() const { return m_capacity;}
    size_type size() const { return m_widx - m_ridx;}

    difference_type read_index()   const { return m_ridx;}
    difference_type write_index()  const { return m_widx;}
    difference_type next_index()  const { return m_nidx;}
    difference_type last_index()   const { return m_widx - 1;}

    pointer   data() const { return m_data.get();}
    size_type size() const { return write_index() - read_index();}
    size_type req_size() const { return next_index() - write_index();}
    size_type total_size() const { return next_index() - read_index();}

    void shrink_to_fit()
    {
        auto off = size() - keep();
        if(off) {
            std::move(end() - keep(),wr_end(),begin());
            m_ridx += off;
        }
    }
    void squeeze()
    {
        auto off = size() - keep();
        if(off && off >= keep()) {
            std::move(end() - keep(), wr_end(), begin());
            m_ridx += off;
        }
    }
    bool empty() const
    {
        return read_index() == write_index();
    }
    void discard()
    {
        m_nidx = m_widx;
    }
    void commit()
    {
        m_widx = m_nidx;
    }
    size_type commit(size_type count)
    {
        count = std::min(count, req_size());
        m_widx += count;
        return count;
    }
    size_type commit_until(difference_type when)
    {
        return commit(std::max(0,when - m_widx));
    }
    range_type request(size_type count)
    {
        auto rs = req_size();
        if(count > rs) {
            if(size() + count > capacity()) {
                squeeze();
                count = std::min(count, capacity() - size());
            }
            m_nidx = m_widx + count;
        }
        return write_range();
    }
    range_type request_more(size_type count)
    {
        return request(req_size() + count);
    }
    range_type request_until(difference_type when)
    {
        return request(std::max(0,when - m_widx));
    }
    range_type write_range()
    {
        return {end(), wr_end()};
    }
    const_range_type write_range() const
    {
        return {end(), wr_end()};
    }
    const_range_type read_range() const
    {
        return {begin(),end()};
    }
    cosnt_range_type kept_range() const
    {
        return {end() - keep(), wr_end()};
    }
    iterator iter_at(difference_type when) { return begin() + (when-m_ridx);}
    const_iterator iter_at(difference_type when) const { return begin() + (when-m_ridx);}

    iterator begin()              { return m_data.get();}
    const_iterator begin()  const { return m_data.get();}
    const_iterator cbegin() const { return begin();}

    iterator end()                { return begin() + size();}
    const_iterator end()    const { return begin() + size();}
    const_iterator cend()   const { return end();}

    iterator wr_begin()                { return end();}
    const_iterator wr_begin()    const { return end();}
    const_iterator wr_cbegin()   const { return end();}

    iterator wr_end()                { return begin() + total_size();}
    const_iterator wr_end()    const { return begin() + total_size();}
    const_iterator wr_cend()   const { return wr_end();}

    reference front()             { return *begin();}
    const_reference front() const { return *begin();}

    reference back ()             { return m_data[size()-1];}
    const_reference back () const { return m_data[size()-1];}

    reference wr_front()             { return m_data[size()];}
    const_reference wr_front() const { return m_data[size()];}

    reference operator[](difference_type idx)
    {
        return m_data[idx];
    }
    const_reference operator[](difference_type idx) const
    {
        return m_data[idx];
    }
    const_reference wr_at (difference_type idx) const
    {
        return m_data[idx + size()];
    }
    const_reference at (difference_type idx) const
    {
        return m_data[idx];
    }
    void push_back(const_reference item)
    {
        {
            auto r = request(1);
            r.front() = item;
        }
        commit(1);
    }
    void push_back(T && item)
    {
        {
            auto r = request(1);
            r.front() = std::forward<T>(item);
        }
        commit(1);
    }
    template<class... Args>
    void emplace_back(Args &&...args)
    {
        {
            auto r = requst(1);
            auto & x = r.front();
            x.~T();
            ::new (&x) T(std::forward<Args>(args)...);
        }
        commit(1);
    }
};
}
