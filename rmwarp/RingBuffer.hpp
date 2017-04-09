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

#include "Math.hpp"
#include "Simd.hpp"
#include "sysutils.hpp"
#include "Range.hpp"
#include "ModIterator.hpp"

namespace RMWarp {
template<class T>
class RingBuffer {
protected:
    std::atomic<int64_t>   m_ridx{0};
    std::atomic<int64_t>   m_widx{0};

    size_t                 m_capacity{0};
    size_t                 m_mask    {m_capacity ? (m_capacity - 1ul) : 0ul};
    aligned_ptr<T[]>       m_data{m_capacity ? make_aligned<T[]>(m_capacity) : nullptr};
public:
    using value_type            = T;
    using size_type             = std::size_t;
    using difference_type       = int64_t;
    using reference             = T&;
    using const_reference       = const T&;
    using pointer               = T*;
    using const_pointer         = const T*;
    using iterator              = mod_iterator<T>;
    using const_iterator        = mod_iterator<const T>;
    using range_type            = Range<iterator>;
    using const_range_type      = Range<const_iterator>;
    using contig_type           = Range<pointer>;
    constexpr RingBuffer() = default;
    explicit RingBuffer(size_type cap)
    : m_capacity(roundup(cap))
    , m_mask(m_capacity - 1)
    , m_data(make_aligned<T[]>(m_capacity)){}
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

    size_type       read_offset()  const { return read_index() & mask();}
    size_type       write_offset() const { return write_index() & mask();}
    size_type       last_offset()  const { return last_index() & mask();}

    pointer   data() const { return m_data.get();}
    size_type size() const { return write_index() - read_index();}
    size_type space()const { return (read_index() + capacity()) - write_index();}
    size_type getReadSpace() const { return size(); }
    size_type getWriteSpace() const { return space(); }
    bool      full() const { return (read_index() + capacity()) == write_index();}
    bool      empty()const { return read_index() == write_index();}

    const_range_type read_range() const
    {
        return make_range( cbegin() , cend() );
    }
    const_range_type read_range(size_type _size) const
    {
        _size = std::min<size_type>(_size, size());
        return make_range(cbegin(), cbegin() + _size);
    }

    range_type read_range(size_type _size, difference_type _diff) const
    {
        _diff = std::min<size_type>(std::max<difference_type>(0,_diff), size());
        _size = std::min<size_type>(_size, size() - _diff);
        return make_range(begin() + _diff, begin() + _diff+ _size);
    }
    std::array<contig_type,2> read_contig() const
    {
        auto ri = read_index();
        auto wi = write_index();
        auto ptr = m_data.get();

        auto ro = ri & mask();
        auto wo = wi & mask();

        if(wo > ro) {
            return { make_range(ptr + ro, ptr + wo),make_range(ptr + wo, ptr + wo)};
        }else if(wi != ri) {
            return { make_range(ptr + ro, ptr + capacity()),make_range(ptr, ptr + wo)};
        }else{
            return { make_range(ptr + ro, ptr + ro),make_range(ptr + ro, ptr + ro)};
        }
    }
    std::array<contig_type,2> read_contig(size_type _size) const
    {
        auto ri = read_index();
        auto wi = write_index();
        auto ptr = m_data.get();

        _size = std::min<size_type>(_size, wi - ri);

        auto ei = ri + _size;
        auto ro = ri & m_mask;
        auto eo = ei & m_mask;

        if(eo > ro) {
            return { make_range(ptr + ro, ptr + eo),make_range(ptr + eo, ptr + eo)};
        }else if(wi != ri) {
            return { make_range(ptr + ro, ptr + m_capacity),make_range(ptr, ptr + eo)};
        }else{
            return { make_range(ptr + ro, ptr + ro),make_range(ptr + ro, ptr + ro)};
        }
    }
    std::array<contig_type,2> read_contig(size_type _size, difference_type _diff) const
    {
        auto ri = read_index();
        auto wi = write_index();
        auto ptr = m_data.get();
        _diff = std::min<difference_type>(_diff, wi - ri);
        auto bi = ri + _diff;
        _size = std::min<size_type>(_size, wi - bi);
        auto ei = bi + difference_type(_size);

        auto bo = bi & m_mask;
        auto eo = ei & m_mask;

        if(eo > bo) {
            return { make_range(ptr + bo, ptr + eo),make_range(ptr + eo, ptr + eo)};
        }else if(bi != ei) {
            return { make_range(ptr + bo, ptr + m_capacity),make_range(ptr, ptr + eo)};
        }else{
            return { make_range(ptr + bo, ptr + bo),make_range(ptr + bo, ptr + bo)};
        }
    }
    range_type write_range()
    {
        return make_range(end(), wr_end());
    }
    range_type write_range(size_type _size)
    {
        _size = std::min<size_type>(_size,space());
        return make_range( end(), end() + _size);
    }
    std::array<contig_type,2> write_contig()
    {
        auto ri = read_index() + capacity();
        auto wi = write_index();
        auto ptr = m_data.get();

        auto ro = ri & mask();
        auto wo = wi & mask();

        if(ro < wo) {
            return { make_range(ptr + wo, ptr + ro),make_range(ptr + ro, ptr + ro)};
        }else if(wi != ri ) {
            return { make_range(ptr + wo, ptr + capacity()),make_range(ptr, ptr + ro)};
        }else{
            return { make_range(ptr + wo, ptr + wo),make_range(ptr + wo, ptr + wo)};
        }
    }
    std::array<contig_type,2> write_contig(size_type _size)
    {
        auto ri = read_index() + capacity();
        auto wi = write_index();
        auto ptr = m_data.get();

        _size = std::min<size_type>(_size, ri - wi);

        auto ei = wi + difference_type(_size);
        auto eo = ei & mask();
        auto wo = wi & mask();

        if(eo > wo) {
            return { make_range(ptr + wo, ptr + eo),make_range(ptr + eo, ptr + eo)};
        }else if(wi != ei) {
            return { make_range(ptr + wo, ptr + capacity()),make_range(ptr, ptr + eo)};
        }else{
            return { make_range(ptr + wo, ptr + wo),make_range(ptr + wo, ptr + wo)};
        }
    }
    size_type write_advance(size_type count)
    {
        count = std::min(count,space());
        m_widx.fetch_add(count,std::memory_order_release);
        return count;
    }
    size_type read_advance(size_type count)
    {
        count = std::min(count, size());
        m_ridx.fetch_add(count,std::memory_order_release);
        return count;
    }
    template<class Iter>
    size_type peek_n(Iter start, size_type count, difference_type offset = 0) const
    {
        auto stop = start;
        auto wrote = size_type{};
        for(auto r : read_contig(count,offset)) {
            if(r) {
                stop = std::copy(r.cbegin(),r.cend(),stop);
                wrote += r.size();
            }
        }
        return wrote;
    }
    template<class Iter>
    Iter peek(Iter start, Iter stop, difference_type offset = 0) const
    {
        auto count = std::distance(start,stop);
        stop = start;
        for(auto r : read_contig(count,offset)) {
            if(r)
                stop = std::copy(r.cbegin(),r.cend(),stop);
        }
        return std::distance(start,stop);
    }
    template<class Iter>
    size_type read_n(Iter start, size_type count)
    {
        auto stop = start;
        auto wrote = size_type{};
        for(auto r : read_contig(count)) {
            if(r) {
                stop = std::copy(r.cbegin(),r.cend(),stop);
                m_ridx.fetch_add(r.size());
                wrote += r.size();
            }
        }
        return wrote;
    }
    template<class Iter>
    Iter read(Iter start, Iter stop)
    {
        auto count = std::distance(start,stop);
        stop = start;
        for(auto r : read_contig(count)) {
            if(r) {
                stop = std::copy(r.cbegin(),r.cend(),stop);
                m_ridx.fetch_add(r.size());
            }
        }
        return stop;
    }
    template<class Iter>
    size_type write(Iter start, size_type count)
    {
        return write_n(start,count);
    }
    template<class Iter>
    size_type read(Iter start, size_type count)
    {
        return read_n(start,count);
    }
    template<class Iter>
    size_type peek(Iter start, size_type count, difference_type diff = 0) const
    {
        return peek_n(start,count,diff);
    }
    template<class Iter>
    size_type write_n(Iter start, size_type count)
    {
        auto wrote = size_type{};
        for(auto r : write_contig(count)) {
            if(r) {
                auto sz = r.size();
                std::copy_n(start, sz, r.begin());
                start += sz;
                wrote += sz;
                m_widx.fetch_add(sz);
            }
        }
        return wrote;
    }
    template<class Iter>
    Iter write(Iter start, Iter stop)
    {
        auto count = std::distance(start,stop);
        for(auto r : write_contig(count)) {
            if(r) {
                auto sz = r.size();
                std::copy_n(start, sz, r.begin());
                start += sz;
                m_widx.fetch_add(sz);
            }
        }
        return start;
    }
    iterator find(int64_t _pts)
    {
        auto ri = read_index();
        auto wi = write_index();
        if(_pts < ri || _pts > wi)
            iterator(data(),wi, mask());
        return iterator(data(),_pts, mask());
    }
    iterator wr_find(int64_t _pts)
    {
        auto ri = read_index() + capacity();
        auto wi = write_index();
        if(_pts < wi || _pts > ri)
            iterator(data(),ri, mask());
        return iterator(data(),_pts, mask());
    }
    iterator begin()              { return iterator(data(),read_index(),mask());}
    const_iterator begin()  const { return const_iterator(data(),read_index(),mask());}
    const_iterator cbegin() const { return begin();}

    iterator end()                { return iterator(data(),write_index(),mask());}
    const_iterator end()    const { return const_iterator(data(),write_index(),mask());}
    const_iterator cend()   const { return end();}

    iterator wr_end()                { return iterator(data(),read_index() + capacity(),mask());}
    const_iterator wr_end()    const { return const_iterator(data(),read_index() + capacity(),mask());}
    const_iterator wr_cend()   const { return const_iterator(data(),read_index() + capacity(),mask());}


    reference front()             { return m_data[read_offset()];}
    const_reference front() const { return m_data[read_offset()];}

    reference back ()             { return m_data[last_offset()];}
    const_reference back () const { return m_data[last_offset()];}

    reference wr_front()             { return m_data[write_offset()];}
    const_reference wr_front() const { return m_data[write_offset()];}

    reference operator[](difference_type idx) { return begin()[idx]; }
    const_reference operator[](difference_type idx) const { return cbegin()[idx]; }
    const_reference at (difference_type idx) const
    {
        if(idx < 0 || size_type(idx) >= size())
            throw std::out_of_range();
        return cbegin()[idx];
    }
    reference wr_at(difference_type idx)
    {
        return end()[idx];
    }
    bool try_push_back(const_reference item)
    {
        if(!full()) { *end() = item; m_widx.fetch_add(1); return true;}
        else return false;
    }
    bool try_push_back(T&& item)
    {
        if(!full()) { *end() = std::forward<T>(item); m_widx.fetch_add(1); return true;}
        else return false;
    }
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
    size_type skip(size_type count)
    {
        count = std::min(count, size());
        m_ridx.fetch_add(count,std::memory_order_release);
        return count;
    }
    difference_type skip_to(difference_type idx)
    {
        auto ri = m_ridx.load(std::memory_order_acquire);
        auto wi = m_widx.load(std::memory_order_acquire);
        if(size_type(idx - ri) > size_type(idx - wi)) {
            idx = wi;
        }
        m_ridx.store(idx,std::memory_order_release);
        return idx;
    }
    size_type zero(size_type count)
    {
        count = std::min(count, size());
        auto zerod = size_type{};
        for(auto r : write_contig(count)) {
            if(r) {
                std::fill(r.cbegin(),r.cend(), T{});
                zerod += r.size();
                m_widx.fetch_add(r.size(),std::memory_order_release);
            }
        }
        return zerod;
    }
    void clear()
    {
        m_ridx.fetch_add(size(),std::memory_order_release);
    }
    void reset(difference_type _pts )
    {
        m_ridx.store(_pts);
        m_widx.store(_pts);
    }
    std::unique_ptr<RingBuffer<T> > resized(size_type _size) const
    {
        auto rr = read_range();
        _size = roundup(std::max<size_type>(_size, rr.size()));
        auto o = std::make_unique<RingBuffer<T> >(_size);
        o->reset(rr.cbegin().index());
        o->write(rr.cbegin(),rr.cend());
        return std::move(o);
    }

    size_type writeOne(const T& val)
    {
        if(full())
            return 0;
        push_back(val);
        return 1;
    }
    T readOne()
    {
        auto t = T{};
        if(!empty()) {
            t = std::move(front());
            pop_front();
        }
        return t;
    }
    T peekOne() const
    {
        auto t = T{};
        if(!empty()) t = front();
        return t;
    }
};
}
