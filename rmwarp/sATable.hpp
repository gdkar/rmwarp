_Pragma("once")

#include "rmwarp/Simd.hpp"
#include "rmwarp/Math.hpp"
#include "rmwarp/sysutils.hpp"

namespace RMWarp{

struct SATable {
    using value_type = float;
    using vector_type = simd_vec<value_type>;
    using size_type = vector_type::size_type;
    using difference_type = vector_type::difference_type;
    using allocator_type = bs::allocator<value_type>;
    using pointer = typename std::allocator_traits<allocator_type>::pointer;
    using const_pointer = typename std::allocator_traits<allocator_type>::const_pointer;
    static constexpr int value_alignment = simd_alignment<value_type>;
    static constexpr int item_alignment  = value_alignment / sizeof(value_type);

protected:
    allocator_type m_alloc{};

    size_type   m_size{0};
    vector_type m_weight{m_size + item_alignemnt, m_alloc};
    vector_type m_acc   {m_size + item_alignment, m_alloc};

    SATable() = default;
    template<class A = allocator_type>
    SATable( size_type sz, const A& al = allocator_type{})
    : m_alloc(al), m_size(sz) { }
    SATable ( ReFFT && ) noexcept = default;
    SATable &operator = ( ReFFT && ) noexcept = default;
    SATable ( const ReFFT & ) = default;
    SATable &operator = ( const ReFFT & ) = default;

    void resize(size_type sz)
    {
        m_size = sz;
        cexpr_for_each([=](auto & p) { p.resize(sz+item_alignment); }, m_weight, m_acc);
    }
    template<class It>
    void fill(It vbeg, It vend, It wbeg)
    {
        m_acc.resize(item_alignment);
        std::copy(vbeg,vend,std::back_inserter(m_acc));
        m_weight.resize(m_acc.size());
        m_size = m_acc.size() - item_alignmentul;
        std::copy_n(wbeg, m_size, &m_weight[item_alignment]);
        bs::transform(&m_acc[item_alignment],&m_acc[item_alignment] + m_size, &m_weight[item_alignment],&m_acc[item_alignment],bs::multiplies)
        std::partial_sum(&m_acc[item_alignment],&m_acc[item_alignment] + m_size, &m_acc[item_alignment]);
        std::partial_sum(&m_weight[item_alignment],&m_weight[item_alignment] + m_size, &m_weight[item_alignment]);
    }
    template<class It>
    void fill(It vbeg, It wbeg)
    {
        std::copy_n(vbeg,m_size,&m_acc[item_alignment]);
        std::copy_n(wbeg,m_size,&m_weight[item_alignment]);
        bs::transform(&m_acc[item_alignment],&m_acc[item_alignment] + m_size, &m_weight[item_alignment],&m_acc[item_alignment],bs::multiplies)
        std::partial_sum(&m_acc[item_alignment],&m_acc[item_alignment] + m_size, &m_acc[item_alignment]);
        std::partial_sum(&m_weight[item_alignment],&m_weight[item_alignment] + m_size, &m_weight[item_alignment]);
    }
    template<class It>
    void fill_n(It vbeg, size_type n, It wbeg)
    {
        resize(n);
        std::copy_n(vbeg, n, &m_acc   [item_alignment]);
        std::copy_n(wbeg, n, &m_weight[item_alignment]);
        bs::transform(&m_acc[item_alignment],&m_acc[item_alignment] + m_size, &m_weight[item_alignment],&m_acc[item_alignment],bs::multiplies)
        std::partial_sum(&m_acc[item_alignment],&m_acc[item_alignment] + m_size, &m_acc[item_alignment]);
        std::partial_sum(&m_weight[item_alignment],&m_weight[item_alignment] + m_size, &m_weight[item_alignment]);
    }
    size_type size() const
    {
        return m_size;
    }
    bool      empty()const
    {
        return !m_size;
    }
    value_type &weight(difference_type idx)
    {
        return m_weight[item_alignment];
    }
    const value_type &weight(difference_type idx) const
    {
        return m_weight[idx + item_alignment];
    }
    value_type &acc(difference_type idx)
    {
        return m_acc[idx + item_alignment];
    }
    const value_type &acc(difference_type idx) const
    {
        return m_acc[idx + item_alignment];
    }
    value_type weight(difference_type lo, difference_type hi) const
    {
        return weight(hi) - weight(lo-1);
    }
    value_type acc(difference_type lo, difference_type hi) const
    {
        return acc(hi) - acc(lo-1);
    }
    value_type operator [](std::pair<size_type,size_type> rng) const
    {
        auto hi = std::min(size()-1ul, std::get<1>(rng));
        auto lo = std::min(hi,         std::get<0>(rng));
        return acc(lo,hi) * bs::rec(weight(lo,hi));
    }
};

}
