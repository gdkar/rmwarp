#pragma once

#include "Simd.hpp"
#include "Math.hpp"

namespace RMWarp {

class ReSpectrum {
public:
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
    int m_size;
    int m_coef{m_size / 2 + 1};
    int m_spacing{align_up(m_coef, item_alignment)};
    int64_t m_when{0};
public:

    allocator_type m_alloc{};

    vector_type X {  size_type(spacing()) * 2, m_alloc};
    vector_type M {  size_type(spacing()), m_alloc};
    vector_type Phi{ size_type(spacing()), m_alloc};
    vector_type mag{ size_type(spacing()), m_alloc};

    vector_type dM_dt  { size_type(spacing()), m_alloc };
    vector_type dPhi_dt{ size_type(spacing()), m_alloc };

    vector_type dM_dw  { size_type(spacing()), m_alloc};
    vector_type dPhi_dw{ size_type(spacing()), m_alloc };
    vector_type d2Phi_dtdw{size_type(spacing()), m_alloc };

    vector_type       lgd       { size_type(spacing()), m_alloc };
    vector_type       lgd_weight{ size_type(spacing()), m_alloc };
    vector_type       lgd_acc   { size_type(spacing()), m_alloc };
    vector_type       ltime     { size_type(spacing()), m_alloc };

    template<class A = allocator_type>
    ReSpectrum(int _size = 0, const A &al = allocator_type{}) : m_size(_size), m_alloc{al} {}
    ReSpectrum(ReSpectrum && ) noexcept = default;
    ReSpectrum & operator = (ReSpectrum && ) noexcept = default;
    ReSpectrum(const ReSpectrum &) = default;
    ReSpectrum&operator=(const ReSpectrum &) = default;

    pointer X_real() { return &X[0];}
    const_pointer X_real() const { return &X[0];}

    pointer X_imag() { return &X[spacing()];}
    const_pointer X_imag() const { return &X[spacing()];}

    pointer mag_data() { return &mag[0];}
    const_pointer mag_data() const { return &mag[0];}

    pointer M_data() { return &M[0];}
    const_pointer M_data() const { return &M[0];}

    pointer Phi_data() { return &Phi[0];}
    const_pointer Phi_data() const { return &Phi[0];}

    pointer dM_dt_data() { return &dM_dt[0];}
    const_pointer dM_dt_data() const { return &dM_dt[0];}

    pointer dM_dw_data() { return &dM_dw[0];}
    const_pointer dM_dw_data() const { return &dM_dw[0];}

    pointer dPhi_dt_data() { return &dPhi_dt[0];}
    const_pointer dPhi_dt_data() const { return &dPhi_dt[0];}

    pointer dPhi_dw_data() { return &dPhi_dw[0];}
    const_pointer dPhi_dw_data() const { return &dPhi_dw[0];}

    pointer d2Phi_dtdw_data() { return &d2Phi_dtdw[0];}
    const_pointer d2Phi_dtdw_data() const { return &d2Phi_dtdw[0];}

    pointer local_group_delay() { return &lgd[0];}
    pointer local_group_delay_weight() { return &lgd_weight[0];}
    pointer local_group_delay_acc      () { return &lgd_acc[0];}
    pointer local_time() { return &ltime[0];}

    const_pointer local_group_delay() const { return &lgd[0];}
    const_pointer local_group_delay_weight() const { return &lgd_weight[0];}
    const_pointer local_group_delay_acc() const { return &lgd_acc[0];}
    const_pointer local_time() const { return &ltime[0];}

    int size() const
    {
        return m_size;
    }
    int coefficients() const
    {
        return m_coef;
    }
    int spacing() const
    {
        return m_spacing;
    }
    int64_t when() const
    {
        return m_when;
    }
    void set_when(int64_t _when)
    {
        m_when = _when;
    }
    void resize(int _size)
    {
        if(_size == size())
            return;
        m_size = _size;
        m_coef = m_size / 2 + 1;
        m_spacing = align_up(m_coef, item_alignment);
        X.resize(m_spacing * 2);
        cexpr_for_each([sz=m_spacing](auto & item){item.resize(sz);}
            , mag
            , M
            , Phi
            , dM_dw
            , dPhi_dw
            , dM_dt
            , dPhi_dt
            , d2Phi_dtdw
            , lgd
            , lgd_acc
            , lgd_weight
            , ltime);
//            ,&d2Phi_dtdw, &impulse_position, &impulse_quality}
//            })
//            p->resize(m_spacing);
//        rm_when.resize(m_spacing);
    }
    void reset(int _size, int64_t _when)
    {
        resize(_size);
        set_when(_when);
    }
};
}
