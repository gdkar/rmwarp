#pragma once

#include "Simd.hpp"
#include "Math.hpp"

namespace RMWarp {

class RMSpectrum {
    int m_size;
    int m_coef{m_size / 2 + 1};
    int m_spacing{ (m_coef + 15) & ~15};
    int64_t m_when{0};
public:
    using vector_type = simd_vec<float>;
    using size_type = vector_type::size_type;
    using difference_type = vector_type::difference_type;

    vector_type X {  size_type(spacing()) * 2, bs::allocator<float>{}};
    vector_type X_log{size_type(spacing()) * 2 , bs::allocator<float>{}};

    vector_type dM_dt  { size_type(spacing()), bs::allocator<float>{} };
    vector_type dPhi_dt{ size_type(spacing()), bs::allocator<float>{} };

    vector_type dM_dw  { size_type(spacing()), bs::allocator<float>{}};
    vector_type dPhi_dw{ size_type(spacing()), bs::allocator<float>{} };

    vector_type d2Phi_dtdw{size_type(spacing()), bs::allocator<float>{} };

    RMSpectrum(int _size = 0) : m_size{_size} {}
    RMSpectrum(RMSpectrum && ) noexcept = default;
    RMSpectrum & operator = (RMSpectrum && ) noexcept = default;
    RMSpectrum(const RMSpectrum &) = default;
    RMSpectrum&operator=(const RMSpectrum &) = default;

    float * X_real() { return &X[0];}
    float * X_imag() { return &X[spacing()];}
    const float * X_real() const { return &X[0];}
    const float * X_imag() const { return &X[spacing()];}

    float * M_data() { return &X_log[0];}
    float * Phi_data() { return &X_log[spacing()];}
    const float * M_data() const { return &X_log[0];}
    const float * Phi_data() const { return &X_log[spacing()];}

    float * dM_dt_data() { return &dM_dt[0];}
    float * dPhi_dt_data() { return &dPhi_dt[0];}
    const float * dM_dt_data() const { return &dM_dt[0];}
    const float * dPhi_dt_data() const { return &dPhi_dt[0];}

    float * dM_dw_data() { return &dM_dw[0];}
    float * dPhi_dw_data() { return &dPhi_dw[0];}
    const float * dM_dw_data() const { return &dM_dw[0];}
    const float * dPhi_dw_data() const { return &dPhi_dw[0];}

    float * d2Phi_dtdw_data() { return &d2Phi_dtdw[0];}
    const float * d2Phi_dtdw_data() const { return &d2Phi_dtdw[0];}

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
        m_spacing;
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
        m_spacing = (m_coef + 15) & ~15;
        X.resize(m_spacing * 2);
        X_log.resize(m_spacing * 2);
        for(auto p : { &dM_dw, &dPhi_dw, &dM_dt, &dPhi_dt, &d2Phi_dtdw} )
            p->resize(m_spacing);
    }
    void reset(int _size, int64_t _when)
    {
        resize(_size);
        set_when(_when);
    }
};
}
