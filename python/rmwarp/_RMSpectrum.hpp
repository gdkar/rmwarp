#pragma once

#include "system/Simd.hpp"
#include "system/Math.h"

namespace RubberBand {

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
    vector_type X_mag{size_type(spacing()), bs::allocator<float>{}};

    vector_type dM_dt  { size_type(spacing()), bs::allocator<float>{} };
    vector_type dPhi_dt{ size_type(spacing()), bs::allocator<float>{} };

    vector_type dM_dw  { size_type(spacing()), bs::allocator<float>{}};
    vector_type dPhi_dw{ size_type(spacing()), bs::allocator<float>{} };

    vector_type       lgd       { size_type(spacing()), bs::allocator<float>{} };
    vector_type       lgd_weight{ size_type(spacing()), bs::allocator<float>{} };
    simd_vec<int64_t> rm_when{ size_type(spacing()), bs::allocator<float>{} };

//    vector_type d2Phi_dtdw{size_type(spacing()), bs::allocator<float>{} };

    RMSpectrum(int _size = 0) : m_size(_size) {}
    RMSpectrum(RMSpectrum && ) noexcept = default;
    RMSpectrum & operator = (RMSpectrum && ) noexcept = default;
    RMSpectrum(const RMSpectrum &) = default;
    RMSpectrum&operator=(const RMSpectrum &) = default;

    float * X_real() { return &X[0];}
    float * X_imag() { return &X[spacing()];}
    const float * X_real() const { return &X[0];}
    const float * X_imag() const { return &X[spacing()];}

    float * mag_data() { return &X_mag[0];}
    float * M_data() { return &X_log[0];}
    float * Phi_data() { return &X_log[spacing()];}

    const float * mag_data() const { return &X_mag[0];}
    const float * M_data() const { return &X_log[0];}
    const float * Phi_data() const { return &X_log[spacing()];}

    float * local_group_delay() { return &lgd[0];}
    float * local_group_delay_weight() { return &lgd_weight[0];}
    int64_t * reassigned_time() { return &rm_when[0];}

    const float * local_group_delay() const { return &lgd[0];}
    const float * local_group_delay_weight() const { return &lgd_weight[0];}
    const int64_t * reassigned_time() const { return &rm_when[0];}

    float * dM_dt_data() { return &dM_dt[0];}
    float * dPhi_dt_data() { return &dPhi_dt[0];}
    const float * dM_dt_data() const { return &dM_dt[0];}
    const float * dPhi_dt_data() const { return &dPhi_dt[0];}

    float * dM_dw_data() { return &dM_dw[0];}
    float * dPhi_dw_data() { return &dPhi_dw[0];}
    const float * dM_dw_data() const { return &dM_dw[0];}
    const float * dPhi_dw_data() const { return &dPhi_dw[0];}

//    float * d2Phi_dtdw_data() { return &d2Phi_dtdw[0];}
//    const float * d2Phi_dtdw_data() const { return &d2Phi_dtdw[0];}

//    float & d2Phi_dtdw_at(size_type i) { return d2Phi_dtdw[i];}
//    float   d2Phi_dtdw_at(size_type i) const { return d2Phi_dtdw[i];}

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
        m_spacing = (m_coef + 15) & ~15;
        X.resize(m_spacing * 2);
        X_log.resize(m_spacing * 2);
        for(auto p : { &X_mag,&dM_dw, &dPhi_dw, &dM_dt, &dPhi_dt,
            &lgd, &lgd_weight
//            ,&d2Phi_dtdw, &impulse_position, &impulse_quality}
            })
            p->resize(m_spacing);
        rm_when.resize(m_spacing);
    }
    void reset(int _size, int64_t _when)
    {
        resize(_size);
        set_when(_when);
    }
};
}
