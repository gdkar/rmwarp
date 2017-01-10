#pragma once

#include "Simd.hpp"
#include "Math.hpp"

namespace RMWarp {

struct RMSpectrum {
    using vector_type = align_vector<float>;
    using size_type = vector_type::size_type;
    using difference_type = vector_type::difference_type;

    int m_size;
    int m_coef{m_size / 2 + 1};
    int m_spacing{ (m_coef + 15) & ~15};
    vector_type X { spacing() * 2, align_alloc<float>{}};

    vector_type X_mag{spacing(), align_alloc<float>{}};
    vector_type X_phase{spacing(),align_alloc<float>{}};

    vector_type dM_dw  { spacing(), align_alloc<float>{}};
    vector_type dPhi_dw{ spacing(), align_alloc<float>{} };

    vector_type dM_dt  { spacing(), align_alloc<float>{} };
    vector_type dPhi_dt{ spacing(), align_alloc<float>{} };

    vector_type d2Phi_dtdw{spacing(), align_alloc<float>{} };

    RMSpectrum(int _size = 0) : m_size{_size} {}
    RMSpectrum(RMSpectrum && ) noexcept = default;
    RMSpectrum & operator = (RMSpectrum && ) noexcept = default;
    RMSpectrum(const RMSpectrum &) = default;
    RMSpectrum&operator=(const RMSpectrum &) = default;
    size_type size() const
    {
        return m_size;
    }
    size_type coefficients() const
    {
        return m_coef;
    }
    size_type spacing() const
    {
        m_spacing;
    }
    void resize(int _size)
    {
        if(_size == size())
            return;
        m_size = _size;
        m_coef = m_size / 2 + 1;
        m_spacing = (m_coef + 15) & ~15;
        X.resize(m_spacing * 2);
        X_mag.resize(m_spacing);
        X_phase.resize(m_spacing);
        for(auto p : { &dM_dw, &dPhi_dw, &dM_dt, &dPhi_dt, &d2Phi_dtdw} )
            p->resize(m_spacing);
    }
};
}
