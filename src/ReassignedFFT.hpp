#pragma once

#include "Math.hpp"
#include "Simd.hpp"
#include "FFT.hpp"
#include "TimeWeightedWindow.hpp"
#include "TimeDerivativeWindow.hpp"
#include "ReassignedSpectrum.hpp"
namespace RMWarp {

struct RMFFT {
    using vector_type = align_vector<float>;
    using size_type = vector_type::size_type;
    using difference_type = vector_type::difference_type;

    int m_size{}; /// <- transform size
    int m_coef{m_size / 2 + 1}; /// <- number of non-redundant forier coefficients
    int m_spacing{ (m_coef + 15) & ~15}; /// <- padded number of coefficients to play nicely wiht simd when packing split-complex format into small contigous arrays.

    vector_type m_h  {size(), align_alloc<float>{}};  /// <- transform window
    vector_type m_Dh {size(), align_alloc<float>{}};  /// <- time derivative of window
    vector_type m_Th {size(), align_alloc<float>{}};  /// <- time multiplied window
    vector_type m_TDh{size(), align_alloc<float>{}};  /// <- time multiplied time derivative window;

    vector_type m_flat{size(), align_alloc<float>{}}; /// <- input windowing scratch space.
    vector_type m_split{spacing() * 2, align_alloc<float>{}}; /// <- frequency domain scratch space.

    vector_type m_X    {spacing() * 2, align_alloc<float>{}}; /// <- transform of windowed signal
    vector_type m_X_Dh {spacing() * 2, align_alloc<float>{}}; /// <- transform of time derivative windowed signal
    vector_type m_X_Th {spacing() * 2, align_alloc<float>{}}; /// <- transform of time multiplied window
    vector_type m_X_TDh{spacing() * 2, align_alloc<float>{}}; /// <- transform of time multiplied time derivative window.

    fftwf_plan          m_plan_r2c{0};  /// <- real to complex plan;
    fftwf_plan          m_plan_c2r{0};  /// <0 complex to real plan.
    RMFFT() = default;
    RMFFT ( RMFFT && ) noexcept = default;
    RMFFT &operator = ( RMFFT && ) noexcept = default;
    RMFFT ( int _size );
    template<class It>
    It setWindow(It wbegin, It wend)
    {
        auto wn = std::distance(wbegin,wend);
        if(wn < m_size) {
            std::fill(std::copy(wbegin,wend,m_h.begin()),m_h.end(), 0.f);
        }else{
            std::copy_n(wbegin, m_h.begin());
        }
        time_derivative_window(m_h.cbegin(),m_h.cend(),m_Dh.begin());
        time_weighted_window(m_h.cbegin(),m_h.cend(),m_Th.begin());
        time_weighted_window(m_Dh.cbegin(),m_Dh.cend(),m_TDh.begin());
    }
    template<class It>
    RMFFT ( int _size, It _W)
    : RMFFT(_size)
    {
        setWindow(_W, std::next(_W, _size));
    }
    template<class It>
    RMFFT( It wbegin, It wend)
    :RMFFT(std::distance(wbegin,wend))
    {
        setWindow(wbegin,wend);
    }
   ~RMFFT();
    void process( const float *const src, RMSpectrum & dst);
    size_type spacing() const;
    size_type size() const;
    size_type coefficients() const;
};
}
