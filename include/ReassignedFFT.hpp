#pragma once

#include "Math.hpp"
#include "Simd.hpp"
#include "FFT.hpp"
#include "TimeWeightedWindow.hpp"
#include "TimeDerivativeWindow.hpp"
#include "ReassignedSpectrum.hpp"
namespace RMWarp {

class RMFFT {
public:
    using vector_type = simd_vec<float>;
    using size_type = vector_type::size_type;
    using difference_type = vector_type::difference_type;
protected:
    int m_size{}; /// <- transform size
    int m_coef{m_size / 2 + 1}; /// <- number of non-redundant forier coefficients
    int m_spacing{ (m_coef + 15) & ~15}; /// <- padded number of coefficients to play nicely wiht simd when packing split-complex format into small contigous arrays.



    vector_type m_h  {size_type(size()), bs::allocator<float>{}};  /// <- transform window
    vector_type m_Dh {size_type(size()), bs::allocator<float>{}};  /// <- time derivative of window
    vector_type m_Th {size_type(size()), bs::allocator<float>{}};  /// <- time multiplied window
    vector_type m_TDh{size_type(size()), bs::allocator<float>{}};  /// <- time multiplied time derivative window;

    vector_type m_flat{size_type(size()), bs::allocator<float>{}}; /// <- input windowing scratch space.
    vector_type m_split{size_type(spacing()) * 2, bs::allocator<float>{}}; /// <- frequency domain scratch space.

    vector_type m_X    {size_type(spacing()) * 2, bs::allocator<float>{}}; /// <- transform of windowed signal
    vector_type m_X_Dh {size_type(spacing()) * 2, bs::allocator<float>{}}; /// <- transform of time derivative windowed signal
    vector_type m_X_Th {size_type(spacing()) * 2, bs::allocator<float>{}}; /// <- transform of time multiplied window
    vector_type m_X_TDh{size_type(spacing()) * 2, bs::allocator<float>{}}; /// <- transform of time multiplied time derivative window.

    fftwf_plan          m_plan_r2c{0};  /// <- real to complex plan;
    fftwf_plan          m_plan_c2r{0};  /// <0 complex to real plan.
    void _finish_process(RMSpectrum & dst, int64_t _when);
public:
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
    template<class It>
    void process( It src, RMSpectrum & dst, int64_t when = 0)
    {
        {
            auto do_window = [&](auto w, auto &v) {
                bs::transform(src, src + m_size, &w[0], &m_flat[0],bs::multiplies);
                fftwf_execute_split_dft_r2c(m_plan_r2c, &m_flat[0], &v[0], &v[m_spacing]);
            };
            do_window(m_h , m_X    );
            do_window(m_Dh, m_X_Dh );
            do_window(m_Th, m_X_Th );
            do_window(m_TDh,m_X_TDh);
        }
        _finish_process(dst,when);
    }
    int spacing() const;
    int size() const;
    int coefficients() const;
};
}
