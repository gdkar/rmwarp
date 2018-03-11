#pragma once

#include "Plan.hpp"
#include "sysutils.hpp"
#include "Simd.hpp"
#include "TimeWeightedWindow.hpp"
#include "TimeDerivativeWindow.hpp"
#include "TimeAlias.hpp"
#include "ReSpectrum.hpp"

namespace RMWarp {

struct ReFFT {
    using value_type = float;
    using vector_type = simd_vec<value_type>;
    using size_type = vector_type::size_type;
    using difference_type = vector_type::difference_type;
    using allocator_type = bs::allocator<value_type>;
    using pointer = typename std::allocator_traits<allocator_type>::pointer;
    using const_pointer = typename std::allocator_traits<allocator_type>::const_pointer;
    static constexpr int value_alignment = simd_alignment<value_type>;
    static constexpr int item_alignment  = value_alignment / sizeof(value_type);

    int m_size{}; /// <- transform size
    int m_coef{m_size ? (m_size / 2 + 1) : 0};
    int m_spacing{align_up(m_coef, item_alignment)};

    value_type m_epsilon = std::pow(10.0f, -80.0f/20.0f);//std::numeric_limits<float>::epsilon();

    allocator_type m_alloc{};

    value_type          m_time_width{};
    value_type          m_freq_width{};
    vector_type m_h     {size_type(size()), m_alloc};  /// <- transform window
    vector_type m_Dh    {size_type(size()), m_alloc};  /// <- time derivative of window
    vector_type m_Th    {size_type(size()), m_alloc};  /// <- time multiplied window
    vector_type m_TDh   {size_type(size()), m_alloc};  /// <- time multiplied time derivative window;

    vector_type m_flat  {size_type(size()), m_alloc}; /// <- input windowing scratch space.
    vector_type m_split {size_type(spacing()) * 2, m_alloc}; /// <- frequency domain scratch space.

    vector_type m_X     {size_type(spacing()) * 2, m_alloc}; /// <- transform of windowed signal
    vector_type m_X_Dh  {size_type(spacing()) * 2, m_alloc}; /// <- transform of time derivative windowed signal
    vector_type m_X_Th  {size_type(spacing()) * 2, m_alloc}; /// <- transform of time multiplied window
    vector_type m_X_TDh {size_type(spacing()) * 2, m_alloc}; /// <- transform of time multiplied time derivative window.

    FFTPlan             m_plan_r2c{};
    FFTPlan             m_plan_c2r{};
    void _finish_set_window();
    void _finish_process(float *src, ReSpectrum & dst, int64_t _when);

    template<class A = allocator_type>
    ReFFT(const A &al = allocator_type{})
    : m_alloc{al} {}

    ReFFT ( ReFFT && ) noexcept = default;
    ReFFT &operator = ( ReFFT && ) noexcept = default;

    void initPlans();
    template<class A = allocator_type>
    ReFFT ( int _size, const A& al = allocator_type{})
    : m_size{_size}, m_alloc(al){ if(_size) initPlans(); }

    template<class It>
    It setWindow(It wbegin, It wend)
    {
        auto wn = std::distance(wbegin,wend);
        if(wn < m_size) {
            auto off = (m_size - wn)/2;
            std::fill(m_h.begin(),m_h.begin() + off,0.f);
            std::fill(std::copy(wbegin,wend, m_h.begin() + off),m_h.end(),0.f);
            wbegin += wn;
        }else if(wn > m_size) {
            auto off = (wn - m_size)/2;
            wbegin += off;
            std::copy_n(wbegin,size(), m_h.begin());
            wbegin += size() + off;
        } else {
            std::copy(wbegin,wend,m_h.begin());
            wbegin += size();
        }
        _finish_set_window();
        return wbegin;
    }
    template<class It, class A = allocator_type>
    ReFFT( It wbegin, It wend, const A & al = allocator_type{})
    : ReFFT(int(std::distance(wbegin,wend)), al)
    {
        setWindow(wbegin,wend);
    }
    static ReFFT Kaiser(int _size, float alpha);
    virtual ~ReFFT();

    template<class It>
    void process( It src, It send, ReSpectrum & dst, int64_t when = 0)
    {
        auto tsrc = static_cast<float*>(alloca(
            m_size * sizeof(float))),
             tsend = tsrc + m_size;
        bs::fill(std::copy(src,send,tsrc),tsend,0.0f);
        _finish_process(tsrc,dst,when);
    }
    template<class It>
    void process( It src, ReSpectrum & dst, int64_t when = 0)
    {
        process(src,std::next(src,m_size),dst,when);
    }
    void _process_inverse();
    template<class It, class iIt>
    void inverse( It dst, iIt _M, iIt _Phi)
    {
        const auto _coef    = m_coef;
        std::copy_n(_M, _coef, &m_split[0]);
        std::copy_n(_Phi,_coef,&m_split[0] + m_spacing);

        _process_inverse();
        std::rotate_copy(
            m_flat.cbegin()
           ,m_flat.cbegin() + (m_size/2)
           ,m_flat.cend()
           ,dst
            );
    }
    template<class I, class O>
    void inverseCepstral(O dst, I src)
    {
        bs::transform(src, src + m_coef, &m_split[0], bs::log);
        std::fill_n(&m_split[spacing()], m_coef, 0.0f);
        m_plan_c2r.execute(&m_split[0], &m_flat[0]);
        std::copy(&m_flat[0],&m_flat[m_size], dst);
    }
    int spacing() const;
    int size() const;
    int coefficients() const;
    const_pointer   h_data() const;
    const_pointer   Dh_data() const;
    const_pointer   Th_data() const;
    const_pointer   TDh_data() const;
    value_type      time_width() const;
    value_type      freq_width() const;
};
}
