#pragma once

#include "Math.hpp"
#include "Simd.hpp"
#include "VectorOpsComplex.hpp"
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

    float m_epsilon = std::numeric_limits<float>::epsilon();

    allocator_type m_alloc{};
    vector_type m_h  {size_type(size()), m_alloc};  /// <- transform window
    vector_type m_Dh {size_type(size()), m_alloc};  /// <- time derivative of window
    vector_type m_Th {size_type(size()), m_alloc};  /// <- time multiplied window
    vector_type m_TDh{size_type(size()), m_alloc};  /// <- time multiplied time derivative window;

    vector_type m_flat{size_type(size()), m_alloc}; /// <- input windowing scratch space.
    vector_type m_split{size_type(spacing()) * 2, m_alloc}; /// <- frequency domain scratch space.

    vector_type m_X    {size_type(spacing()) * 2, m_alloc}; /// <- transform of windowed signal
    vector_type m_X_Dh {size_type(spacing()) * 2, m_alloc}; /// <- transform of time derivative windowed signal
    vector_type m_X_Th {size_type(spacing()) * 2, m_alloc}; /// <- transform of time multiplied window
    vector_type m_X_TDh{size_type(spacing()) * 2, m_alloc}; /// <- transform of time multiplied time derivative window.

    fftwf_plan          m_plan_r2c{0};  /// <- real to complex plan;
    fftwf_plan          m_plan_c2r{0};  /// <0 complex to real plan.
    void _finish_process(ReSpectrum & dst, int64_t _when);
    template<class A = allocator_type>
    ReFFT(const A &al = allocator_type{}) : m_alloc{al} {}
    ReFFT ( ReFFT && ) noexcept ;
    ReFFT &operator = ( ReFFT && ) noexcept ;
    void swap(ReFFT & o) noexcept;
    void initPlans();
    template<class A = allocator_type>
    ReFFT ( int _size, const A& al = allocator_type{})
    : m_size{_size}, m_alloc(al){ if(_size) initPlans(); }
    template<class It>
    It setWindow(It wbegin, It wend)
    {
        auto wn = std::distance(wbegin,wend);
        if(wn < m_size) {
        std::fill(m_h.begin(),m_h.begin() + (m_size-wn)/2,0.f);
            std::fill(std::copy(wbegin,wend, m_h.begin() + (m_size-wn)/2),m_h.end(),0.f);
        }else{
            std::copy_n(wbegin,size(), m_h.begin());
            wbegin += wn;
        }
        time_derivative_window(m_h.cbegin(),m_h.cend(),m_Dh.begin());
        time_weighted_window(m_h.cbegin(),m_h.cend(),m_Th.begin());
        time_weighted_window(m_Dh.cbegin(),m_Dh.cend(),m_TDh.begin());
        return wbegin;
    }
    template<class It>
    void setWindow(It wbegin, It wend, It dt_begin, It dt_end)
    {
        {
            auto wn = std::distance(wbegin,wend);
            if(wn < m_size) {
            std::fill(m_h.begin(),m_h.begin() + (m_size-wn)/2,0.f);
            std::fill(std::copy(wbegin,wend, m_h.begin() + (m_size-wn)/2),m_h.end(),0.f);
            }else{
                std::copy_n(wbegin,size(), m_h.begin());
            }
        }
        {
            auto dt_n = std::distance(dt_begin,dt_end);
            if(dt_n < m_size) {
                std::fill(m_Dh.begin(),m_Dh.begin() + (m_size-dt_n)/2,0.f);
                std::fill(std::copy(dt_begin,dt_end, m_Dh.begin() + (m_size-dt_n)/2),m_Dh.end(),0.f);
            }else{
                std::copy_n(dt_begin, size(),m_Dh.begin());
            }
        }
        time_weighted_window(m_h.cbegin(),m_h.cend(),m_Th.begin());
        time_weighted_window(m_Dh.cbegin(),m_Dh.cend(),m_TDh.begin());
    }
    template<class It, class A = allocator_type>
    ReFFT( It wbegin, It wend, const A & al = allocator_type{})
    :ReFFT(int(std::distance(wbegin,wend)), al)
    {
        setWindow(wbegin,wend);
    }
    template<class It, class A = allocator_type>
    ReFFT( It wbegin, It wend, It dt_begin, It dt_end, const A & al = allocator_type{})
    :ReFFT(int(std::distance(wbegin,wend)), al)
    {
        setWindow(wbegin,wend,dt_begin,dt_end);
    }
    static ReFFT Kaiser(int _size, float alpha);
   ~ReFFT();
    void updateGroupDelay(ReSpectrum &dst);

    template<class It>
    void process( It src, It send, ReSpectrum & dst, int64_t when = 0, bool do_group_delay = false)
    {
        {
            auto do_window = [&](auto &w, auto &v) {
                cutShift(&m_flat[0], src,send, w);
                fftwf_execute_split_dft_r2c(m_plan_r2c, &m_flat[0], &v[0], &v[m_spacing]);
            };
            do_window(m_h , m_X    );
            do_window(m_Dh, m_X_Dh );
            do_window(m_Th, m_X_Th );
            do_window(m_TDh,m_X_TDh);
        }
        _finish_process(dst,when);
        if(do_group_delay)
            updateGroupDelay(dst);
    }
    template<class It>
    void process( It src, ReSpectrum & dst, int64_t when = 0, bool do_group_delay=false)
    {
        {
            auto do_window = [&](auto &w, auto &v) {
                cutShift(&m_flat[0], src, w);
                fftwf_execute_split_dft_r2c(m_plan_r2c, &m_flat[0], &v[0], &v[m_spacing]);
            };
            do_window(m_h , m_X    );
            do_window(m_Dh, m_X_Dh );
            do_window(m_Th, m_X_Th );
            do_window(m_TDh,m_X_TDh);
        }
        _finish_process(dst,when);
        if(do_group_delay)
            updateGroupDelay(dst);
    }
    template<class It, class iIt>
    void inverse( It dst, iIt _M, iIt _Phi)
    {
        auto norm = 0.5f * bs::rec(float(size()));
        bs::transform(
            _M
           ,_M + m_coef
           , &m_X[0]
           , [norm](auto x){
                return norm * bs::exp(x);
            });
        std::copy(
            _Phi
           ,_Phi+ m_coef
           , &m_X[0] + spacing()
            );
        v_polar_to_cartesian(
              &m_split[0]
            , &m_split[0] + spacing()
            , &m_X[0]
            , &m_X[0] + spacing()
            , m_coef);
        fftwf_execute_split_dft_c2r(
            m_plan_c2r
          , &m_split[0]
          , &m_split[0] + spacing()
          , &m_flat[0]);
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
        fftwf_execute_split_dft_c2r(
            m_plan_c2r
          , &m_split[0]
          , &m_split[0] + spacing()
          , &m_flat[0]);
        std::copy(&m_flat[0],&m_flat[m_size], dst);
    }

    int spacing() const;
    int size() const;
    int coefficients() const;
};
}
