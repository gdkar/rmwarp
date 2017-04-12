#pragma once

#include "Math.hpp"
#include "Simd.hpp"
#include "VectorOpsComplex.hpp"
#include "TimeWeightedWindow.hpp"
#include "TimeDerivativeWindow.hpp"
#include "TimeAlias.hpp"
#include "ReSpectrum.hpp"
#include "KaiserWindow.hpp"

namespace RMWarp {

struct FFTPlan {
    using value_type = fftwf_plan;
    using reference  = value_type&;
    using const_reference = const value_type&;
    typedef void (r2c_exec)(const fftwf_plan, float*ri,float*ro,float*io);
    typedef void (c2c_exec)(const fftwf_plan, float*ri,float*ii,float*ro,float*io);
    value_type  m_d{0};
    r2c_exec *m_r2c{};
    c2c_exec*m_c2c{};


    operator value_type() const { return m_d; }
    value_type get() const { return m_d; }
    bool operator !() const { return !m_d; }
    operator bool() const { return !!m_d; }
    value_type release() { return std::exchange(m_d, value_type{}); }
    void reset( value_type val = value_type{0})
    {
        if(m_d && m_d != val)
            fftwf_destroy_plan(m_d);
        m_d = val;
    }
    void reset( value_type val, r2c_exec * _r2c)
    {
        if(m_d && m_d != val)
            fftwf_destroy_plan(m_d);
        m_d = val;
        m_c2c = nullptr;
        m_r2c = val ? _r2c : nullptr;
    }
    void reset( value_type val, c2c_exec * _c2c)
    {
        if(m_d && m_d != val)
            fftwf_destroy_plan(m_d);
        m_d = val;
        m_r2c = nullptr;
        m_c2c = val ? _c2c : nullptr;
    }
    constexpr FFTPlan() = default;
    explicit constexpr FFTPlan(fftwf_plan _d, r2c_exec* fn) noexcept : m_d{_d}, m_r2c{fn} {}
    explicit constexpr FFTPlan(fftwf_plan _d, c2c_exec* fn) noexcept : m_d{_d}, m_c2c{fn} {}
    FFTPlan(FFTPlan && o) noexcept : m_d{o.release()} {}
    FFTPlan &operator=(FFTPlan && o) noexcept { swap(*this,o);return *this; }
   ~FFTPlan() { reset(); }
    friend void swap(FFTPlan &lhs, FFTPlan &rhs) noexcept
    {
        using std::swap;
        swap(lhs.m_d,rhs.m_d);
        swap(lhs.m_r2c,rhs.m_r2c);
        swap(lhs.m_c2c,rhs.m_c2c);
    }
    static FFTPlan dft_1d_r2c(int _n, float *_t, float *_r, float *_i)
    {
        auto dims = fftwf_iodim{ _n, 1, 1};
        return FFTPlan(fftwf_plan_guru_split_dft_r2c(
            1, &dims, 0, nullptr, _t, _r, _i, FFTW_ESTIMATE)
            , &fftwf_execute_split_dft_r2c);
    }
    static FFTPlan dft_1d_c2r(int _n, float *_r, float *_i, float *_t)
    {
        auto dims = fftwf_iodim{ _n, 1, 1};
        return FFTPlan(fftwf_plan_guru_split_dft_c2r(
            1, &dims, 0, nullptr, _r, _i, _t, FFTW_ESTIMATE)
            , &fftwf_execute_split_dft_c2r);
    }
    static FFTPlan dft_1d_c2c(int _n, float *_ri, float *_ii, float *_ro,float *_io)
    {
        auto dims = fftwf_iodim{ _n, 1, 1};
        return FFTPlan(fftwf_plan_guru_split_dft(
            1, &dims, 0, nullptr, _ri, _ii, _ro, _io,FFTW_ESTIMATE)
            , &fftwf_execute_split_dft);
    }
    void execute(float *_x0, float *_x1, float *_x2)
    {
        if(m_r2c) (*m_r2c) (m_d,  _x0, _x1, _x2);
    }
    void execute(float *_x0, float *_x1, float *_x2, float *_x3)
    {
        if(m_c2c) (*m_c2c) ( m_d, _x0, _x1, _x2,_x3);
    }
    void execute()
    {
        fftwf_execute(m_d);
    }
};
}
