#pragma once

#include <fftw3.h>
#include "Math.hpp"
#include "Simd.hpp"
#include "TimeAlias.hpp"
#include "ReSpectrum.hpp"
#include "KaiserWindow.hpp"

namespace RMWarp {

struct FFTPlan {
    using value_type        = fftwf_plan;
    using reference         = value_type&;
    using const_reference   = const value_type&;
    using difference_type   = std::ptrdiff_t;
    using size_type         = std::size_t;
    using float_type        = float;

    typedef void (r2c_exec)(const fftwf_plan, float_type*ri,float_type*ro,float_type*io);
    typedef void (c2c_exec)(const fftwf_plan, float_type*ri,float_type*ii,float_type*ro,float_type*io);
    value_type  m_d{};
    union {
        r2c_exec *m_r2c{};
        c2c_exec *m_c2c;
    };
    difference_type m_off_in {0};
    difference_type m_off_out{0};


    explicit operator value_type() const { return m_d;   }
    value_type get() const               { return m_d;   }
    bool operator !() const              { return !m_d;  }
    operator bool() const                { return !!m_d; }
    value_type release() noexcept
    {
        auto res = std::exchange(m_d, value_type{});
        m_r2c = nullptr;
        m_off_in = m_off_out = 0;
        return res;
    }
    void reset()
    {
        if(m_d)
            fftwf_destroy_plan(m_d);
        m_d = 0;
        m_off_in = 0;
        m_off_out = 0;
        m_r2c = nullptr;
    }
    constexpr FFTPlan() = default;
//    : m_d{0}, m_r2c{nullptr}, m_off_in{0},m_off_out{0}
//    { }
    constexpr FFTPlan(fftwf_plan _d, r2c_exec* fn, difference_type off_in, difference_type off_out) noexcept
    : m_d{_d}
    , m_r2c{fn}
    , m_off_in{off_in}
    , m_off_out{off_out}
    {
    }
    constexpr FFTPlan(fftwf_plan _d, c2c_exec* fn, difference_type off_in, difference_type off_out   ) noexcept
    : m_d{_d}
    , m_c2c{fn}
    , m_off_in{off_in}
    , m_off_out{off_out}
    {
    }
    FFTPlan(FFTPlan && o) noexcept
    : FFTPlan{}
    {
        swap(o);
    }
    FFTPlan &operator=(FFTPlan && o) noexcept
    {
        swap(o);
        return *this;
    }
   ~FFTPlan()
    {
        reset();
    }
    void swap(FFTPlan &rhs) noexcept
    {
        using std::swap;
        swap(m_d,       rhs.m_d);
        swap(m_r2c,     rhs.m_r2c);
        swap(m_off_in,  rhs.m_off_in);
        swap(m_off_out, rhs.m_off_out);
    }
    static FFTPlan dft_1d_r2c(int _n, float_type *_ti, float_type *_ro, float_type *_io)
    {
        auto dims = fftwf_iodim{ _n, 1, 1};
        auto off_out = _io - _ro;

        if(off_out == 1 || off_out == -1)
            dims.os = 2;

        auto _d = fftwf_plan_guru_split_dft_r2c(
            1, &dims, 0, nullptr, _ti, _ro, _ro + off_out, FFTW_ESTIMATE);
        return { _d, &fftwf_execute_split_dft_r2c, 0, off_out};
    }
    static FFTPlan dft_1d_c2r(int _n, float_type *_ri, float_type *_ii, float_type *_to)
    {
        auto dims = fftwf_iodim{ _n, 1, 1};
        auto off_in = _ii - _ri;
        if(off_in == 1 || off_in == -1)
            dims.is = 2;

        auto _d = fftwf_plan_guru_split_dft_c2r(
            1, &dims, 0, nullptr, _ri, _ri + off_in, _to, FFTW_ESTIMATE);
        return {_d, &fftwf_execute_split_dft_c2r, off_in, 0};
    }
    static FFTPlan dft_1d_c2c(int _n, float_type *_ri, float_type *_ii, float_type *_ro,float_type *_io)
    {
        auto dims = fftwf_iodim{ _n, 1, 1};
        auto off_in = _ii - _ri;
        auto off_out= _io - _ro;

        if(off_in == 1 || off_in == -1)
            dims.is = 2;
        if(off_out == 1 || off_out == -1)
            dims.os = 2;

        auto flags = FFTW_ESTIMATE;
        auto _d = fftwf_plan_guru_split_dft(
            1, &dims, 0, nullptr, _ri, _ri + off_in, _ro, _ro + off_out,flags);
        return { _d, &fftwf_execute_split_dft, off_in, off_out};
    }
    void execute(float_type *_in, float_type *_out)
    {
        if(!m_d)
//            return;
            throw std::runtime_error("cannot execute unplanned FFTs.");
        if(!_in || !_out)
//            return;
            throw std::runtime_error("you gonna segfault, dumbass!");
        if(m_off_out && !m_off_in) {
            (*m_r2c)(
                m_d
                , _in
                , _out
                , _out + m_off_out
            );
        }else if(m_off_in && !m_off_out) {
            (*m_r2c)(
                m_d
                , _in
                , _in + m_off_in
                , _out
            );
        } else {
            (*m_c2c) (
                m_d
                , _in
                , _in + m_off_in
                , _out
                , _out + m_off_out
            );
        }
    }
    void execute()
    {
        if(m_d)
            fftwf_execute(m_d);
    }
};
}
