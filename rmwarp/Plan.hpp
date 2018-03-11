#pragma once

#include <fftw3.h>
#include <type_traits>
#include "Math.hpp"
#include "Simd.hpp"
#include "TimeAlias.hpp"
#include "ReSpectrum.hpp"
#include "KaiserWindow.hpp"

namespace RMWarp {

struct FFTPlan {
    using plan_type         = fftwf_plan;
    using const_plan        = const fftwf_plan;
    using float_type        = float;
    using pointer           = float_type *;
    using difference_type   = typename std::pointer_traits<pointer>::difference_type;
    using size_type         = typename std::make_unsigned<difference_type>::type;

    typedef void (r2c_exec)(const_plan, pointer ri,pointer ro,pointer io);
    typedef void (c2c_exec)(const_plan, pointer ri,pointer ii,pointer ro,pointer io);

    plan_type  m_d{0};
    union {
        r2c_exec *m_r2c{};
        c2c_exec *m_c2c;
    };
    difference_type m_off_in {0};
    difference_type m_off_out{0};

    explicit operator plan_type() const
    {
        return m_d;
    }
    plan_type get() const
    {
        return m_d;
    }
    bool operator !() const
    {
        return !m_d;
    }
    explicit operator bool() const
    {
        return !!m_d;
    }
    plan_type release() noexcept;
    void reset();
    constexpr FFTPlan() = default;
    FFTPlan &operator=(FFTPlan && o) noexcept
    {
        swap(o);
        return *this;
    }
    FFTPlan(FFTPlan && o) noexcept
    : FFTPlan{}
    {
        swap(o);
    }
    constexpr FFTPlan(fftwf_plan _d, r2c_exec* fn, difference_type off_in, difference_type off_out) noexcept
    : m_d{_d}
    , m_r2c{_d ? fn : nullptr}
    , m_off_in{_d ? off_in : 0}
    , m_off_out{_d ? off_out : 0}
    { }
    constexpr FFTPlan(fftwf_plan _d, c2c_exec* fn, difference_type off_in, difference_type off_out   ) noexcept
    : m_d{_d}
    , m_c2c{_d?fn : nullptr}
    , m_off_in{_d? off_in : 0}
    , m_off_out{_d? off_out : 0}
    { }
   ~FFTPlan()
    {
        if(m_d)
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
    static FFTPlan dft_1d_r2c(int _n, pointer _ti, pointer _ro, pointer _io);
    static FFTPlan dft_1d_c2r(int _n, pointer _ri, pointer _ii, pointer _to);
    static FFTPlan dft_1d_c2c(int _n, pointer _ri, pointer _ii, pointer _ro,pointer _io);
    void execute(pointer _in, pointer _out) const;
    void execute() const;
};
}
