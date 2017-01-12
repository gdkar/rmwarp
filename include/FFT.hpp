#pragma once

#include "Math.hpp"
#include "Simd.hpp"

#include <fftw3.h>

namespace RMWarp {

class RFFT {
public:
    using vector_type = simd_vec<float>;
    using size_type = vector_type::size_type;
    using difference_type = vector_type::difference_type;
protected:
    int m_size;
    int m_coef{m_size/2 + 1};
    vector_type m_flat{ size_type(size()), align_alloc<float>{} };
    vector_type m_real{ size_type(coefficients()), align_alloc<float>{} };
    vector_type m_imag{ size_type(coefficients()), align_alloc<float>{} };
    fftwf_plan  m_plan_r2c;
    fftwf_plan  m_plan_c2r;
public:
    RFFT(int _size = 0);
   ~RFFT();
    RFFT(RFFT && ) noexcept = default;
    RFFT &operator=( RFFT && ) noexcept = default;
    int size() const;
    int coefficients() const;
    template<class I, class O>
    void forward(O rdst,O idst, I src)
    {
        std::copy_n(src, m_size, m_flat.begin());
        fftwf_execute(m_plan_r2c);
        std::copy(m_real.cbegin(),m_real.cend(),rdst);
        std::copy(m_imag.cbegin(),m_imag.cend(),idst);
    }
    template<class I, class O>
    void backward(O dst, I rsrc, I isrc)
    {
        std::copy_n(rsrc,m_size, &m_real);
        std::copy_n(isrc,m_size, &m_imag);
        fftwf_execute(m_plan_c2r);
        std::copy(m_flat.cbegin(),m_flat.cend(),dst);
    }
    template<class I, class O>
    void inverse(O dst, I rsrc, I isrc)
    {
        std::copy_n(rsrc,m_size, &m_real);
        std::copy_n(isrc,m_size, &m_imag);
        fftwf_execute(m_plan_c2r);
        auto scale = 1/float(m_size);
        std::transform(m_flat.cbegin(),m_flat.cend(),dst,[scale](auto x){return x * scale;});
    }
};
class FFT {
public:
    using vector_type = simd_vec<float>;
    using size_type = vector_type::size_type;
    using difference_type = vector_type::difference_type;
protected:
    int m_size {};
    vector_type m_rsrc{ size_type(size()), align_alloc<float>{}};
    vector_type m_isrc{ size_type(size()), align_alloc<float>{}};
    vector_type m_rdst{ size_type(size()), align_alloc<float>{}};
    vector_type m_idst{ size_type(size()), align_alloc<float>{}};
    fftwf_plan          m_plan;
public:
    FFT(int _size = 0);
   ~FFT();
    FFT(FFT && ) noexcept = default;
    FFT &operator=( FFT && ) noexcept = default;
    int size() const;
    int coefficients() const;
    template<class I, class O>
    void forward(O rdst, O idst, I rsrc, I isrc)
    {
        std::copy_n(rsrc,m_size, m_rsrc.begin());
        std::copy_n(isrc,m_size, m_isrc.begin());
        fftwf_execute(m_plan);
        std::copy_n(m_rdst.cbegin(), m_size, rdst);
        std::copy_n(m_idst.cbegin(), m_size, idst);
    }
    template<class I, class O>
    void backward(O rdst, O idst, I rsrc, I isrc)
    {
        forward(idst, rdst, isrc, rsrc);
    }
    template<class I, class O>
    void inverse(O rdst, O idst, I rsrc, I isrc)
    {
        std::copy_n(isrc,m_size, m_rsrc.begin());
        std::copy_n(rsrc,m_size, m_isrc.begin());
        fftwf_execute(m_plan);
        auto scale = 1/float(m_size);
        std::transform(m_rdst.cbegin(),m_rdst.cend(),idst,[scale](auto x){return x * scale;});
        std::transform(m_idst.cbegin(),m_idst.cend(),rdst,[scale](auto x){return x * scale;});
    }
};
}
