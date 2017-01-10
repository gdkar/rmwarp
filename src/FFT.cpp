#include "FFT.hpp"

using namespace RMWarp;

RFFT::size_type RFFT::size() const
{
    return m_size;
}
RFFT::size_type RFFT::coefficients() const
{
    return m_coef;
}

RFFT::~RFFT()
{
    fftwf_destroy_plan(m_plan_r2c);
    fftwf_destroy_plan(m_plan_c2r);
}
RFFT::RFFT(int _size )
: m_size{_size}
{
    const auto dims = fftwf_iodim{ _size, 1, 1};
    m_plan_r2c = fftwf_plan_guru_split_dft_r2c(
        1, &dims
        , 0, nullptr
        , &m_flat[0]
        , &m_real[0], &m_imag[0]
        , FFTW_ESTIMATE);
    m_plan_c2r = fftwf_plan_guru_split_dft_c2r(
        1, &dims
        , 0, nullptr
        , &m_real[0], &m_imag[0]
        , &m_flat[0]
        , FFTW_ESTIMATE);
}
FFT::FFT(int _size)
: m_size{_size}
{
    const auto dims = fftwf_iodim{ _size, 1, 1};
    m_plan = fftwf_plan_guru_split_dft(
        1, &dims
        , 0, nullptr
        , &m_rsrc[0], &m_isrc[0]
        , &m_rdst[0], &m_idst[0]
        , FFTW_ESTIMATE);
}
FFT::~FFT()
{
    fftwf_destroy_plan(m_plan);
}
FFT::size_type FFT::size() const
{
    return m_size;
}
FFT::size_type FFT::coefficients() const
{
    return m_size;
}
