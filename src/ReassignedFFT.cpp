#include "ReassignedFFT.hpp"

using namespace RMWarp;

RMFFT::RMFFT(int _size)
: m_size(_size)
{
        const auto dims = fftwf_iodim{ m_size, 1, 1};
        auto _real = &m_split[0]; auto _imag = &m_split[m_spacing]; auto _time = &m_flat[0];

        m_plan_r2c = fftwf_plan_guru_split_dft_r2c(
            1, &dims, 0, nullptr, _time, _real, _imag, FFTW_ESTIMATE);
        m_plan_c2r = fftwf_plan_guru_split_dft_c2r(
            1, &dims, 0, nullptr, _real, _imag, _time, FFTW_ESTIMATE);
}

RMFFT::~RMFFT()
{
    if(m_size) {
        fftwf_destroy_plan(m_plan_r2c);
        fftwf_destroy_plan(m_plan_c2r);
    }
}

void RMFFT::process( const float *const src, RMSpectrum & dst)
{
    using reg = simd_reg<float>;
    constexpr auto w = int(simd_width<float>);

    const auto _real = &m_X[0], _imag = &m_X[m_spacing]
        ,_real_Dh = &m_X_Dh[0], _imag_Dh = &m_X_Dh[m_spacing]
        ,_real_Th = &m_X_Th[0], _imag_Th = &m_X_Th[m_spacing]
        ,_real_TDh = &m_X_TDh[0], _imag_TDh = &m_X_TDh[m_spacing];
    {
        auto do_window = [&](auto w, auto d_r, auto d_i) {
            bs::transform(src, src + m_size, &w[0], &m_flat[0],bs::multiplies);
            fftwf_execute_split_dft_r2c(m_plan_r2c, &m_flat[0], d_r, d_i);
        };
        do_window(m_h , _real   , _imag);
        do_window(m_Dh, _real_Dh, _imag_Dh);
        do_window(m_Th, _real_Th, _imag_Th);
        do_window(m_TDh,_real_TDh,_imag_TDh);
    }
    auto _cmul = [](auto r0, auto i0, auto r1, auto i1) {
        return std::make_pair(r0 * r1 - i0 * i1, r0 * i1 + r1 * i0);
    };
    auto _cmul2 = [_cmul](auto a, auto b) {
        return _cmul(std::get<0>(a),std::get<1>(a),std::get<0>(b),std::get<1>(b));
    };
    auto _cinv = [](auto r, auto i) {
        auto n = bs::rec(bs::sqr(r) + bs::sqr(i) + bs::Eps<float>());
        return std::make_pair(r * n , -i * n);
    };
    dst.resize(m_size);
    std::copy(_real, _real + m_coef, dst.X.begin());
    std::copy(_imag, _imag + m_coef, dst.X.begin() + m_spacing);

    for(auto i = 0; i < m_coef; i += w ) {
        auto _X_r = reg(_real + i), _X_i = reg(_imag + i);

        bs::store(bs::hypot(_X_i,_X_r), &dst.X_mag[0] + i);
        bs::store(bs::atan2(_X_i,_X_r), &dst.X_phase[0] + i);

        std::tie(_X_r, _X_i) = _cinv(_X_r,_X_i);

        auto _Dh_over_X = _cmul( reg(_real_Dh + i),reg(_imag_Dh + i) ,_X_r, _X_i );

        bs::store(std::get<0>(_Dh_over_X),  &dst.dM_dt  [0] + i);
        bs::store(std::get<1>(_Dh_over_X), &dst.dPhi_dt[0] + i);

        auto _Th_over_X = _cmul( reg(_real_Th + i),reg(_imag_Th + i) ,_X_r, _X_i );

        bs::store(-std::get<1>(_Th_over_X), &dst.dM_dw  [0] + i);
        bs::store( std::get<0>(_Th_over_X), &dst.dPhi_dw[0] + i);

        auto _TDh_over_X = reg(_real_TDh + i) * _X_r - reg(_imag_TDh + i) * _X_i;
        auto _Th_Dh_over_XX = std::get<0>(_Th_over_X) * std::get<0>(_Dh_over_X)
                            - std::get<1>(_Th_over_X) * std::get<1>(_Dh_over_X);
        bs::store(_TDh_over_X - _Th_Dh_over_XX,&dst.d2Phi_dtdw[0] + i);
    }
}
RMFFT::size_type RMFFT::spacing() const
{
    return m_spacing;
}
RMFFT::size_type RMFFT::size() const
{
    return m_size;
}
RMFFT::size_type RMFFT::coefficients() const
{
    return m_coef;
}
