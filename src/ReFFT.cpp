#include <thread>
#include <mutex>
#include "ReFFT.hpp"
#include "KaiserWindow.hpp"
#include "TimeAlias.hpp"
using namespace RMWarp;
void ReFFT::initPlans()
{
    if(m_size) {
        m_coef = m_size ? (m_size / 2 + 1) : 0; /// <- number of non-redundant forier coefficients
        m_spacing = align_up(m_coef, item_alignment);
        cexpr_for_each([sz=size_type(m_size)](auto & item){item.resize(sz);}
            , m_h
            , m_Dh
            , m_Th
            , m_TDh
            , m_flat
              );
        cexpr_for_each([sz=size_type(spacing() * 2)](auto & item){item.resize(sz);}
            , m_split
            , m_X
            , m_X_Dh
            , m_X_Th
            , m_X_TDh
              );
        auto _real = &m_split[0]; auto _imag = &m_split[m_spacing]; auto _time = &m_flat[0];
        m_plan_r2c = FFTPlan::dft_1d_r2c(m_size, _time, _real, _imag);
        m_plan_c2r = FFTPlan::dft_1d_c2r(m_size, _real, _imag, _time);
    }
    _finish_set_window();
}
/*static*/ ReFFT ReFFT::Kaiser(int _size, float alpha)
{
    auto win    = vector_type(_size, 0.0f);
    make_kaiser_window(win.begin(),win.end(), alpha);
    return ReFFT(win.cbegin(),win.cend());
}

ReFFT::~ReFFT() = default;

void ReFFT::_finish_set_window()
{
    if(!m_size){
        m_time_width = 0.0f;
        m_freq_width = 0.0f;
    } else {
        using std::reverse;
        using std::copy;
        using std::rotate;

//        fftshift(m_h.begin(),m_h.end());
//        fftshift(m_h.begin(),m_h.end(),m_Th.begin());
//        copy(m_h.cbegin(),m_h.cend(),m_Th.begin());
        time_weighted_window(m_h.cbegin(),m_h.cend(),m_Th.begin());
//        ifftshift(m_Th.begin(),m_Th.end());

//        fftshift(m_h.begin(),m_h.end(),m_Dh.begin());
//        copy(m_h.cbegin(),m_h.cend(),m_Dh.begin());
        time_derivative_window(m_h.cbegin(),m_h.cend(),m_Dh.begin());
//        ifftshift(m_Dh.begin(),m_Dh.end());

//        fftshift(m_Dh.begin(),m_Dh.end(),m_TDh.begin());
//        copy(m_Dh.cbegin(),m_Dh.cend(),m_TDh.begin());
        time_weighted_window(m_Dh.cbegin(),m_Dh.cend(),m_TDh.begin());
//        ifftshift(m_TDh.begin(),m_TDh.end());

        auto norm_       = bs::transform_reduce(&m_h[0],  &m_h[0] + m_size,  bs::sqr, value_type{}, bs::plus);
        auto var_t_unorm = bs::transform_reduce(&m_Th[0], &m_Th[0] + m_size, bs::sqr, value_type{}, bs::plus);
        auto var_w_unorm = bs::transform_reduce(&m_Dh[0], &m_Dh[0] + m_size, bs::sqr, value_type{}, bs::plus);

        m_time_width = 2 * bs::sqrt(bs::Pi<value_type>() * var_t_unorm) * bs::rsqrt(norm_);
        m_freq_width = 2 * bs::sqrt(bs::Pi<value_type>() * var_w_unorm) * bs::rsqrt(norm_);

        reverse(m_h.begin(),m_h.end());
        reverse(m_Th.begin(),m_Th.end());
        reverse(m_Dh.begin(),m_Dh.end());
        reverse(m_TDh.begin(),m_TDh.end());
//        fftshift(m_h.begin(),m_h.end());
//        fftshift(m_Th.begin(),m_Th.end());
//        fftshift(m_Dh.begin(),m_Dh.end());
//        fftshift(m_TDh.begin(),m_TDh.end());
    }

}
void ReFFT::_finish_process(float *sbeg , ReSpectrum & dst, int64_t _when )
{
    {
//        auto send = sbeg + m_size;
        auto do_window = [&](auto &w, auto &v) {
            cutShift(sbeg, &m_flat[0], &w[0],&w[0] + w.size());
            m_plan_r2c.execute(&m_flat[0], &v[0]);
        };
        do_window(m_h , m_X    );
        do_window(m_Dh, m_X_Dh );
        do_window(m_Th, m_X_Th );
        do_window(m_TDh,m_X_TDh);
    }

    using reg = simd_reg<float>;
    using std::tie; using std::make_pair; using std::copy; using std::get;

    constexpr auto w = int(simd_width<float>);
    auto i = 0;

    const auto _real = &m_X[0],     _imag       = &m_X[m_spacing]
        ,_real_Dh = &m_X_Dh[0],     _imag_Dh    = &m_X_Dh[m_spacing]
        ,_real_Th = &m_X_Th[0],     _imag_Th    = &m_X_Th[m_spacing]
        ,_real_TDh = &m_X_TDh[0],   _imag_TDh   = &m_X_TDh[m_spacing]
        ;
    constexpr auto _cmul = [](auto r0, auto i0, auto r1, auto i1) {
        return make_pair(r0 * r1 - i0 * i1, r0 * i1 + r1 * i0);
        };
    constexpr auto _pcmul = [&](auto x0, auto x1) {
        return _cmul(get<0>(x0),get<1>(x0),
                     get<0>(x1),get<1>(x1));
    };

    dst.reset(m_size, _when);
    {
        const auto _dst_X_real = dst.X_real()
                  ,_dst_X_imag = dst.X_imag()
                  ,_dst_mag_data = dst.mag_data()
                  ,_dst_M_data = dst.M_data()
                  ,_dst_Phi_data = dst.Phi_data();
        for(; i < m_coef; i += w ) {
            auto _X_r = reg(_real + i), _X_i = reg(_imag + i);
            bs::store(_X_r, _dst_X_real + i);
            bs::store(_X_i, _dst_X_imag + i);
            {
                auto _X_mag = bs::hypot(_X_i,_X_r);
                auto _X_phi = bs::atan2(_X_i,_X_r);
                bs::store(_X_mag, _dst_mag_data + i);
                bs::store(bs::log(_X_mag), _dst_M_data + i);
                bs::store(bs::if_else_zero(bs::is_not_nan(_X_phi),_X_phi), _dst_Phi_data + i);
            }
        }
        for(; i < m_coef; ++i) {
            auto _X_r = *(_real + i), _X_i = *(_imag + i);
            bs::store(_X_r, _dst_X_real + i);
            bs::store(_X_i, _dst_X_imag + i);
            {
                auto _X_mag = bs::hypot(_X_i,_X_r);
                auto _X_phi = bs::atan2(_X_i,_X_r);
                bs::store(_X_mag, _dst_mag_data + i);
                bs::store(bs::log(_X_mag), _dst_M_data + i);
                bs::store(bs::if_else_zero(bs::is_not_nan(_X_phi),_X_phi), _dst_Phi_data + i);
            }
        }
    }
    if(m_epsilon > 0.0f) {
        auto max_mag = bs::max_val(dst.mag_data(),dst.mag_data() + m_coef);
        if(!max_mag)
            max_mag = std::numeric_limits<float>::epsilon();
        dst.epsilon = max_mag * m_epsilon;
    } else {
        dst.epsilon = 0.0f;
    }
    auto _cinv = [e=bs::sqr(dst.epsilon)](auto r, auto i) {
        auto n = bs::sqr(r) + bs::sqr(i);
        auto m = bs::if_zero_else(bs::is_less_equal(n,e), bs::rec(n));
        return make_pair(r * m , -i * m);
    };
    i = 0;
    {
        const auto   _dst_dM_dt         = &dst.dM_dt[0]
                    ,_dst_dPhi_dt       = &dst.dPhi_dt[0]
                    ,_dst_dM_dw         = &dst.dM_dw[0]
                    ,_dst_dPhi_dw       = &dst.dPhi_dw[0]
                    ,_dst_d2Phi_dtdw    = &dst.d2Phi_dtdw[0]
                    ,_dst_d2M_dtdw      = &dst.d2M_dtdw[0];

        for(; i < m_coef; i += w ) {
            auto _X_r = reg(_real + i), _X_i = reg(_imag + i);
            tie(_X_r, _X_i) = _cinv(_X_r,_X_i);

            auto _Dh_over_X = _cmul( reg(_real_Dh + i),reg(_imag_Dh + i) ,_X_r, _X_i );

            bs::store(get<0>(_Dh_over_X), _dst_dM_dt   + i);
            bs::store(get<1>(_Dh_over_X), _dst_dPhi_dt + i);

            auto _Th_over_X = _cmul( reg(_real_Th + i),reg(_imag_Th + i) ,_X_r, _X_i );

            bs::store(-get<1>(_Th_over_X), _dst_dM_dw   + i);
            bs::store( get<0>(_Th_over_X), _dst_dPhi_dw + i);

            auto _TDh_over_X    = (_cmul( reg(_real_TDh + i),reg(_imag_TDh + i) ,_X_r, _X_i ));
            auto _Th_Dh_over_X2 = (_pcmul(_Th_over_X,_Dh_over_X));

            bs::store(-std::get<1>(_TDh_over_X) + std::get<1>(_Th_Dh_over_X2), _dst_d2M_dtdw + i);
            bs::store( std::get<0>(_TDh_over_X) - std::get<0>(_Th_Dh_over_X2), _dst_d2Phi_dtdw + i);
        }
        for(; i < m_coef; ++i) {
            auto _X_r = *(_real + i), _X_i = *(_imag + i);
            tie(_X_r, _X_i) = _cinv(_X_r,_X_i);

            auto _Dh_over_X = _cmul( *(_real_Dh + i),*(_imag_Dh + i) ,_X_r, _X_i );

            bs::store(get<0>(_Dh_over_X), _dst_dM_dt   + i);
            bs::store(get<1>(_Dh_over_X), _dst_dPhi_dt + i);

            auto _Th_over_X = _cmul( *(_real_Th + i),*(_imag_Th + i) ,_X_r, _X_i );

            bs::store(-get<1>(_Th_over_X), _dst_dM_dw   + i);
            bs::store( get<0>(_Th_over_X), _dst_dPhi_dw + i);

            auto _TDh_over_X    = _cmul( *(_real_TDh + i),*(_imag_TDh + i) ,_X_r, _X_i );
            auto _Th_Dh_over_X2 = _pcmul(_Th_over_X,_Dh_over_X);

            bs::store(-std::get<1>(_TDh_over_X) + std::get<1>(_Th_Dh_over_X2), _dst_d2M_dtdw + i);
            bs::store( std::get<0>(_TDh_over_X) - std::get<0>(_Th_Dh_over_X2), _dst_d2Phi_dtdw + i);
        }
    }
}
int ReFFT::spacing() const
{
    return m_spacing;
}
ReFFT::const_pointer ReFFT::h_data() const
{
    return &m_h[0];
}
ReFFT::const_pointer ReFFT::Dh_data() const
{
    return &m_Dh[0];
}
ReFFT::const_pointer ReFFT::Th_data() const
{
    return &m_Th[0];
}
ReFFT::const_pointer ReFFT::TDh_data() const
{
    return &m_TDh[0];
}

int ReFFT::size() const
{
    return m_size;
}
int ReFFT::coefficients() const
{
    return m_coef;
}
ReFFT::value_type ReFFT::time_width() const
{
    return m_time_width;
}
ReFFT::value_type ReFFT::freq_width() const
{
    return m_freq_width;
}
void ReFFT::_process_inverse()
{

    const auto norm = bs::rec(float(size()));
    auto _coef = m_coef;
    constexpr auto w = int(simd_width<float>);
    using reg = simd_reg<float>;
    {
        auto i =0;
        const auto _split_r = &m_split[0];
        const auto _split_i = _split_r + m_spacing;

        for(; i < _coef; i += w) {
            auto _mag = norm * bs::exp(reg(_split_r + i));
            auto _i_r = bs::sincos(reg(_split_i+ i));
            bs::store(std::get<0>(_i_r) * _mag, _split_i + i);
            bs::store(std::get<1>(_i_r) * _mag, _split_r + i);
        }
        for(; i < _coef; i += 1) {
            auto _mag = norm * bs::exp((_split_r[i]));
            auto _i_r = bs::sincos(_split_i[i]);
            bs::store(std::get<0>(_i_r) * _mag, _split_i + i);
            bs::store(std::get<1>(_i_r) * _mag, _split_r + i);
        }
    }
    m_plan_c2r.execute(&m_split[0], &m_flat[0]);
}

