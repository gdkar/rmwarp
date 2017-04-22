#include "Simd.hpp"
#include "Math.hpp"
#include "sysutils.hpp"
#include "ReSpectrum.hpp"

using namespace RMWarp;
void ReSpectrum::resize(int _size)
{
    if(_size == size())
        return;
    m_size = _size;
    m_coef = m_size / 2 + 1;
    m_spacing = align_up(m_coef, item_alignment);
    X.resize(m_spacing * 2);
    cexpr_for_each([sz=size_type(m_spacing)](auto & item){if(item.size() != sz) item.resize(sz);}
        , mag
        , M
        , Phi
        , dM_dw
        , dPhi_dw
        , dM_dt
        , dPhi_dt
        , d2Phi_dtdw
        , d2Phi_dtdw_weight
        , d2Phi_dtdw_acc
        , lgd
        , lgd_acc
        , lgd_weight
        , ltime);
}
void ReSpectrum::updateGroupDelay()
{
    auto _lgd     = local_group_delay();
    auto _lgda    = local_group_delay_acc();
    auto _lgdw    = local_group_delay_weight();
    auto _ltime   = local_time();

    auto _d2Phi   = d2Phi_dtdw_data();
    auto _d2Phia  = d2Phi_dtdw_acc_data();
    auto _mag     = mag_data();

    auto _dPhi_dw = dPhi_dw_data();

    auto fr = 0.9f;//bs::pow(10.0f, -40.0f/20.0f);
    auto ep = bs::sqrt(epsilon * fr * bs::rec(1-fr));

    bs::transform(_mag,_mag + m_coef, _lgdw, [ep](auto m) {
        auto _res = bs::is_less(m,decltype(m)(ep));
        return bs::if_zero_else_one(_res);
    });

    bs::transform(_lgdw,_lgdw + m_coef, _dPhi_dw, _lgda,bs::multiplies);
    bs::transform(_lgdw,_lgdw+ m_coef, _d2Phi, _d2Phia, bs::multiplies);

    std::partial_sum(_lgda,_lgda+m_coef,_lgda);
    std::partial_sum(_lgdw,_lgdw+m_coef,_lgdw);
    std::partial_sum(_d2Phia,_d2Phia, _d2Phia);

    auto i = 0;
    auto hi_bound = [](auto x){return (x * 1200)/1024;};
    auto lo_bound = [](auto x){return (x * 860 )/1024;};
    for(; i < m_coef - 8 && hi_bound(i) < i + 8; ++i) {
        auto hi = i + 8;
        auto lo = lo_bound(i);
        auto _w = -bs::rec(_lgdw[hi] - _lgdw[lo] + epsilon);
        _lgd[i] = (_lgda[hi] - _lgda[lo]) * _w;
        _d2Phi[i] = (_d2Phia[hi] - _d2Phia[lo]) * _w;

    }
    for(; hi_bound(i) < m_coef; ++i) {
        auto hi = hi_bound(i);
        auto lo = lo_bound(i);
        auto _w = -bs::rec(_lgdw[hi] - _lgdw[lo] + epsilon);
        _lgd[i] = (_lgda[hi] - _lgda[lo]) * _w;
        _d2Phi[i] = (_d2Phia[hi] - _d2Phia[lo])*_w;
    }
    {
        auto hi = m_coef - 1;
        auto hiw = _lgdw[hi];
        auto hid = _lgda[hi];
        auto hip = _d2Phia[hi];
        for(; i < m_coef; ++i) {
            auto lo = lo_bound(i);
            auto _w = -bs::rec(hiw - _lgdw[lo] + epsilon);
            _lgd[i]= (hid - _lgda[lo])*_w;
            _d2Phi[i] = (hip - _d2Phia[lo])*_w;

        }
    }
    bs::transform(_d2Phi,_d2Phi + m_coef, _d2Phia, [ep](auto m) {
        auto _res = bs::is_less(bs::abs(m+decltype(m)(1)),decltype(m)(0.25));
        return bs::if_one_else_zero(_res);
    });
    std::partial_sum(_d2Phia,_d2Phia, _d2Phia);
    bs::transform(_lgd,_lgd+m_coef, _ltime, [w=float(when())](auto x){return x + w;});
}
