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
        , d2M_dtdw
        , epsilon_weight
        , d2Phi_dtdw_acc
        , d2M_dtdw_acc
        , lgd
        , lgd_acc
        , ltime);
}
void ReSpectrum::updateGroupDelay()
{
    auto _ew      = weight_data();

    auto _lgd     = local_group_delay();
    auto _lgda    = local_group_delay_acc();
    auto _ltime   = local_time();

    auto _d2Phi   = d2Phi_dtdw_data();
    auto _d2Phia  = d2Phi_dtdw_acc_data();
    auto _d2M   = d2M_dtdw_data();
    auto _d2Ma  = d2M_dtdw_acc_data();

    auto _mag     = mag_data();

    auto _dPhi_dw = dPhi_dw_data();

    auto fr = 0.9f;//bs::pow(10.0f, -40.0f/20.0f);
    auto ep = bs::sqrt(bs::sqr(epsilon) * fr * bs::rec(1-fr));

    bs::transform(_mag,_mag + m_coef, _ew, [ep](auto m) {
        auto _res = bs::is_less(m,decltype(m)(ep));
        return bs::if_zero_else_one(_res);
    });
//    bs::transform(_dPhi_dw, _dPhi_dw + m_coef,  _lgd,     [](auto && x){return -x;});
    std::copy(_dPhi_dw,     _dPhi_dw + m_coef,  _lgd);
    bs::transform(_ew,      _ew + m_coef,       _dPhi_dw, _lgda,    bs::multiplies);
    bs::transform(_ew,      _ew + m_coef,       _d2Phi,   _d2Phia,  bs::multiplies);
    bs::transform(_ew,      _ew + m_coef,       _d2M,     _d2Ma,    bs::multiplies);

    std::partial_sum(_ew ,      _ew + m_coef,   _ew);
    std::partial_sum(_lgda,     _lgda+m_coef,   _lgda);
    std::partial_sum(_d2Phia,   _d2Phia+m_coef, _d2Phia);
    std::partial_sum(_d2Ma,     _d2Ma+m_coef,   _d2Ma);

    bs::transform(_lgd,         _lgd+m_coef,    _ltime, [w=float(when())](auto x){return x + w;});
}
