#include "Simd.hpp"
#include "Math.hpp"
#include "sysutils.hpp"
#include "ReSpectrum.hpp"

using namespace RMWarp;
void ReSpectrum::resize(int _size)
{
    if(_size == size())
        return;
    m_size    = _size;
    m_coef    = m_size / 2 + 1;
    m_spacing = align_up(m_coef, item_alignment);
    X.resize(m_spacing * 2);
    cexpr_for_each(
          [sz=size_type(m_spacing)](auto & item){
              if(item.size() != sz)
                item.resize(sz);
          }
        , mag
        , M
        , Phi
        , dM_dw
        , dPhi_dw
        , dM_dt
        , dPhi_dt
        , d2Phi_dtdw
        , d2M_dtdw
        );
}
void ReSpectrum::unwrapFrom(const ReSpectrum &o)
{
    if(!size() || (o.size() != size()))
        return;

    using reg = simd_reg<float>;
    using std::tie; using std::make_pair; using std::copy; using std::get;
    constexpr auto w = int(simd_width<float>);
    auto i = 0;

    const auto _dp0 = dPhi_dt_data();
    const auto _dp1 = o.dPhi_dt_data();
    const auto _p0  = Phi_data();
    const auto _p1  = o.Phi_data();

    const auto _base_hop   = (when() - o.when());
    const auto _base_unit = 2 * bs::Pi<value_type>() / size();

    for(; i < m_coef; i += w) {
        auto _unit = bs::enumerate<reg>(i) * _base_unit;
        auto _incr = (_unit + (reg(_dp0 + i) + reg(_dp1 + i)) * 0.5f) * _base_hop;
        auto _next = reg(_p1 + i) + _incr;
        _next = bs::if_zero_else(bs::is_nan(_next),_next);
        auto _prev = reg(_p0 + i);
        auto _gen  = _next + princarg(_prev-_next);
        bs::store(bs::if_zero_else(bs::is_nan(_gen),_gen),_p0 + i);
    }
    for(; i < m_coef; ++i) {
        auto _unit = _base_unit * i;
        auto _incr = (_unit + (*(_dp0 + i) + *(_dp1 + i)) * 0.5f) * _base_hop;
        auto _next = *(_p1 + i) + _incr;
        _next = bs::if_zero_else(bs::is_nan(_next),_next);
        auto _prev = *(_p0 + i);
        auto _gen  = _next + princarg(_prev-_next);
        bs::store(bs::if_zero_else(bs::is_nan(_gen),_gen),_p0 + i);
    }
}
