#include "Simd.hpp"
#include "Math.hpp"
#include "sysutils.hpp"
#include "ReSpectrum.hpp"

using namespace RMWarp;


ReSpectrum::pointer ReSpectrum::X_real() { return &X[0];}
ReSpectrum::const_pointer ReSpectrum::X_real() const { return &X[0];}

ReSpectrum::pointer ReSpectrum::X_imag() { return &X[spacing()];}
ReSpectrum::const_pointer ReSpectrum::X_imag() const { return &X[spacing()];}

ReSpectrum::pointer ReSpectrum::mag_data() { return &mag[0];}
ReSpectrum::const_pointer ReSpectrum::mag_data() const { return &mag[0];}

ReSpectrum::pointer ReSpectrum::M_data() { return &M[0];}
ReSpectrum::const_pointer ReSpectrum::M_data() const { return &M[0];}

ReSpectrum::pointer ReSpectrum::Phi_data() { return &Phi[0];}
ReSpectrum::const_pointer ReSpectrum::Phi_data() const { return &Phi[0];}

ReSpectrum::pointer ReSpectrum::dM_dt_data() { return &dM_dt[0];}
ReSpectrum::const_pointer ReSpectrum::dM_dt_data() const { return &dM_dt[0];}

ReSpectrum::pointer ReSpectrum::dM_dw_data() { return &dM_dw[0];}
ReSpectrum::const_pointer ReSpectrum::dM_dw_data() const { return &dM_dw[0];}

ReSpectrum::pointer ReSpectrum::dPhi_dt_data() { return &dPhi_dt[0];}
ReSpectrum::const_pointer ReSpectrum::dPhi_dt_data() const { return &dPhi_dt[0];}

ReSpectrum::pointer ReSpectrum::dPhi_dw_data() { return &dPhi_dw[0];}
ReSpectrum::const_pointer ReSpectrum::dPhi_dw_data() const { return &dPhi_dw[0];}

ReSpectrum::pointer ReSpectrum::d2Phi_dtdw_data() { return &d2Phi_dtdw[0];}
ReSpectrum::const_pointer ReSpectrum::d2Phi_dtdw_data() const { return &d2Phi_dtdw[0];}
ReSpectrum::pointer ReSpectrum::local_group_delay() { return &lgd[0];}
ReSpectrum::pointer ReSpectrum::local_group_delay_weight() { return &lgd_weight[0];}
ReSpectrum::pointer ReSpectrum::local_group_delay_acc      () { return &lgd_acc[0];}
ReSpectrum::pointer ReSpectrum::local_time() { return &ltime[0];}

ReSpectrum::const_pointer ReSpectrum::local_group_delay() const { return &lgd[0];}
ReSpectrum::const_pointer ReSpectrum::local_group_delay_weight() const { return &lgd_weight[0];}
ReSpectrum::const_pointer ReSpectrum::local_group_delay_acc() const { return &lgd_acc[0];}
ReSpectrum::const_pointer ReSpectrum::local_time() const { return &ltime[0];}


ReSpectrum::range_type ReSpectrum::X_real_range()
{
    return { X_real(), X_real() + coefficients()};}
ReSpectrum::const_range_type ReSpectrum::X_real_range() const
{
    return { X_real(), X_real() + coefficients()};
}

ReSpectrum::range_type ReSpectrum::X_imag_range()
{
    return { X_imag(), X_imag() + coefficients()};}
ReSpectrum::const_range_type ReSpectrum::X_imag_range() const
{
    return { X_imag(), X_imag() + coefficients()};}

ReSpectrum::range_type ReSpectrum::mag_range()
{
    return { mag_data(), mag_data() + coefficients()};
}
ReSpectrum::const_range_type ReSpectrum::mag_range() const
{
    return { mag_data(), mag_data() + coefficients()};
}

ReSpectrum::range_type ReSpectrum::M_range()
{
    return { M_data(), M_data() + coefficients()};
}
ReSpectrum::const_range_type ReSpectrum::M_range() const
{
    return { M_data(), M_data() + coefficients()};
}

ReSpectrum::range_type ReSpectrum::Phi_range()
{
    return { Phi_data(), Phi_data() + coefficients()};
}
ReSpectrum::const_range_type ReSpectrum::Phi_range() const
{
    return { Phi_data(), Phi_data() + coefficients()};
}

ReSpectrum::range_type ReSpectrum::dM_dt_range()
{
    return { dM_dt_data(), dM_dt_data() + coefficients()};
}
ReSpectrum::const_range_type ReSpectrum::dM_dt_range() const
{
    return { dM_dt_data(), dM_dt_data() + coefficients()};
}

ReSpectrum::range_type ReSpectrum::dM_dw_range()
{
    return { dM_dw_data(), dM_dw_data() + coefficients()};
}
ReSpectrum::const_range_type ReSpectrum::dM_dw_range() const
{
    return { dM_dw_data(), dM_dw_data() + coefficients()};
}

ReSpectrum::range_type ReSpectrum::dPhi_dt_range()
{
    return { dPhi_dt_data(), dPhi_dt_data() + coefficients()};
}
ReSpectrum::const_range_type ReSpectrum::dPhi_dt_range() const
{
    return { dPhi_dt_data(), dPhi_dt_data() + coefficients()};
}

ReSpectrum::range_type ReSpectrum::dPhi_dw_range()
{
    return { dPhi_dw_data(), dPhi_dw_data() + coefficients()};
}
ReSpectrum::const_range_type ReSpectrum::dPhi_dw_range() const
{
    return { dPhi_dw_data(), dPhi_dw_data() + coefficients()};
}

ReSpectrum::range_type ReSpectrum::d2Phi_dtdw_range()
{
    return { d2Phi_dtdw_data(), d2Phi_dtdw_data() + coefficients()};
}
ReSpectrum::const_range_type ReSpectrum::d2Phi_dtdw_range() const
{
    return { d2Phi_dtdw_data(), d2Phi_dtdw_data() + coefficients()};
}
ReSpectrum::range_type ReSpectrum::local_group_delay_range()
{
    return { local_group_delay(), local_group_delay() + coefficients()};
}
ReSpectrum::range_type ReSpectrum::local_group_delay_weight_range()
{
    return { local_group_delay_weight(), local_group_delay_weight() + coefficients()};
}
ReSpectrum::range_type ReSpectrum::local_group_delay_acc_range()
{
    return { local_group_delay_acc(), local_group_delay_acc() + coefficients()};
}

ReSpectrum::range_type ReSpectrum::local_time_range()
{
    return { local_time(), local_time() + coefficients()};
}

ReSpectrum::const_range_type ReSpectrum::local_group_delay_range() const
{
    return { local_group_delay(), local_group_delay() + coefficients()};
}
ReSpectrum::const_range_type ReSpectrum::local_group_delay_weight_range() const
{
    return { local_group_delay_weight(), local_group_delay_weight() + coefficients()};
}
ReSpectrum::const_range_type ReSpectrum::local_group_delay_acc_range() const
{
    return { local_group_delay_acc(), local_group_delay_acc() + coefficients()};
}
ReSpectrum::const_range_type ReSpectrum::local_time_range() const
{
    return { local_time(), local_time() + coefficients()};
}


ReSpectrum::value_type ReSpectrum::epsilon() const
{
    return m_epsilon;
}
int ReSpectrum::size() const
{
    return m_size;
}
int ReSpectrum::coefficients() const
{
    return m_coef;
}
int ReSpectrum::spacing() const
{
    return m_spacing;
}
int64_t ReSpectrum::when() const
{
    return m_when;
}
void ReSpectrum::set_when(int64_t _when)
{
    m_when = _when;
}
void ReSpectrum::resize(int _size)
{
    if(_size == size())
        return;
    m_size = _size;
    m_coef = m_size / 2 + 1;
    m_spacing = align_up(m_coef, item_alignment);
    X.resize(m_spacing * 2);
    cexpr_for_each([sz=m_spacing](auto & item){item.resize(sz);}
        , mag
        , M
        , Phi
        , dM_dw
        , dPhi_dw
        , dM_dt
        , dPhi_dt
        , d2Phi_dtdw
        , lgd
        , lgd_acc
        , lgd_weight
        , ltime);
}
void ReSpectrum::reset(int _size, int64_t _when)
{
    resize(_size);
    set_when(_when);
}
