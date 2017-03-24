#include <thread>
#include <mutex>
#include "dsp/ReFFT.hpp"
#include "dsp/KaiserWindow.hpp"
using namespace RMWarp ;
namespace detail {
struct _wisdom_reg {
    template<class F,class D>
    static void wisdom(F && ffunc, D && dfunc, const char mode[]) {
        if(auto home = getenv("HOME")){
            char fn[256];
            ::snprintf(fn, sizeof(fn), "%s/%s.f", home, ".rubberband.wisdom");
            if(auto f = ::fopen(fn, mode)){
                std::forward<F>(ffunc)(f);
                fclose(f);
            }
            ::snprintf(fn, sizeof(fn), "%s/%s.d", home, ".rubberband.wisdom");
            if(auto f = ::fopen(fn, mode)){
                std::forward<D>(dfunc)(f);
                fclose(f);
            }
        }
    }
    static std::once_flag _wisdom_once;
    _wisdom_reg(){
        fftwf_init_threads();
        fftwf_make_planner_thread_safe();
        fftw_init_threads();
        fftw_make_planner_thread_safe();
        std::call_once(_wisdom_once,[](){
            wisdom(fftwf_import_wisdom_from_file,fftw_import_wisdom_from_file,"rb");
        });
    }
   ~_wisdom_reg() {
        wisdom(fftwf_export_wisdom_to_file,fftw_export_wisdom_to_file,"wb");
    }
};
/*static*/ std::once_flag _wisdom_reg::_wisdom_once{};
_wisdom_reg the_registrar{};
}
void ReFFT::initPlans()
{
    if(m_size) {
        if(m_plan_r2c) {
            fftwf_destroy_plan(m_plan_r2c);
            m_plan_r2c = 0;
        }
        if(m_plan_c2r) {
            fftwf_destroy_plan(m_plan_c2r);
            m_plan_c2r = 0;
        }
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
        auto dims = fftwf_iodim{ m_size, 1, 1};
        auto _real = &m_split[0]; auto _imag = &m_split[m_spacing]; auto _time = &m_flat[0];

        m_plan_r2c = fftwf_plan_guru_split_dft_r2c(
            1, &dims, 0, nullptr, _time, _real, _imag, FFTW_ESTIMATE);
        dims = fftwf_iodim{ m_size, 1, 1};
        m_plan_c2r = fftwf_plan_guru_split_dft_c2r(
            1, &dims, 0, nullptr, _real, _imag, _time, FFTW_ESTIMATE);
    }
}
/*static*/ ReFFT ReFFT::Kaiser(int _size, float alpha)
{
    auto win    = vector_type(_size, 0.0f);
    auto win_dt = vector_type(_size, 0.0f);
    make_kaiser_window(win.begin(),win.end(), alpha);
    return ReFFT(win.cbegin(),win.cend());
}

ReFFT& ReFFT::operator=(ReFFT && o ) noexcept
{
    swap(o);
    return *this;
}
void ReFFT::swap(ReFFT &o) noexcept
{
    using std::swap;
    swap(m_size,o.m_size);
    swap(m_coef,o.m_coef);
    swap(m_spacing,o.m_spacing);
    swap(m_h,o.m_h);
    swap(m_Dh,o.m_Dh);
    swap(m_Th,o.m_Th);
    swap(m_TDh,o.m_TDh);
    swap(m_flat,o.m_flat);
    swap(m_split,o.m_split);
    swap(m_X,o.m_X);
    swap(m_X_Dh,o.m_X_Dh);
    swap(m_X_Th,o.m_X_Th);
    swap(m_X_TDh,o.m_X_TDh);
    swap(m_plan_r2c,o.m_plan_r2c);
    swap(m_plan_c2r,o.m_plan_c2r);
}
ReFFT::ReFFT(ReFFT && o ) noexcept
: ReFFT(0)
{
    swap(o);
}
ReFFT::~ReFFT()
{
    if(m_size) {
        if(m_plan_r2c) {
            fftwf_destroy_plan(m_plan_r2c);
            m_plan_r2c = 0;
        }
        if(m_plan_c2r) {
            fftwf_destroy_plan(m_plan_c2r);
            m_plan_c2r = 0;
        }
    }
}

void ReFFT::_finish_process( ReSpectrum & dst, int64_t _when )
{
    using reg = simd_reg<float>;
    using std::tie; using std::make_pair; using std::copy; using std::get;

    constexpr auto w = int(simd_width<float>);

    const auto _real = &m_X[0], _imag    = &m_X[m_spacing]
        ,_real_Dh = &m_X_Dh[0], _imag_Dh = &m_X_Dh[m_spacing]
        ,_real_Th = &m_X_Th[0], _imag_Th = &m_X_Th[m_spacing]
        ,_real_TDh = &m_X_TDh[0], _imag_TDh = &m_X_TDh[m_spacing]
        ;
    auto _cmul = [](auto r0, auto i0, auto r1, auto i1) {
        return make_pair(r0 * r1 - i0 * i1, r0 * i1 + r1 * i0);
        };
    auto _pcmul = [&](auto x0, auto x1) {
        return _cmul(std::get<0>(x0),std::get<1>(x0),
                     std::get<0>(x1),std::get<1>(x1));
    };
    auto _cinv = [e=m_epsilon](auto r, auto i) {
        auto n = bs::rec(bs::sqr(r) + bs::sqr(i) + e);//bs::Eps<float>());
        return make_pair(r * n , -i * n);
    };
    dst.reset(m_size, _when);

    for(auto i = 0; i < m_coef; i += w ) {
        auto _X_r = reg(_real + i), _X_i = reg(_imag + i);
        bs::store(_X_r, dst.X_real() + i);
        bs::store(_X_i, dst.X_imag() + i);
        {
//            auto _X_mag = bs::sqr(_X_i) + bs::sqr(_X_r);
            auto _X_mag = bs::hypot(_X_i,_X_r);
            bs::store(_X_mag, dst.mag_data() + i);
            bs::store(bs::log(_X_mag), dst.M_data() + i);
            bs::store(bs::atan2(_X_i,_X_r), dst.Phi_data() + i);
        }

        tie(_X_r, _X_i) = _cinv(_X_r,_X_i);

        auto _Dh_over_X = _cmul( reg(_real_Dh + i),reg(_imag_Dh + i) ,_X_r, _X_i );

        bs::store(get<0>(_Dh_over_X), &dst.dM_dt  [0] + i);
        bs::store(get<1>(_Dh_over_X), &dst.dPhi_dt[0] + i);

        auto _Th_over_X = _cmul( reg(_real_Th + i),reg(_imag_Th + i) ,_X_r, _X_i );

        bs::store(-get<1>(_Th_over_X), &dst.dM_dw  [0] + i);
        bs::store( get<0>(_Th_over_X), &dst.dPhi_dw[0] + i);

        auto _TDh_over_X    = std::get<0>(_cmul( reg(_real_TDh + i),reg(_imag_TDh + i) ,_X_r, _X_i ));
        auto _Th_Dh_over_X2 = std::get<0>(_pcmul(_Th_over_X,_Dh_over_X));

        bs::store(_TDh_over_X - _Th_Dh_over_X2,&dst.d2Phi_dtdw[0] + i);    }
    for(auto i = 0; i < m_coef; ++i) {
        auto _X_r = *(_real + i), _X_i = *(_imag + i);
        bs::store(_X_r, dst.X_real() + i);
        bs::store(_X_i, dst.X_imag() + i);
        {
            auto _X_mag = bs::hypot(_X_i,_X_r);
//            auto _X_mag = bs::sqr(_X_i) + bs::sqr(_X_r);
            bs::store(_X_mag, dst.mag_data() + i);
            bs::store(bs::log(_X_mag), dst.M_data() + i);
            bs::store(bs::atan2(_X_i,_X_r), dst.Phi_data() + i);
        }
        tie(_X_r, _X_i) = _cinv(_X_r,_X_i);

        auto _Dh_over_X = _cmul( *(_real_Dh + i),*(_imag_Dh + i) ,_X_r, _X_i );

        bs::store(get<0>(_Dh_over_X), &dst.dM_dt  [0] + i);
        bs::store(get<1>(_Dh_over_X), &dst.dPhi_dt[0] + i);

        auto _Th_over_X = _cmul( *(_real_Th + i),*(_imag_Th + i) ,_X_r, _X_i );

        bs::store(-get<1>(_Th_over_X), &dst.dM_dw  [0] + i);
        bs::store( get<0>(_Th_over_X), &dst.dPhi_dw[0] + i);

        auto _TDh_over_X    = std::get<0>(_cmul( *(_real_TDh + i),*(_imag_TDh + i) ,_X_r, _X_i ));
        auto _Th_Dh_over_X2 = std::get<0>(_pcmul(_Th_over_X,_Dh_over_X));

        bs::store(_TDh_over_X - _Th_Dh_over_X2,&dst.d2Phi_dtdw[0] + i);
    }
}
int ReFFT::spacing() const
{
    return m_spacing;
}
int ReFFT::size() const
{
    return m_size;
}
int ReFFT::coefficients() const
{
    return m_coef;
}
void ReFFT::updateGroupDelay(ReSpectrum &spec)
{
    auto _lgd     = spec.local_group_delay();
    auto _lgda    = spec.local_group_delay_acc();
    auto _lgdw    = spec.local_group_delay_weight();
    auto _ltime   = spec.local_time();

    auto _mag     = spec.mag_data();

    auto _dPhi_dw = spec.dPhi_dw_data();

    auto fr = 0.9f;//bs::pow(10.0f, -40.0f/20.0f);
    auto ep = bs::sqrt(m_epsilon * fr * bs::rec(1-fr));

    bs::transform(_mag,_mag + m_coef, _lgdw, [ep](auto m) {
        auto _res = bs::is_less(m,decltype(m)(ep));
        return bs::if_zero_else_one(_res);
    });
/*    std::transform(_mag,_mag + m_coef, _lgdw,[ep](auto m) {
        return (m > ep) ? 1.0f : 0.0f;
    });*/
    bs::transform(_lgdw,_lgdw + m_coef, _dPhi_dw, _lgda,bs::multiplies);

    std::partial_sum(_lgda,_lgda+m_coef,_lgda);
    std::partial_sum(_lgdw,_lgdw+m_coef,_lgdw);

    auto i = 0;
    auto hi_bound = [](auto x){return (x * 1200)/1024;};
    auto lo_bound = [](auto x){return (x * 860 )/1024;};
    for(; i < m_coef - 8 && hi_bound(i) < i + 8; ++i) {
        auto hi = i + 8;
        auto lo = lo_bound(i);

        auto _w = _lgdw[hi] - _lgdw[lo] + m_epsilon;
        auto _d = _lgda[hi] - _lgda[lo];
        _lgd[i] = -_d * bs::rec(_w);
    }
    for(; hi_bound(i) < m_coef; ++i) {
        auto hi = hi_bound(i);
        auto lo = lo_bound(i);
        auto _w = _lgdw[hi] - _lgdw[lo] + m_epsilon;
        auto _d = _lgda[hi] - _lgda[lo];
        _lgd[i] = -_d * bs::rec(_w);
    }
    {
        auto hi = m_coef - 1;
        auto hiw = _lgdw[hi];
        auto hid = _lgda[hi];
        for(; i < m_coef; ++i) {
            auto lo = lo_bound(i);
            auto _w = hiw - _lgdw[lo] + m_epsilon;
            auto _d = hid - _lgda[lo];
            _lgd[i] = -_d * bs::rec(_w);
        }
    }
    bs::transform(_lgd,_lgd+m_coef, _ltime, [w=float(spec.when())](auto x){return x + w;});
}
