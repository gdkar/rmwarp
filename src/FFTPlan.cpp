#include <thread>
#include <mutex>
#include "ReFFT.hpp"
#include "KaiserWindow.hpp"
#include "TimeAlias.hpp"

namespace RMWarp {
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
static void _do_initialize() {
    std::call_once(_wisdom_once,[]() {
        fftwf_make_planner_thread_safe();
        fftw_make_planner_thread_safe();
//            fftwf_init_threads();
//            fftw_init_threads();
        wisdom(fftwf_import_wisdom_from_file,fftw_import_wisdom_from_file,"rb");
    });
}
_wisdom_reg(int x) {
    void(sizeof(x));
    _do_initialize();
}
~_wisdom_reg() {
    wisdom(fftwf_export_wisdom_to_file,fftw_export_wisdom_to_file,"wb");
    fftw_cleanup();
    fftwf_cleanup();
}
static std::once_flag _wisdom_once;
};
/*static*/ std::once_flag _wisdom_reg::_wisdom_once{};
auto the_registrar = _wisdom_reg{0};
}
}
using namespace RMWarp;
FFTPlan FFTPlan::dft_1d_r2c(int _n, pointer _ti, pointer _ro, pointer _io)
{
    detail::_wisdom_reg::_do_initialize();
    auto dims = fftwf_iodim{ _n, 1, 1};
    auto off_out = _io - _ro;

    if(off_out == 1 || off_out == -1)
        dims.os = 2;

    auto _d = fftwf_plan_guru_split_dft_r2c(
        1, &dims, 0, nullptr, _ti, _ro, _ro + off_out, FFTW_ESTIMATE);
    return { _d, &fftwf_execute_split_dft_r2c, 0, off_out};
}
FFTPlan FFTPlan::dft_1d_c2r(int _n, pointer _ri, pointer _ii, pointer _to)
{
    detail::_wisdom_reg::_do_initialize();
    auto dims = fftwf_iodim{ _n, 1, 1};
    auto off_in = _ii - _ri;
    if(off_in == 1 || off_in == -1)
        dims.is = 2;

    auto _d = fftwf_plan_guru_split_dft_c2r(
        1, &dims, 0, nullptr, _ri, _ri + off_in, _to, FFTW_ESTIMATE);
    return {_d, &fftwf_execute_split_dft_c2r, off_in, 0};
}
FFTPlan FFTPlan::dft_1d_c2c(int _n, pointer _ri, pointer _ii, pointer _ro,pointer _io)
{
    detail::_wisdom_reg::_do_initialize();
    auto dims = fftwf_iodim{ _n, 1, 1};
    auto off_in = _ii - _ri;
    auto off_out= _io - _ro;

    if(off_in == 1 || off_in == -1)
        dims.is = 2;
    if(off_out == 1 || off_out == -1)
        dims.os = 2;

    auto flags = FFTW_ESTIMATE;
    auto _d = fftwf_plan_guru_split_dft(
        1, &dims, 0, nullptr, _ri, _ri + off_in, _ro, _ro + off_out,flags);
    return { _d, &fftwf_execute_split_dft, off_in, off_out};
}
void FFTPlan::execute(pointer _in, pointer _out) const
{
    if(!m_d || !m_r2c || !_in || !_out)
        return;

    if(m_off_out && !m_off_in) {
        (*m_r2c)(
            m_d
            , _in
            , _out
            , _out + m_off_out
        );
    }else if(m_off_in && !m_off_out) {
        (*m_r2c)(
            m_d
            , _in
            , _in + m_off_in
            , _out
        );
    } else {
        (*m_c2c) (
            m_d
            , _in
            , _in + m_off_in
            , _out
            , _out + m_off_out
        );
    }
}
void FFTPlan::execute() const
{
    if(m_d)
        fftwf_execute(m_d);
}
void FFTPlan::reset()
{
    detail::_wisdom_reg::_do_initialize();
    if(m_d)
        fftwf_destroy_plan(m_d);
    m_d = 0;
    m_off_in = 0;
    m_off_out = 0;
    m_r2c = nullptr;
}
FFTPlan::plan_type FFTPlan::release() noexcept
{
    auto res = std::exchange(m_d, plan_type{});
    m_r2c = nullptr;
    m_off_in = m_off_out = 0;
    return res;
}

