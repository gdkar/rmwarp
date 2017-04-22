# -*- coding: utf-8 -*-

import operator
from rmwarp import ReFFT,ReSpec
def run_example(_globals, frame_size = 2048, shaping=64.,hop_size = 8,window=None,filename = '/home/zen/.test.mp4',seek_to = 64.):
    import numpy as np,scipy as sp,scipy.signal as ss,pyfftw as fftw,rmwarp,rmwarp.framer
    import matplotlib.pyplot as plt
    if window is None:
        window = ss.kaiser(frame_size,shaping)
    r = ReFFT(len(window))
    r.set_window(window)
#    r = rmwarp.RMFFT(win=window);
    f = rmwarp.NpFramer(filename,frame_size=r.size,hop_size=hop_size,layout='mono')
    f.seek_sample(seek_to)
    next(_ for _ in f if (_.std() >= 0.125))
    def fill_a_spec(n):
        return list(r.process(next(f),when=f.next_sample) for _ in range(n))
    spec = fill_a_spec(2048)
    def accumulate_reassambled(lst, cond = None, field='M'):
        _t, _w, _m = list(),list(),list()
        for __t, __w, __m in (_.reassigned_points(cond=cond,_m=field) for _ in lst):
            _t.append(__t)
            _w.append(__w)
            _m.append(__m)
        return np.concatenate(_t),np.concatenate(_w),np.concatenate(_m)

    def princarg(x):
        return np.remainder(x ,np.pi * 2)

    unit = np.arange(r.coefficients) * (2 * np.pi / r.size)
    def sames(n):
        plt.plot(-(spec[n + 1].dPhi_dt + spec[n-1].dPhi_dt)*0.5-np.pi/(2 * f.hop_size)+(n+0.25)* 4/f.hop_size)
    def diffs(n):
        plt.plot(
            np.fmod(
                (princarg(np.asarray(spec[n].Phi)-np.asarray(spec[n-1].Phi))/f.hop_size - unit) - np.pi/(2 * f.hop_size)
              , np.pi/(f.hop_size/2))
        + n * 4/f.hop_size)
    def dPhi_dt_diff(n):
        return princarg(np.asarray(spec[n].Phi)-np.asarray(spec[n-1].Phi))/f.hop_size
    def diff_resp(n):
        return np.fmod(
            (princarg(np.asarray(spec[n].Phi)-np.asarray(spec[n-1].Phi))/f.hop_size - unit) - np.pi/(2 * f.hop_size)
          , np.pi/(f.hop_size/2)
            ) + np.pi/(2*f.hop_size)
    def freq_resp (n):
        plt.plot(
            np.asarray(spec[n].dPhi_dt) + unit
          , np.asarray(spec[n].M)
            )
        plt.plot(
            unit
          , np.asarray(spec[n].M)
            )
    def dPhi_freq_resp(n):
        plt.plot(
            -diff_resp(n) + unit
          , np.asarray(spec[n].M)
            )
        plt.plot(unit,spec[n].M)
    def do_keyed(_plt,spec, key, offset):
        if _plt is None:
            _plt = plt
        vals = np.fromiter(map(lambda x:np.asarray(key(x)), spec),dtype=np.float32)
        _plt.plot(vals + offset)

    def do_one(_plt,i):
        do_keyed(_plt,spec, (lambda s:s.dPhi_dw[i]),i * f.frame_size)

#        plt.plot(
#            np.fromiter(
#                (_.dPhi_dw[i] for _ in spec)
#              , dtype=np.float32
#                ) + (
#                    np.arange(len(spec))
#                  *  f.hop_size
#                  ) + (
#                    i
#                    * f.frame_size
#                )
#            )
    def do_hann_avg(spec, lo, hi,func = operator.attrgetter('dPhi_dw'), exponent=2.):
        weight = ss.hanning(hi-lo)
        return np.sum(np.asarray(func(spec)[lo:hi]) * weight)/np.sum(weight)

    def do_uniform_avg(spec, lo, hi,func = operator.attrgetter('dPhi_dw'), exponent=2.):
        weight = np.ones_like(spec.M[lo:hi])
        return np.sum(np.asarray(func(spec))[lo:hi] * weight)/np.sum(weight)
    def do_null_avg(spec, lo, hi,func = operator.attrgetter('dPhi_dw'), exponent=2.):
        return np.asarray(func(spec))[(lo+hi)//2]

    def do_power_avg(spec, lo, hi,func = operator.attrgetter('dPhi_dw'), exponent=2.):
        weight = np.exp(exponent*np.asarray(spec.M[lo:hi]))
        return np.sum(np.asarray(func(spec))[lo:hi] * weight)/np.sum(weight)

    def do_magnitude_avg(spec, lo, hi,func = operator.attrgetter('dPhi_dw')):
        weight = np.exp(np.asarray(spec.M[lo:hi]))
        return np.sum(np.asarray(func(spec))[lo:hi] * weight)/np.sum(weight)
    def do_clipping_avg(spec, lo, hi,func = operator.attrgetter('dPhi_dw')):
        weight = (np.exp(np.asarray(spec.M)[lo:hi]*2) > r.__epsilon * 4).astype(np.float32)
        return np.sum(np.asarray(func(spec))[lo:hi] * weight)/(np.sum(weight) + r.__epsilon)

    def do_points(spec, slcount,slsize= 0,slspacing = 16, line_spacing=1,func=None):
        if func is None:
            func = operator.attrgetter('dPhi_dw')
        if not slsize:
            slsize = spec[0].coefficients //slcount
            slstride = slsize
        else:
            slstride = (spec[0].coefficients - slsize) // (slcount - 1)
        for slindex in range(slcount):
            plt.scatter(
                np.fromiter((_.when for _ in spec),dtype=np.float32),
                np.ones(len(spec)) * (slindex * slstride + 0.5 * slsize)*line_spacing,
                np.fromiter(
                    (do_magnitude_avg(_,slindex* slstride,slindex*slstride + slsize,func) for _ in spec)
                    ,dtype=np.float32))


    def do_avg_custom(spec, slcount,slsize= 0,slspacing = 16, line_spacing=1,func=None, avg = do_magnitude_avg, ref_lines = 0):
        if func is None:
            func = operator.attrgetter('dPhi_dw')
        if not slsize:
            slsize = spec[0].coefficients //slcount
            slstride = slsize
        else:
            slstride = (spec[0].coefficients - slsize) // (slcount - 1)
        for slindex in range(slcount):
            _t = np.fromiter((_.when for _ in spec),dtype=np.float32)
            _v = np.fromiter(
                    (avg(_,slindex* slstride,slindex*slstride + slsize,func) for _ in spec)
                    ,dtype=np.float32) + (slindex * slstride + 0.5 * slsize)*line_spacing
            _r = np.fromiter((0 for _ in spec) ,dtype=np.float32) + (slindex * slstride + 0.5 * slsize)*line_spacing

            line = plt.plot( _t, _v, marker='^')[0]
            plt.plot( _t, _r, linestyle='--',color=line.get_color())
            if ref_lines:
                plt.plot( _t, _r + ref_lines, linestyle=':',color=line.get_color())
                plt.plot( _t, _r - ref_lines, linestyle=':',color=line.get_color())
    def do_avg_retimed(spec, slcount,slsize= 0,slspacing = 16, line_spacing=1,func=None, avg = do_magnitude_avg, avg1 = None):
        if avg1 is None:
            avg1 = avg
        if func is None:
            func = operator.attrgetter('dPhi_dw')
        if not slsize:
            slsize = spec[0].coefficients //slcount
            slstride = slsize
        else:
            slstride = (spec[0].coefficients - slsize) // (slcount - 1)
        for slindex in range(slcount):
            plt.plot(
                np.fromiter((_.when  + _.local_group_delay[slindex*slstride + slsize//2] for _ in spec),dtype=np.float32) #slindex+avg1(_,slindex* slstride,slindex*slstride + slsize,operator.attrgetter('dPhi_dw')) for _ in spec),dtype=np.float32),
               ,np.fromiter(
                    (avg(_,slindex* slstride,slindex*slstride + slsize,func) for _ in spec)
                    ,dtype=np.float32) + (slindex * slstride + 0.5 * slsize)*line_spacing,marker= '+'
                )

    def do_avg(spec, slcount,slsize= 0,slspacing = 16, line_spacing=1,field='dPhi_dw'):
        func = operator.attrgetter(field)
        if not slsize:
            slsize = spec[0].coefficients //slcount
            slstride = slsize
        else:
            slstride = (spec[0].coefficients - slsize) // (slcount - 1)
        for slindex in range(slcount):
            plt.plot(
                np.fromiter((_.when for _ in spec),dtype=np.float32),
                np.fromiter(
                    (do_magnitude_avg(_,slindex* slstride,slindex*slstride + slsize,func) for _ in spec)
                    ,dtype=np.float32) + (slindex * slstride + 0.5 * slsize)*line_spacing
                )

    def do_one_avg(spec):
        plt.plot(np.fromiter((np.asarray(_.dPhi_dw).mean() for _ in spec),dtype=np.float32))

    _globals.update(locals())
