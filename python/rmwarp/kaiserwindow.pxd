
cdef extern from "rmwarp/KaiserWindow.hpp" namespace "RMWarp" nogil:
    It make_kaiser_window[It](It _beg, It _end, float alpha)
    It make_kaiser_bessel_derived_window[It](It _beg, It _end, float alpha)
    It make_xiph_vorbis_window[It](It _beg, It _end)
    It make_sinc_window[It,S](It _beg, It _end, S p)

