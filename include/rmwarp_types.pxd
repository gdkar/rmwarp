cdef extern from "FFT.hpp" namespace "RMWarp" nogil:
    cdef cppclass RFFT:
        RFFT()
        RFFT(int _size)
        ~RFFT();
