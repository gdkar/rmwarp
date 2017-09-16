cimport numpy as np, cython
import numpy as np, scipy as sp


cpdef cubic_hermite(
    p0, m0
  , p1, m1
  , cython.floating x
  ,cython.floating x_lo = *
  ,cython.floating x_hi = *)

cpdef linear_interp(
    p0, p1
  , cython.floating x
  , cython.floating x_lo = *
  , cython.floating x_hi = *)

cpdef np.ndarray windowed_diff(np.ndarray a, int w)
cpdef np.ndarray find_tagged_runs(np.ndarray a, int tag, int base = *)
cpdef np.ndarray find_runs(np.ndarray a, int base = *)
cpdef np.ndarray binary_dilation(np.ndarray a, int w)
cpdef np.ndarray binary_erosion(np.ndarray a, int w)
cpdef np.ndarray binary_closing(np.ndarray a, int w)
cpdef np.ndarray binary_opening(np.ndarray a, int w)
cpdef np.ndarray binary_smooth(np.ndarray a, int w)
