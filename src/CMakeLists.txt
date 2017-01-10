set( rmwarp_SOURCES FFT.cpp
                    ReassignedFFT.cpp )

set( rmwarp_HEADERS ../include/FFT.hpp
                    ../include/Simd.hpp
                    ../include/Math.hpp
                    ../include/ReassignedSpectrum.hpp
                    ../include/ReassignedFFT.hpp
                    TimeWeightedWindow.hpp
                    TimeDerivativeWindow.hpp )

set( rmwarp_FILES ${rmwarp_SOURCES} ${rmwarp_HEADERS} )
add_library( rmwarp SHARED ${rmwarp_FILES} )

#target_link_libraries(rmwarp Qt5::Widgets Qt5::Quick Qt5::Gui Qt5::Core)
target_link_libraries(rmwarp  "${FFTW_LIBRARIES}" )
#target_link_libraries(rmwarp "${AVCODEC_LIBRARIES}" "${AVFORMAT_LIBRARIES}" "${AVUTIL_LIBRARIES}" "${SWRESAMPLE_LIBRARIES}" "${RUBBERBAND_LIBRARIES}" "${PORTAUDIO_LIBRARIES}")
