set( rmwarp_SOURCES FFT.cpp
                    ReassignedFFT.cpp )

set( rmwarp_HEADERS ../include/FFT.hpp
                    ../include/Simd.hpp
                    ../include/Math.hpp
                    ../include/ReSpectrum.hpp
                    ../include/ReFFT.hpp
                    TimeWeightedWindow.hpp
                    TimeDerivativeWindow.hpp )

set( rmwarp_FILES ${rmwarp_SOURCES} ${rmwarp_HEADERS} )
add_library( rmwarp-shared SHARED ${rmwarp_FILES} )
add_library( rmwarp-static STATIC ${rmwarp_FILES} )
#add_library( rmwarp_static STATIC ${rmwarp_FILES} )
set_target_properties(rmwarp-static PROPERTIES OUTPUT_NAME rmwarp)
set_target_properties(rmwarp-shared PROPERTIES OUTPUT_NAME rmwarp)

#target_link_libraries(rmwarp Qt5::Widgets Qt5::Quick Qt5::Gui Qt5::Core)
target_link_libraries(rmwarp-shared  "${FFTW_LIBRARIES}" )
target_link_libraries(rmwarp-static "${FFTW_LIBRARIES}" )
#target_link_libraries(rmwarp_static  "${FFTW_LIBRARIES}" )
#target_link_libraries(rmwarp "${AVCODEC_LIBRARIES}" "${AVFORMAT_LIBRARIES}" "${AVUTIL_LIBRARIES}" "${SWRESAMPLE_LIBRARIES}" "${RUBBERBAND_LIBRARIES}" "${PORTAUDIO_LIBRARIES}")
