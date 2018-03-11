set( rmwarp_SOURCES ReFFT.cpp
                    ReSpectrum.cpp
                    )
file(GLOB rmwarp_HEADERS ../rmwarp/*.hpp)

set( rmwarp_FILES ${rmwarp_SOURCES} ${rmwarp_HEADERS} )
add_library( rmwarp-shared SHARED ${rmwarp_FILES} )
add_library( rmwarp-static STATIC ${rmwarp_FILES} )
#add_library( rmwarp_static STATIC ${rmwarp_FILES} )

#target_link_libraries(rmwarp Qt5::Widgets Qt5::Quick Qt5::Gui Qt5::Core)
target_link_libraries(rmwarp-shared rmwarp-lib ${FFTW_LIBRARIES})
target_link_libraries(rmwarp-static rmwarp-lib ${FFTW_LIBRARIES})

set_target_properties(rmwarp-static PROPERTIES OUTPUT_NAME rmwarp)
set_target_properties(rmwarp-shared PROPERTIES OUTPUT_NAME rmwarp)

install(TARGETS rmwarp-static rmwarp-shared DESTINATION ${rmwarp_LIBRARY_INSTALL_DIR})
#target_link_libraries(rmwarp_static  "${FFTW_LIBRARIES}" )
#target_link_libraries(rmwarp "${AVCODEC_LIBRARIES}" "${AVFORMAT_LIBRARIES}" "${AVUTIL_LIBRARIES}" "${SWRESAMPLE_LIBRARIES}" "${RUBBERBAND_LIBRARIES}" "${PORTAUDIO_LIBRARIES}")
