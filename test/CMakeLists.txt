
find_package(Portaudio REQUIRED)
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5Quick REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(Qt5Qml REQUIRED)

pkg_check_modules(RUBBERBAND rubberband)
pkg_check_modules(AVCODEC libavcodec)
pkg_check_modules(AVFORMAT libavformat)
pkg_check_modules(AVUTIL libavutil)
pkg_check_modules(SWRESAMPLE libswresample)

#target_link_libraries(remixxx_test remixxx)

#target_link_libraries(remixxx_test Qt5::Widgets Qt5::Quick Qt5::Gui Qt5::Core)
#target_link_libraries(remixxx_test "${AVCODEC_LIBRARIES}" "${AVFORMAT_LIBRARIES}" "${AVUTIL_LIBRARIES}" "${SWRESAMPLE_LIBRARIES}" "${RUBBERBAND_LIBRARIES}" "${PORTAUDIO_LIBRARIES}")
