find_package(PkgConfig)

PKG_CHECK_MODULES(PC_GR_CUSTOMS gnuradio-customs)

FIND_PATH(
    GR_CUSTOMS_INCLUDE_DIRS
    NAMES gnuradio/customs/api.h
    HINTS $ENV{CUSTOMS_DIR}/include
        ${PC_CUSTOMS_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    GR_CUSTOMS_LIBRARIES
    NAMES gnuradio-customs
    HINTS $ENV{CUSTOMS_DIR}/lib
        ${PC_CUSTOMS_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/gnuradio-customsTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GR_CUSTOMS DEFAULT_MSG GR_CUSTOMS_LIBRARIES GR_CUSTOMS_INCLUDE_DIRS)
MARK_AS_ADVANCED(GR_CUSTOMS_LIBRARIES GR_CUSTOMS_INCLUDE_DIRS)
