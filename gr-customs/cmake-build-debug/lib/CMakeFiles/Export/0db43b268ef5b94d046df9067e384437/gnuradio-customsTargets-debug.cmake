#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "gnuradio::gnuradio-customs" for configuration "Debug"
set_property(TARGET gnuradio::gnuradio-customs APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(gnuradio::gnuradio-customs PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libgnuradio-customs.1.0.0.0.dylib"
  IMPORTED_SONAME_DEBUG "/usr/local/lib/libgnuradio-customs.1.0.0.dylib"
  )

list(APPEND _cmake_import_check_targets gnuradio::gnuradio-customs )
list(APPEND _cmake_import_check_files_for_gnuradio::gnuradio-customs "${_IMPORT_PREFIX}/lib/libgnuradio-customs.1.0.0.0.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
