# Copyright 2020 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

########################################################################
# Check if there is C++ code at all
########################################################################
if(NOT customs_sources)
    message(STATUS "No C++ sources... skipping python bindings")
    return()
endif(NOT customs_sources)

########################################################################
# Check for pygccxml
########################################################################
gr_python_check_module_raw("pygccxml" "import pygccxml" PYGCCXML_FOUND)

include(GrPybind)

########################################################################
# Python Bindings
########################################################################

list(APPEND customs_python_files
    ofdm_carrier_preamble_python.cc python_bindings.cc)

gr_pybind_make_oot(customs ../../.. gr::customs "${customs_python_files}")

# copy bindings extension for use in QA test module
add_custom_command(
    TARGET customs_python
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:customs_python>
            ${PROJECT_BINARY_DIR}/test_modules/gnuradio/customs/)

install(
    TARGETS customs_python
    DESTINATION ${GR_PYTHON_DIR}/gnuradio/customs
    COMPONENT pythonapi)
