cmake_minimum_required(VERSION 3.13)
cmake_policy(VERSION 3.13)
project(mpopt)
include_directories(include)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(MPOPT_VERSION_MAJOR 0)
set(MPOPT_VERSION_MINOR 0)
set(MPOPT_VERSION_PATCH 0)
set(MPOPT_VERSION_STRING "${MPOPT_VERSION_MAJOR}.${MPOPT_VERSION_MINOR}.${MPOPT_VERSION_PATCH}")

#
# C++ settings
#

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#
# Python settings
#

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# We can't use Python3_SITELIB and _SITEARCH here, as it contains the
# installations prefix, but we really would like to have it relative to
# ${CMAKE_INSTALL_PREFIX} :-/

execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python_sitelib_helper" sitelib
  OUTPUT_VARIABLE Python3_RELATIVE_SITELIB
  OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python_sitelib_helper" sitearch
  OUTPUT_VARIABLE Python3_RELATIVE_SITEARCH
  OUTPUT_STRIP_TRAILING_WHITESPACE)

find_package(SWIG REQUIRED)
include(UseSWIG)

#
# cell tracking library
#

find_package(GUROBI REQUIRED)

add_library(libmpopt_ct_shared SHARED lib/ct.cpp)
add_library(libmpopt_ct_static STATIC lib/ct.cpp)
foreach(target libmpopt_ct_shared libmpopt_ct_static)
  target_include_directories(${target} PUBLIC ${GUROBI_INCLUDE_DIRS})
  target_link_libraries(${target} PUBLIC ${GUROBI_LIBRARIES})
endforeach()
set_target_properties(libmpopt_ct_shared libmpopt_ct_static PROPERTIES
  OUTPUT_NAME mpopt_ct
  POSITION_INDEPENDENT_CODE ON
  VERSION "${MPOPT_VERSION_STRING}")
install(TARGETS libmpopt_ct_shared libmpopt_ct_static DESTINATION lib)
install(DIRECTORY include/mpopt/ct DESTINATION include/mpopt)

#
# cell tracking Python bindings
#

set_source_files_properties(swig/ct.i PROPERTIES DEPENDS include/mpopt/ct.h)
swig_add_library(libmpopt_ct_swig_py
  TYPE SHARED
  LANGUAGE python
  SOURCES swig/ct.i
  NO_PROXY)
set_target_properties(libmpopt_ct_swig_py PROPERTIES
  OUTPUT_NAME libmpopt_ct
  LIBRARY_OUTPUT_DIRECTORY libmpopt_ct_swig_py)
target_include_directories(libmpopt_ct_swig_py PUBLIC ${Python3_INCLUDE_DIRS})
target_link_libraries(libmpopt_ct_swig_py PUBLIC libmpopt_ct_static ${Python3_LIBRARIES})

# This should be handled in FindSWIG.cmake if LANGUAGE is set to Python.
# TODO: File upstream bug/PR.
if(APPLE)
  set_target_properties(libmpopt_ct_swig_py PROPERTIES SUFFIX .so)
endif()

install(DIRECTORY python/mpopt/ct
  DESTINATION "${Python3_RELATIVE_SITELIB}/mpopt")

install(TARGETS libmpopt_ct_swig_py
  DESTINATION "${Python3_RELATIVE_SITEARCH}/mpopt/ct")

#
# cell tracking executables
#

install(PROGRAMS bin/ct_test bin/ct_jug bin/ct_jug_2ts bin/ct_jug_gurobi
  DESTINATION bin)

#
# Bytecompile Python code
#

install(CODE "execute_process(COMMAND \"${Python3_EXECUTABLE}\" -m compileall \"${CMAKE_INSTALL_PREFIX}/${Python3_RELATIVE_SITELIB}/mpopt\")")

# vim: ts=8 sts=2 sw=2 et:
