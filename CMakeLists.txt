cmake_minimum_required(VERSION 3.13)
cmake_policy(VERSION 3.13)
project(libct)
include_directories(include)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(LIBCT_VERSION_MAJOR 0)
set(LIBCT_VERSION_MINOR 0)
set(LIBCT_VERSION_PATCH 0)
set(LIBCT_VERSION_STRING "${LIBCT_VERSION_MAJOR}.${LIBCT_VERSION_MINOR}.${LIBCT_VERSION_PATCH}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#
# libct library
#

find_package(GUROBI REQUIRED)

add_library(libct_shared SHARED lib/ct.cpp)
add_library(libct_static STATIC lib/ct.cpp)
foreach(target libct_shared libct_static)
  target_include_directories(${target} PUBLIC ${GUROBI_INCLUDE_DIRS})
  target_link_libraries(${target} PUBLIC ${GUROBI_LIBRARIES})
endforeach()
set_target_properties(libct_shared libct_static PROPERTIES
  OUTPUT_NAME ct
  POSITION_INDEPENDENT_CODE ON
  VERSION "${LIBCT_VERSION_STRING}")
install(TARGETS libct_shared libct_static DESTINATION lib)
install(DIRECTORY include/ct DESTINATION include)

#
# Python binding
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
set_source_files_properties(swig/ct.i PROPERTIES DEPENDS include/ct/ct.h)
swig_add_library(libct_swig_py
  TYPE SHARED
  LANGUAGE python
  SOURCES swig/ct.i
  NO_PROXY)
set_target_properties(libct_swig_py PROPERTIES
  OUTPUT_NAME libct
  LIBRARY_OUTPUT_DIRECTORY libct_swig_py)
target_include_directories(libct_swig_py PUBLIC ${Python3_INCLUDE_DIRS})
target_link_libraries(libct_swig_py PUBLIC libct_static ${Python3_LIBRARIES})

# This should be handled in FindSWIG.cmake if LANGUAGE is set to Python.
# TODO: File upstream bug/PR.
if(APPLE)
  set_target_properties(libct_swig_py PROPERTIES SUFFIX .so)
endif()

install(DIRECTORY python/ct
  DESTINATION "${Python3_RELATIVE_SITELIB}")

install(TARGETS libct_swig_py
  DESTINATION "${Python3_RELATIVE_SITEARCH}/ct")

install(CODE "execute_process(COMMAND \"${Python3_EXECUTABLE}\" -m compileall \"${CMAKE_INSTALL_PREFIX}/${Python3_RELATIVE_SITELIB}\")")

#
# Executables
#

install(PROGRAMS bin/ct_test bin/jug_ct bin/jug_ct_2ts bin/jug_gurobi
  DESTINATION bin)

# vim: ts=8 sts=2 sw=2 et:
