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
# Customizable choices
#

set(ENABLE_GUROBI OFF CACHE BOOL "Enable Gurobi LP/ILP support")

#
# C++ settings
#

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

configure_file(include/mpopt/common/config.h.in include/mpopt/common/config.h @ONLY)
include_directories("${CMAKE_CURRENT_BINARY_DIR}/include")
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/mpopt" DESTINATION include)

#
# Python settings
#

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
set(SUPERSEDE_PYTHON_SITELIB "" CACHE PATH "Force installation of Python3 module to a different directory.")
if(NOT "${SUPERSEDE_PYTHON_SITELIB}" STREQUAL "")
  set(Python3_SITELIB "${SUPERSEDE_PYTHON_SITELIB}")
  set(Python3_SITEARCH "${SUPERSEDE_PYTHON_SITELIB}")
endif()

set(Python3_OPTIMIZEFLAG "-OO")
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  set(Python3_OPTIMIZEFLAG "")
endif()

#
# SWIG
#

find_package(SWIG REQUIRED)
include(UseSWIG)

#
# native libraries
#

if(ENABLE_GUROBI)
  find_package(GUROBI REQUIRED)
else()
  set(GUROBI_INCLUDE_DIRS "")
  set(GUROBI_LIBRARIES "")
endif()

foreach(sublibrary ct gm qap)
  foreach(kind shared static)
    set(target libmpopt_${sublibrary}_${kind})
    string(TOUPPER ${kind} KIND)
    add_library(${target} ${KIND} lib/${sublibrary}.cpp)
    target_include_directories(${target} PUBLIC ${GUROBI_INCLUDE_DIRS})
    target_link_libraries(${target} PUBLIC ${GUROBI_LIBRARIES})
    set_target_properties(${target} PROPERTIES
      OUTPUT_NAME mpopt_${sublibrary}
      POSITION_INDEPENDENT_CODE ON
      VERSION "${MPOPT_VERSION_STRING}")
    install(TARGETS ${target} DESTINATION lib)
  endforeach()
endforeach()

install(DIRECTORY include/mpopt DESTINATION include)

#
# Python bindings
#

configure_file(python/mpopt/common/config.py.in python/mpopt/common/config.py @ONLY)
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python/mpopt" DESTINATION "${Python3_SITELIB}")

foreach(sublibrary ct gm qap)
  set_source_files_properties(swig/${sublibrary}.i PROPERTIES DEPENDS include/mpopt/${sublibrary}.h)
  set(target libmpopt_${sublibrary}_swig_py)
  swig_add_library(${target}
    TYPE SHARED
    LANGUAGE python
    SOURCES swig/${sublibrary}.i
    NO_PROXY)
  set_target_properties(${target} PROPERTIES
    OUTPUT_NAME libmpopt_${sublibrary}
    LIBRARY_OUTPUT_DIRECTORY ${target})
  target_include_directories(${target} PUBLIC ${Python3_INCLUDE_DIRS})
  target_link_libraries(${target} PUBLIC libmpopt_${sublibrary}_static ${Python3_LIBRARIES})

  # This should be handled in FindSWIG.cmake if LANGUAGE is set to Python.
  # TODO: File upstream bug/PR.
  if(APPLE)
    set_target_properties(${target} PROPERTIES SUFFIX .so)
  endif()

  install(TARGETS ${target} DESTINATION "${Python3_SITEARCH}/mpopt/${sublibrary}")
endforeach()

install(DIRECTORY python/mpopt DESTINATION "${Python3_SITELIB}")

#
# Bytecompile Python code
#

install(CODE "execute_process(COMMAND \"${Python3_EXECUTABLE}\"     -m compileall \"${Python3_SITELIB}/mpopt\")")
install(CODE "execute_process(COMMAND \"${Python3_EXECUTABLE}\" -O  -m compileall \"${Python3_SITELIB}/mpopt\")")
install(CODE "execute_process(COMMAND \"${Python3_EXECUTABLE}\" -OO -m compileall \"${Python3_SITELIB}/mpopt\")")

#
# Command line scripts
#

function(process_pywrapper cmdline_script module)
  configure_file("bin/pywrapper.in" "bin/${cmdline_script}" @ONLY)
  install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/bin/${cmdline_script}" DESTINATION bin)
endfunction()

process_pywrapper(ct_jug        mpopt.ct.cmdline.jug)
process_pywrapper(ct_jug_2ts    mpopt.ct.cmdline.jug2_ts)
process_pywrapper(ct_jug_gurobi mpopt.ct.cmdline.jug_gurobi)
process_pywrapper(ct_test       mpopt.ct.cmdline.test)
process_pywrapper(gm_uai        mpopt.gm.cmdline.uai)
process_pywrapper(dd_to_uai     mpopt.qap.cmdline.dd_to_uai)
process_pywrapper(qap_dd        mpopt.qap.cmdline.dd)

# vim: ts=8 sts=2 sw=2 et:
