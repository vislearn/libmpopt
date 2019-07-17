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
# native libraries
#

find_package(GUROBI REQUIRED)

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

  install(TARGETS ${target} DESTINATION "${Python3_RELATIVE_SITEARCH}/mpopt/${sublibrary}")
endforeach()

install(DIRECTORY python/mpopt DESTINATION "${Python3_RELATIVE_SITELIB}")
install(PROGRAMS bin/ct_test bin/ct_jug bin/ct_jug_2ts bin/ct_jug_gurobi bin/gm_uai DESTINATION bin)

#
# Bytecompile Python code
#

install(CODE "execute_process(COMMAND \"${Python3_EXECUTABLE}\" -m compileall \"${CMAKE_INSTALL_PREFIX}/${Python3_RELATIVE_SITELIB}/mpopt\")")

# vim: ts=8 sts=2 sw=2 et:
