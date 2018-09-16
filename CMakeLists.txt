cmake_minimum_required(VERSION 3.0)
project(libct)

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

include_directories(include)

add_library(libct_shared SHARED lib/ct.cpp)
add_library(libct_static STATIC lib/ct.cpp)
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

execute_process(COMMAND "${Python3_EXECUTABLE}" -c "from distutils import sysconfig as sc; print(sc.get_python_lib(prefix='', plat_specific=False))"
    OUTPUT_VARIABLE Python3_RELATIVE_SITELIB
    OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND "${Python3_EXECUTABLE}" -c "from distutils import sysconfig as sc; print(sc.get_python_lib(prefix='', plat_specific=True))"
    OUTPUT_VARIABLE Python3_RELATIVE_SITEARCH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

find_package(SWIG REQUIRED)
include(UseSWIG)
set(UseSWIG_TARGET_NAME_PREFERENCE STANDARD)
swig_add_library(libct_swig TYPE SHARED
    LANGUAGE python
    SOURCES swig/ct.i)
set_target_properties(libct_swig PROPERTIES OUTPUT_NAME libct)
target_include_directories(libct_swig PUBLIC ${Python3_INCLUDE_DIRS})
target_link_libraries(libct_swig PUBLIC libct_static ${Python3_LIBRARIES})

install(PROGRAMS bin/jugct DESTINATION bin)

install(DIRECTORY python/ct
    DESTINATION "${Python3_RELATIVE_SITELIB}")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_libct.so"
              "${CMAKE_CURRENT_BINARY_DIR}/libct.py"
    DESTINATION "${Python3_RELATIVE_SITEARCH}/ct")

install(CODE "execute_process(COMMAND \"${Python3_EXECUTABLE}\" -m compileall \"${CMAKE_INSTALL_PREFIX}/${Python3_RELATIVE_SITELIB}\")")
