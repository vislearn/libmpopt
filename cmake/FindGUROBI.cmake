set(GUROBI_HOME "$ENV{GUROBI_HOME}" CACHE PATH "GUROBI root directory.")

set(GUROBI_LIBRARY_POSSIBLE_NAMES libgurobi.so)
foreach(major RANGE 9 7 -1)
  foreach(minor RANGE 9 0 -1)
    list(APPEND GUROBI_LIBRARY_POSSIBLE_NAMES "libgurobi${major}${minor}.so")
  endforeach(minor)
endforeach(major)

find_path(GUROBI_INCLUDE_DIR gurobi_c++.h HINTS "${GUROBI_HOME}/include")
find_library(GUROBI_C_LIBRARY NAMES ${GUROBI_LIBRARY_POSSIBLE_NAMES} HINTS "${GUROBI_HOME}/lib")
find_library(GUROBI_CPP_LIBRARY libgurobi_g++5.2.a HINTS "${GUROBI_HOME}/lib")
mark_as_advanced(GUROBI_INCLUDE_DIR GUROBI_C_LIBRARY GUROBI_CPP_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GUROBI DEFAULT_MSG GUROBI_C_LIBRARY GUROBI_CPP_LIBRARY GUROBI_INCLUDE_DIR)

if(GUROBI_FOUND)
  set(GUROBI_INCLUDE_DIRS "${GUROBI_INCLUDE_DIR}")
  set(GUROBI_LIBRARIES "${GUROBI_CPP_LIBRARY}" "${GUROBI_C_LIBRARY}")
endif()

add_library(GUROBI INTERFACE)
target_include_directories(GUROBI INTERFACE ${GUROBI_INCLUDE_DIRS})
target_link_libraries(GUROBI INTERFACE ${GUROBI_LIBRARIES})

# vim: set ts=8 sts=2 sw=2 et:
