#ifndef LIBMPOPT_COMMON_SYSTEM_INCLUDES_HPP
#define LIBMPOPT_COMMON_SYSTEM_INCLUDES_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <mpopt/common/config.h>

#if __GNUC__ >= 11
#   include <source_location>
#else
#   include <experimental/source_location>
    namespace std { using source_location = experimental::source_location; };
#endif

#ifdef ENABLE_GUROBI
#   include <gurobi_c++.h>
#endif

#ifdef ENABLE_QPBO
#   include <qpbo/qpbo.h>
#endif

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
