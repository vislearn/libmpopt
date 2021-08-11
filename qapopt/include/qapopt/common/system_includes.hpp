#ifndef LIBQAPOPT_COMMON_SYSTEM_INCLUDES_HPP
#define LIBQAPOPT_COMMON_SYSTEM_INCLUDES_HPP

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include <qapopt/common/config.h>

#ifdef ENABLE_GUROBI
#  include <gurobi_c++.h>
#endif

#ifdef ENABLE_QPBO
#  include <qpbo/QPBO.h>
#endif

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
