#ifndef LIBCT_TYPES_HPP
#define LIBCT_TYPES_HPP

namespace ct {

using cost = double;
constexpr const cost epsilon = 1e-8;

using index = unsigned int;
using short_index = unsigned char;

constexpr index max_number_of_detection_edges = 128;
constexpr index max_number_of_conflict_edges = 5;

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
