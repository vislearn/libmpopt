#ifndef LIBMPOPT_COMMON_TYPES_HPP
#define LIBMPOPT_COMMON_TYPES_HPP

namespace mpopt {

using cost = double;
constexpr const cost epsilon = 1e-8;
constexpr const cost infinity = std::numeric_limits<cost>::infinity();
static_assert(std::numeric_limits<cost>::has_infinity);
static_assert(std::numeric_limits<cost>::has_signaling_NaN);
static_assert(std::numeric_limits<cost>::has_quiet_NaN);

using index = unsigned int;
using short_index = unsigned char;

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
