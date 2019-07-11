#ifndef LIBMPOPT_COMMON_TYPES_HPP
#define LIBMPOPT_COMMON_TYPES_HPP

namespace mpopt {

using cost = double;
constexpr const cost epsilon = 1e-8;
static_assert(std::numeric_limits<cost>::has_infinity);

using index = unsigned int;
using short_index = unsigned char;

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
