#ifndef LIBMPOPT_COMMON_MISC_HPP
#define LIBMPOPT_COMMON_MISC_HPP

namespace mpopt {

template<typename FORWARD_ITERATOR_DATA, typename FORWARD_ITERATOR_BOOL>
auto min_element(FORWARD_ITERATOR_DATA data_begin, FORWARD_ITERATOR_DATA data_end,
                 FORWARD_ITERATOR_BOOL active_begin, FORWARD_ITERATOR_BOOL active_end)
{
  assert(std::distance(active_begin, active_end) >= std::distance(data_begin, data_end));
  auto minimum = data_end;

  auto data_it = data_begin;
  auto active_it = active_begin;
  for (; data_it != data_end; ++data_it, ++active_it) {
    if (*active_it && (minimum == data_end || *data_it < *minimum))
      minimum = data_it;
  }

  return minimum;
}

template<typename FORWARD_ITERATOR>
auto least_two_elements(FORWARD_ITERATOR begin, FORWARD_ITERATOR end)
{
  auto first = end, second = end;

  for (auto it = begin; it != end; ++it) {
    if (first == end || *it < *first) {
      second = first;
      first = it;
    } else if (second == end || *it < *second) {
      second = it;
    }
  }

  return std::make_tuple(first, second);
}

template<typename FORWARD_ITERATOR>
auto least_two_values(FORWARD_ITERATOR begin, FORWARD_ITERATOR end)
{
  using value_type = std::decay_t<decltype(*begin)>;
  constexpr auto inf = std::numeric_limits<value_type>::infinity();

  auto [first, second] = least_two_elements(begin, end);

  const auto first_val = first != end ? *first : inf;
  const auto second_val = second != end ? *second : inf;

  return std::make_tuple(first_val, second_val);
}

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
