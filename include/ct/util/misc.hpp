#ifndef LIBCT_MISC_UTILS_HPP
#define LIBCT_MISC_UTILS_HPP

namespace ct {

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

// FIXME: Compute uniform minorant more efficiently.
inline
std::vector<std::array<cost,2>> uniform_minorant_generic(const std::vector<std::array<cost,2>> &duals)
{
  using minorant_type = std::vector<std::array<cost,2>>;

  std::vector<std::array<int,2>> indicator(duals.size());
  minorant_type minorant(duals.size());
  minorant_type f_minus_g(duals.size()); // f - g

  for (index i = 0; i < duals.size(); ++i) {
    for (bool on : { false, true }) {
      indicator[i][on] = 1;
      minorant[i][on] = 0;
      f_minus_g[i][on] = duals[i][on];
    }
  }

  auto continue_check = [](auto x) { return x[0] > 0 || x[1] > 0; };
  for (int iteration = 0; std::count_if(indicator.begin(), indicator.end(), continue_check) > 0; ++iteration) {
    index argmin = std::numeric_limits<index>::max();
    cost min = std::numeric_limits<cost>::infinity();
    for (index i = 0; i < duals.size(); ++i) {
      index h_x = 0;
      cost current = 0;
      for (index j = 0; j < duals.size(); ++j) {
        current += f_minus_g[j][i == j];
        if (indicator[j][i == j])
          ++h_x;
      }
      current /= h_x;
      if (h_x != 0 && current < min) {
        min = current;
        argmin = i;
      }
    }

    index h_x = 0;
    for (index i = 0; i < duals.size(); ++i)
      if (indicator[i][i == argmin])
        ++h_x;

    const cost& epsilon = min;

    for (index i = 0; i < duals.size(); ++i) {
      for (bool on : {false, true}) {
        minorant[i][on] += epsilon * indicator[i][on];
        f_minus_g[i][on] -= epsilon * indicator[i][on];
      }
    }

    for (index i = 0; i < duals.size(); ++i)
      indicator[i][i == argmin] = 0;
  }

#ifndef NDEBUG
  for (index on = 0; on < duals.size() - 1; ++on) {
    cost o = 0, n = 0;
    for (index i = 0; i < duals.size(); ++i) {
      o += duals[i][i == on];
      n += minorant[i][i == on];
    }
    assert(std::abs(o - n) < 1e-8);
  }
#endif

  return minorant;
}

template<typename ITERATOR_IN /* of cost */, typename ITERATOR_OUT /* of std::array<cost, 2> */>
void uniform_minorant(ITERATOR_IN in_begin, ITERATOR_IN in_end, ITERATOR_OUT out_begin, ITERATOR_OUT out_end)
{
  assert(std::distance(in_begin, in_end) <= std::distance(out_begin, out_end));
  ITERATOR_IN in_it;
  ITERATOR_OUT out_it;
  const auto size = std::distance(in_begin, in_end);

  auto [first, second] = least_two_elements(in_begin, in_end);
  assert(first != in_end && second != in_end);
  const auto first_split = *first / size;
  const auto first_split2 = first_split * (size-2);
  const auto second_split = (*second - first_split2) / 2;

  for (in_it = in_begin, out_it = out_begin; in_it != in_end; ++in_it, ++out_it) {
    if (in_it == first)
      (*out_it) = { second_split, first_split };
    else if (in_it == second)
      (*out_it) = { first_split, second_split };
    else
      (*out_it) = { first_split, *in_it - second_split - first_split2 };
  }

#ifndef NDEBUG
  std::vector<std::array<cost, 2>> tmp;
  for (in_it = in_begin; in_it != in_end; ++in_it)
    tmp.push_back({0, *in_it});
  auto slow_version = uniform_minorant_generic(tmp);

  out_it = out_begin;
  for (auto& x : slow_version) {
    assert(std::abs(x[0] - (*out_it)[0]) < epsilon);
    assert(std::abs(x[1] - (*out_it)[1]) < epsilon);
    ++out_it;
  }
#endif
}

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
