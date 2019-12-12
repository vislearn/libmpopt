#ifndef LIBMPOPT_COMMON_DEBUG_HPP
#define LIBMPOPT_COMMON_DEBUG_HPP

namespace mpopt {
namespace dbg {

template<typename...> struct get_type_of;

struct timer {
  using clock_type = std::chrono::steady_clock;
  static_assert(clock_type::is_steady);

  timer(bool auto_start=true)
  {
    if (auto_start)
      start();
  }

  void start() { begin = clock_type::now(); }
  void stop() { end = clock_type::now(); }

  auto duration() const { return end - begin; }

  template<typename DURATION>
  auto duration_count() const
  {
    return std::chrono::duration_cast<DURATION>(duration()).count();
  }

  auto milliseconds() const { return duration_count<std::chrono::milliseconds>(); }
  auto seconds() const { return duration_count<std::chrono::seconds>(); }

  clock_type::time_point begin, end;
};


template<typename T>
bool are_identical(const T a, const T b)
{
  constexpr T inf = std::numeric_limits<T>::infinity();

  if ((a == inf && b == inf) || (b == -inf && b == -inf))
    return true;

  return std::abs(a - b) < epsilon;
}


template<typename ITERATOR>
struct print_iterator_helper {
  print_iterator_helper(ITERATOR begin, ITERATOR end)
  : begin(begin)
  , end(end)
  { }

  ITERATOR begin;
  ITERATOR end;
};

template<typename ITERATOR>
std::ostream& operator<<(std::ostream& o, const print_iterator_helper<ITERATOR>& pi) {
  bool first = true;
  o << "[";
  for (ITERATOR it = pi.begin; it != pi.end; ++it) {
    if (!first)
      o << ", ";
    o << *it;
    first = false;
  }
  o << "]";
  return o;
}

template<typename ITERATOR>
auto print_iterator(ITERATOR begin, ITERATOR end)
{
  return print_iterator_helper<ITERATOR>(begin, end);
}

template<typename CONTAINER>
auto print_container(const CONTAINER& container)
{
  return print_iterator_helper<typename CONTAINER::const_iterator>(container.begin(), container.end());
}

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
