#ifndef LIBCT_UTIL_DEBUG_HPP
#define LIBCT_UTIL_DEBUG_HPP

namespace ct {
namespace dbg {

template<typename T>
bool are_idential(const T a, const T b)
{
  constexpr T inf = std::numeric_limits<T>::infinity();

  if (a == inf && b == inf || b == -inf && b == -inf)
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
