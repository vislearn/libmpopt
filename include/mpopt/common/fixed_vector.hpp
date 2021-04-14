#ifndef LIBMPOPT_COMMON_FIXED_VECTOR_HPP
#define LIBMPOPT_COMMON_FIXED_VECTOR_HPP

namespace mpopt {

template<typename T, typename ALLOCATOR = std::allocator<T>>
class fixed_vector : protected std::vector<T, ALLOCATOR> {
protected:
  using base = std::vector<T, ALLOCATOR>;
public:
  using typename base::iterator;
  using typename base::const_iterator;

  using base::base;
  using base::back;
  using base::begin;
  using base::cbegin;
  using base::cend;
  using base::end;
  using base::front;
  using base::operator[];
  using base::rbegin;
  using base::rend;
  using base::size;
};

template<typename T, typename ALLOCATOR>
using fixed_vector_alloc_gen = fixed_vector<T, typename std::allocator_traits<ALLOCATOR>::template rebind_alloc<T>>;

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
