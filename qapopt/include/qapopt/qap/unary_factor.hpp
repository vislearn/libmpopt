#ifndef LIBQAPOPT_QAP_UNARY_FACTOR_HPP
#define LIBQAPOPT_QAP_UNARY_FACTOR_HPP

namespace qapopt {
namespace qap {

template<typename ALLOCATOR>
class unary_factor : public ::qapopt::unary_factor<ALLOCATOR> {
public:
  using ::qapopt::unary_factor<ALLOCATOR>::unary_factor;

protected:
  friend struct uniqueness_messages;
};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
