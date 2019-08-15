#ifndef LIBMPOPT_QAP_UNARY_FACTOR_HPP
#define LIBMPOPT_QAP_UNARY_FACTOR_HPP

namespace mpopt {
namespace qap {

template<typename ALLOCATOR>
class unary_factor : public ::mpopt::unary_factor<ALLOCATOR> {
public:
  using ::mpopt::unary_factor<ALLOCATOR>::unary_factor;

protected:
  friend struct uniqueness_messages;
};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
