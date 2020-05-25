#ifndef LIBMPOPT_QAP_UNIQUENESS_FACTOR_HPP
#define LIBMPOPT_QAP_UNIQUENESS_FACTOR_HPP

namespace mpopt {
namespace qap {

// A uniqueness factor expresses the same simplex constraint as a normal
// unary factor. We reuse most of the functionality, but still provide
// a dedicated type.

template<typename ALLOCATOR = std::allocator<cost>>
class uniqueness_factor : public ::mpopt::unary_factor<ALLOCATOR> {
public:
  static constexpr cost initial_cost = 0;

  uniqueness_factor(index number_of_labels, const ALLOCATOR& allocator = ALLOCATOR())
  : ::mpopt::unary_factor<ALLOCATOR>::unary_factor(number_of_labels + 1, allocator)
  {
    // FIXME: We can't initialize members of the base class in a derived class,
    // so we just assign the costs here again. We could circumvent this with a
    // protected constructor interface in the base class.
    std::fill(this->costs_.begin(), this->costs_.end(), 0);
  }

  index size() const { return this->costs_.size() - 1; }

  uniqueness_factor(const uniqueness_factor& other) = delete;
  uniqueness_factor& operator=(const uniqueness_factor& other) = delete;

#ifndef NDEBUG
  std::string dbg_info() const
  {
    std::ostringstream s;
    s << "=1(" << this->index_ << ")";
    return s.str();
  }
#endif

  friend struct uniqueness_messages;
};


#ifdef ENABLE_GUROBI

template<typename ALLOCATOR>
class gurobi_uniqueness_factor : public ::mpopt::gurobi_unary_factor<ALLOCATOR> {
public:
  using allocator_type = ALLOCATOR;
  using factor_type = uniqueness_factor<allocator_type>;

  gurobi_uniqueness_factor(factor_type& factor, GRBModel& model)
  : ::mpopt::gurobi_unary_factor<ALLOCATOR>(factor, model)
  { }
};

#endif

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
