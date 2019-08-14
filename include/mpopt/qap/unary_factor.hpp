#ifndef LIBMPOPT_QAP_UNARY_FACTOR_HPP
#define LIBMPOPT_QAP_UNARY_FACTOR_HPP

namespace mpopt {
namespace qap {

template<typename ALLOCATOR = std::allocator<cost>>
class unary_factor {
public:
  using allocator_type = ALLOCATOR;
  static constexpr cost initial_cost = std::numeric_limits<cost>::signaling_NaN();
  static constexpr index primal_unset = std::numeric_limits<index>::max();

  unary_factor(index number_of_labels, const ALLOCATOR& allocator = ALLOCATOR())
  : costs_(number_of_labels, initial_cost, allocator)
#ifndef NDEBUG
  , index_(-1)
#endif
  {
  }

  unary_factor(const unary_factor& other) = delete;
  unary_factor& operator=(const unary_factor& other) = delete;

#ifndef NDEBUG
  void set_debug_info(index idx) { index_ = idx; }

  std::string dbg_info() const
  {
    std::ostringstream s;
    s << "v(" << index_ << ")";
    return s.str();
  }
#endif

  auto size() const { return costs_.size(); }

  bool is_prepared() const
  {
    bool result = true;
    for (const auto& x : costs_)
        result = result && !std::isnan(x);
    return result;
  }

  void set(const index idx, cost c) { assert_index(idx); costs_[idx] = c; }
  cost get(const index idx) { assert_index(idx); return costs_[idx]; }

  cost normalize()
  {
    const auto lb = lower_bound();
    for (auto& x : costs_) x -= lb;
    return lb;
  }

  cost lower_bound() const
  {
    return *std::min_element(costs_.begin(), costs_.end());
  }

  void repam(const index idx, const cost msg)
  {
    assert_index(idx);
    costs_[idx] += msg;
  }

  void reset_primal() { primal_ = primal_unset; }

  cost evaluate_primal() const
  {
    if (primal_ != primal_unset)
      return costs_[primal_];
    else
      return std::numeric_limits<cost>::infinity();
  }

  auto& primal() { return primal_; }
  const auto& primal() const { return primal_; }

  void round_primal()
  {
    auto min = std::min_element(costs_.cbegin(), costs_.cend());
    primal_ = min - costs_.cbegin();
  }

protected:
  void assert_index(const index idx) const { assert(idx >= 0 && idx < costs_.size()); }

  fixed_vector_alloc_gen<cost, ALLOCATOR> costs_;
  index primal_;

#ifndef NDEBUG
  index index_;
#endif

  friend struct pairwise_messages;
  friend struct uniqueness_messages;
  template<typename> friend class gurobi_model_builder; // FIXME: Get rid of this.
  template<typename> friend class gurobi_unary_factor;
};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
