#ifndef LIBCT_FACTOR_CONFLICT_HPP
#define LIBCT_FACTOR_CONFLICT_HPP

namespace ct {

template<typename ALLOCATOR = std::allocator<cost>>
class conflict_factor {
public:
  using allocator_type = typename std::allocator_traits<ALLOCATOR>::template rebind_alloc<cost>;
  static constexpr cost initial_cost = 0.0;

  conflict_factor(index number_of_detections, const ALLOCATOR& allocator = ALLOCATOR())
  : costs_(number_of_detections + 1, initial_cost, allocator)
  {
  }

  conflict_factor(const conflict_factor& other) = delete;
  conflict_factor& operator=(const conflict_factor& other) = delete;

  auto size() const { return costs_.size(); }

  bool is_prepared() const { return true; }

  void set(const index idx, cost c) { assert_index(idx); costs_[idx] = c; }
  cost get(const index idx) { assert_index(idx); return costs_[idx]; }

  cost lower_bound() const
  {
    assert(costs_.size() > 0);
    return *std::min_element(costs_.begin(), costs_.end());
  }

  void repam(const index idx, const cost msg)
  {
    assert_index(idx);
    costs_[idx] += msg;
  }

  void reset_primal() { primal_ = std::numeric_limits<decltype(primal_)>::max(); }

  cost evaluate_primal() const
  {
    assert(primal_ >= 0);
    if (primal_ < costs_.size())
      return costs_[primal_];
    else
      return std::numeric_limits<cost>::infinity();
  }

  auto& primal() { return primal_; }

  void round_primal()
  {
    if (primal_ >= costs_.size()) {
      auto min = std::min_element(costs_.cbegin(), costs_.cend());
      primal_ = min - costs_.cbegin();
      assert(primal_ >= 0 && primal_ < costs_.size());
    }
  }

  void fix_primal()
  {
    if (primal_ >= costs_.size())
      primal_ = costs_.size() - 1;
  }

protected:
  void assert_index(const index idx) const { assert(idx >= 0 && idx < costs_.size() - 1); }

  fixed_vector<cost, allocator_type> costs_;
  index primal_;

  template<typename> friend class transition_messages;
  template<typename> friend class conflict_messages;
};

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
