#ifndef LIBCT_FACTOR_CONFLICT_HPP
#define LIBCT_FACTOR_CONFLICT_HPP

namespace ct {

using conflict_primal = index;

template<typename ALLOCATOR = std::allocator<cost>>
class conflict_factor {
public:
  using allocator_type = ALLOCATOR;
  static constexpr cost initial_cost = 0.0;

  conflict_factor(index number_of_detections, const ALLOCATOR& allocator = ALLOCATOR())
  : costs_(number_of_detections + 1, initial_cost, allocator)
#ifndef NDEBUG
  , timestep_(-1)
  , index_(-1)
#endif
  {
  }

  conflict_factor(const conflict_factor& other) = delete;
  conflict_factor& operator=(const conflict_factor& other) = delete;

#ifndef NDEBUG
  void set_debug_info(index timestep, index idx)
  {
    timestep_ = timestep;
    index_ = idx;
  }

  std::string dbg_info() const
  {
    std::ostringstream s;
    s << "c(" << timestep_ << ", " << index_ << ")";
    return s.str();
  }
#endif

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
  const auto& primal() const { return primal_; }

  void round_primal()
  {
    if (primal_ >= costs_.size()) {
      auto min = std::min_element(costs_.cbegin(), costs_.cend());
      primal_ = min - costs_.cbegin();
      assert(primal_ >= 0 && primal_ < costs_.size());
    }
  }

protected:
  void assert_index(const index idx) const { assert(idx >= 0 && idx < costs_.size() - 1); }

  fixed_vector_alloc_gen<cost, ALLOCATOR> costs_;
  conflict_primal primal_;

#ifndef NDEBUG
  index timestep_, index_;
#endif

  friend struct transition_messages;
  friend struct conflict_messages;
};

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
