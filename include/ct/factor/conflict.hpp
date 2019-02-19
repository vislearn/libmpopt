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

protected:
  void assert_index(const index idx) const { assert(idx >= 0 && idx < costs_.size() - 1); }

  fixed_vector<cost, allocator_type> costs_;

  template<typename> friend class conflict_messages;
};

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
