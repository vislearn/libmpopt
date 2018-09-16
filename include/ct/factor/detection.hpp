#ifndef LIBCT_FACTOR_DETECTION_HPP
#define LIBCT_FACTOR_DETECTION_HPP

namespace ct {

template<typename ALLOCATOR = std::allocator<cost>>
class detection_factor {
public:
  using allocator_type = ALLOCATOR;
  static constexpr cost initial_cost = std::numeric_limits<cost>::signaling_NaN();

  detection_factor(index number_of_incoming, index number_of_outgoing, const ALLOCATOR& allocator = ALLOCATOR())
  : detection_(initial_cost)
  , detection_off_(initial_cost)
  , incoming_(number_of_incoming + 1, initial_cost, allocator)
  , outgoing_(number_of_outgoing + 1, initial_cost, allocator)
  {
  }

  detection_factor(const detection_factor& other) = delete;
  detection_factor& operator=(const detection_factor& other) = delete;

  //
  // cost getters
  //

  cost detection() const { return detection_; }
  cost detection_off() const { return detection_off_; }
  cost appearance() const { return incoming_.back(); }
  cost disappearance() const { return outgoing_.back(); }
  cost incoming(const index idx) const { assert_incoming(idx); return incoming_[idx]; }
  cost outgoing(const index idx) const { assert_outgoing(idx); return outgoing_[idx]; }

  //
  // methods to initialize costs
  //

  void set_detection_cost(cost on, cost off = 0.0) { detection_ = on; detection_off_ = off; }
  void set_appearance_cost(cost c) { incoming_.back() = c; }
  void set_disappearance_cost(cost c) { outgoing_.back() = c; }
  void set_incoming_cost(index idx, cost c) { assert_incoming(idx); incoming_[idx] = c; }
  void set_outgoing_cost(index idx, cost c) { assert_outgoing(idx); outgoing_[idx] = c; }

  bool is_prepared() const { return true; } // FIXME: Check if costs have been initialized.

  //
  // factor specific logic
  //

  cost min_incoming() const
  {
    assert(incoming_.size() > 0);
    return *std::min_element(incoming_.begin(), incoming_.end());
  }

  cost min_outgoing() const
  {
    assert(outgoing_.size() > 0);
    return *std::min_element(outgoing_.begin(), outgoing_.end());
  }

  cost min_detection() const
  {
    return detection_ + min_incoming() + min_outgoing();
  }

  cost lower_bound() const
  {
    return std::min(min_detection(), detection_off_);
  }

  void repam_detection_on(const cost msg) { detection_ += msg; }
  void repam_detection_off(const cost msg) { detection_off_ += msg; }
  void repam_incoming(const index idx, const cost msg) { assert_incoming(idx); incoming_[idx] += msg; }
  void repam_outgoing(const index idx, const cost msg) { assert_outgoing(idx); outgoing_[idx] += msg; }

protected:
  void assert_incoming(const index idx) const { assert(idx >= 0 && idx < incoming_.size() - 1); }
  void assert_outgoing(const index idx) const { assert(idx >= 0 && idx < outgoing_.size() - 1); }
  cost detection_, detection_off_;
  fixed_vector<cost, ALLOCATOR> incoming_;
  fixed_vector<cost, ALLOCATOR> outgoing_;

  template<typename> friend class transition_messages;
};

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
