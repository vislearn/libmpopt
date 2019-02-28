#ifndef LIBCT_FACTOR_DETECTION_HPP
#define LIBCT_FACTOR_DETECTION_HPP

namespace ct {

class detection_primal {
public:
  static constexpr index undecided = std::numeric_limits<index>::max();
  static constexpr index off = undecided - 1;

  detection_primal()
  : incoming_(undecided)
  , outgoing_(undecided)
  { }

  detection_primal(const index no_incoming, const index no_outgoing)
  : incoming_(undecided)
  , outgoing_(undecided)
#ifndef NDEBUG
  , no_incoming_(no_incoming)
  , no_outgoing_(no_outgoing)
#endif
  { }

  void reset()
  {
    incoming_ = undecided;
    outgoing_ = undecided;
  }

  void set_incoming(const index incoming)
  {
    assert(incoming_ == undecided || incoming_ == incoming);
    assert(incoming >= 0 && incoming < no_incoming_);
    incoming_ = incoming;
    check_consistency();
  }

  void set_outgoing(const index outgoing)
  {
    assert(outgoing_ == undecided || outgoing_ == outgoing);
    assert(outgoing >= 0 && outgoing < no_outgoing_);
    outgoing_ = outgoing;
    check_consistency();
  }

  void set_detection_off()
  {
    assert(is_detection_off() || incoming_ == undecided && outgoing_ == undecided);
    incoming_ = off;
    outgoing_ = off;
    check_consistency();
  }

  bool is_undecided() const { check_consistency(); return incoming_ == undecided && outgoing_ == undecided; }
  bool is_incoming_set() const { check_consistency(); return incoming_ != undecided; }
  bool is_outgoing_set() const { check_consistency(); return outgoing_ != undecided; }
  bool is_detection_off() const { check_consistency(); return incoming_ == off && outgoing_ == off; }
  bool is_detection_on() const { return (is_incoming_set() || is_outgoing_set()) && !is_detection_off(); }

  index incoming() const { check_consistency(); return incoming_; }
  index outgoing() const { check_consistency(); return outgoing_; }

protected:
  void check_consistency() const
  {
    assert((incoming_ == off) == (outgoing_ == off));
    if (incoming_ != off && incoming_ != undecided)
      assert(incoming_ >= 0 && incoming_ < no_incoming_);
    if (outgoing_ != off && outgoing_ != undecided)
      assert(outgoing_ >= 0 && outgoing_ < no_outgoing_);
  }

  index incoming_;
  index outgoing_;

#ifndef NDEBUG
  index no_incoming_, no_outgoing_;
#endif
};

template<typename ALLOCATOR = std::allocator<cost>>
class detection_factor {
public:
  using allocator_type = typename std::allocator_traits<ALLOCATOR>::template rebind_alloc<cost>;
  static constexpr cost initial_cost = std::numeric_limits<cost>::signaling_NaN();

  detection_factor(index number_of_incoming, index number_of_outgoing, const ALLOCATOR& allocator = ALLOCATOR())
  : detection_(initial_cost)
  , incoming_(number_of_incoming + 1, initial_cost, allocator)
  , outgoing_(number_of_outgoing + 1, initial_cost, allocator)
  , primal_(incoming_.size(), outgoing_.size())
#ifndef NDEBUG
  , timestep_(-1)
  , index_(-1)
#endif
  {
  }

  detection_factor(const detection_factor& other) = delete;
  detection_factor& operator=(const detection_factor& other) = delete;

#ifndef NDEBUG
  void set_debug_info(index timestep, index idx)
  {
    timestep_ = timestep;
    index_ = idx;
  }
#endif

  //
  // cost getters
  //

  cost detection() const { return detection_; }
  cost appearance() const { return incoming_.back(); }
  cost disappearance() const { return outgoing_.back(); }
  cost incoming(const index idx) const { assert_incoming(idx); return incoming_[idx]; }
  cost outgoing(const index idx) const { assert_outgoing(idx); return outgoing_[idx]; }

  //
  // methods to initialize costs
  //

  void set_detection_cost(cost on) { detection_ = on; }
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
    return std::min(min_detection(), 0.0);
  }

  void repam_detection(const cost msg) { detection_ += msg; }
  void repam_incoming(const index idx, const cost msg) { assert_incoming(idx); incoming_[idx] += msg; }
  void repam_outgoing(const index idx, const cost msg) { assert_outgoing(idx); outgoing_[idx] += msg; }

  void reset_primal() { primal_.reset(); }

  void fix_primal()
  {
    assert(primal_.is_incoming_set() || primal_.is_outgoing_set());
    if (!primal_.is_incoming_set())
      primal_.set_incoming(incoming_.size() - 1);
    if (!primal_.is_outgoing_set())
      primal_.set_outgoing(outgoing_.size() - 1);
    assert(primal_.is_incoming_set() && primal_.is_outgoing_set());
  }

  template<bool from_left, typename CONTAINER>
  void round_primal(const CONTAINER& active)
  {
    if ( from_left && primal_.is_incoming_set() ||
        !from_left && primal_.is_outgoing_set())
      return;

    cost opposite_side = from_left ? min_outgoing()
                                   : min_incoming();

    auto this_side = from_left ? min_element(incoming_.cbegin(), incoming_.cend(), active.cbegin(), active.cend())
                               : min_element(outgoing_.cbegin(), outgoing_.cend(), active.cbegin(), active.cend());

    if (*this_side + detection_ + opposite_side <= 0 || primal_.is_detection_on()) {
      if constexpr (from_left)
        primal_.set_incoming(this_side - incoming_.cbegin());
      else
        primal_.set_outgoing(this_side - outgoing_.cbegin());
    } else {
      primal_.set_detection_off();
    }
  }

  cost evaluate_incoming_primal()
  {
    if (!primal_.is_incoming_set())
      return std::numeric_limits<cost>::infinity();

    if (primal_.incoming() == detection_primal::off)
      return 0.0;
    else
      return incoming_[primal_.incoming()];
  }

  cost evaluate_detection_primal()
  {
    if (primal_.is_detection_off())
      return 0.0;

    if (primal_.is_detection_on())
      return detection_;

    return std::numeric_limits<cost>::infinity();
  }

  cost evaluate_outgoing_primal()
  {
    if (!primal_.is_outgoing_set())
      return std::numeric_limits<cost>::infinity();

    if (primal_.outgoing() == detection_primal::off)
      return 0.0;
    else
      return outgoing_[primal_.outgoing()];
  }

  cost evaluate_primal()
  {
    cost result;
    if (primal_.is_detection_off())
      result = 0.0;
    else if (primal_.is_incoming_set() && primal_.is_outgoing_set())
      result = incoming_[primal_.incoming()] + detection_ + outgoing_[primal_.outgoing()];
    else
      result = std::numeric_limits<cost>::infinity();
    assert(dbg::are_idential(result, evaluate_incoming_primal() + evaluate_detection_primal() + evaluate_outgoing_primal()));
    return result;
  }

  auto& primal() { return primal_; }

protected:
  void assert_incoming(const index idx) const { assert(idx >= 0 && idx < incoming_.size() - 1); }
  void assert_outgoing(const index idx) const { assert(idx >= 0 && idx < outgoing_.size() - 1); }
  cost detection_;
  fixed_vector<cost, allocator_type> incoming_;
  fixed_vector<cost, allocator_type> outgoing_;
  detection_primal primal_;

#ifndef NDEBUG
  index timestep_, index_;
#endif

  template<typename> friend class transition_messages;
  template<typename> friend class conflict_messages;
};

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
