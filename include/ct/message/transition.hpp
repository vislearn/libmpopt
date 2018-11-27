#ifndef LIBCT_TRANSITION_HPP
#define LIBCT_TRANSITION_HPP

namespace ct {

// FIXME: For non-divisions we waste some space here.
// FIXME: Left transitions can never be divisions, so we waste always space. Get rid of this!
template<typename DETECTION>
struct transition_storage {
  transition_storage()
  : detection1(nullptr)
  , detection2(nullptr)
  { }

  DETECTION* detection1;
  DETECTION* detection2;
  index index1, index2;

  bool is_prepared() const
  {
    if (detection2 != nullptr && detection1 == nullptr)
      return false;

    return detection1 != nullptr;
  }

  bool is_division() const
  {
    assert(detection1 != nullptr);
    return detection2 != nullptr;
  }
};

template<typename ALLOCATOR = std::allocator<cost>>
class transition_messages {
public:
  using detection_type = detection_factor<ALLOCATOR>;
  using allocator_type = typename std::allocator_traits<ALLOCATOR>::template rebind_alloc<transition_storage<detection_type>>;

  transition_messages(detection_type* detection, const index number_of_left, const index number_of_right, const ALLOCATOR& allocator)
  : detection_(detection)
  , left_(number_of_left, allocator)
  , right_(number_of_right, allocator)
  { }

  void set_left_transition(index index_this, detection_type* other1, index index_other1)
  {
    set_transition(left_, index_this, other1, index_other1);
  }

  void set_right_transition(index index_this, detection_type* other1, index index_other1, detection_type* other2 = nullptr, index index_other2 = 0)
  {
    set_transition(right_, index_this, other1, index_other1, other2, index_other2);
  }

  bool is_prepared() const
  {
    auto helper = [](auto& vec) {
      for (auto& x : vec)
        if (!x.is_prepared())
          return false;
      return true;
    };

    return helper(left_) && helper(right_);
  }

  template<bool to_right>
  void send_messages()
  {
    const auto min_other_side   = to_right ? detection_->min_incoming()
                                           : detection_->min_outgoing();
    const auto& costs_this_side = to_right ? detection_->outgoing_
                                           : detection_->incoming_;
    const auto cost_nirvana     = to_right ? detection_->disappearance()
                                           : detection_->appearance();

    const auto constant = detection_->detection() + min_other_side;
    const auto [first_minimum, second_minimum] = least_two_values(costs_this_side.begin(), costs_this_side.end() - 1);

    const auto set_to = std::min(constant + std::min(second_minimum, cost_nirvana), 0.0);

    index idx = 0;
    for (auto& edge : (to_right ? right_ : left_)) {
      const auto slot_cost   = to_right ? detection_->outgoing(idx)
                                        : detection_->incoming(idx);
      const auto repam_this  = to_right ? &detection_type::repam_outgoing
                                        : &detection_type::repam_incoming;
      const auto repam_other = to_right ? &detection_type::repam_incoming
                                        : &detection_type::repam_outgoing;

      auto msg = constant + slot_cost - set_to;
      (detection_->*repam_this)(idx, -msg);
      if (edge.is_division()) {
        (edge.detection1->*repam_other)(edge.index1, .5 * msg);
        (edge.detection2->*repam_other)(edge.index2, .5 * msg);
      } else {
        (edge.detection1->*repam_other)(edge.index1, msg);
      }
      ++idx;
    }
  }

protected:
  template<typename CONTAINER>
  static void set_transition(CONTAINER& container, index index_this, detection_type* other1, index index_other1, detection_type* other2 = nullptr, index index_other2 = 0)
  {
    auto& edge = container[index_this];
    assert(edge.detection1 == nullptr);

    edge.detection1 = other1;
    edge.index1 = index_other1;

    edge.detection2 = other2;
    edge.index2 = index_other2;
  }

  detection_type* detection_; // FIXME: Get rid of this pointer.
  fixed_vector<transition_storage<detection_type>, allocator_type> left_, right_;
};

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
