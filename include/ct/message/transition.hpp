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

  void send_messages_to_right()
  {
    const auto constant = detection_->detection() + detection_->min_incoming();
    const auto [first_minimum, second_minimum] = least_two_values(detection_->outgoing_.begin(), detection_->outgoing_.end() - 1);

    const auto set_to = std::min(constant + std::min(second_minimum, detection_->disappearance()), detection_->detection_off());

    index idx = 0;
    for (auto& edge : right_) {
      auto msg = constant + detection_->outgoing(idx) - set_to;
      detection_->repam_outgoing(idx, -msg);
      if (edge.is_division()) {
        edge.detection1->repam_incoming(edge.index1, .5 * msg);
        edge.detection2->repam_incoming(edge.index2, .5 * msg);
      } else {
        edge.detection1->repam_incoming(edge.index1, msg);
      }
      ++idx;
    }
  }

  void send_messages_to_left()
  {
    const auto constant = detection_->detection() + detection_->min_outgoing();
    const auto [first_minimum, second_minimum] = least_two_values(detection_->incoming_.begin(), detection_->incoming_.end() - 1);

    const auto set_to = std::min(constant + std::min(second_minimum, detection_->appearance()), detection_->detection_off());

    index idx = 0;
    for (auto& edge : left_) {
      auto msg = constant + detection_->incoming(idx) - set_to;
      assert(!edge.is_division());
      detection_->repam_incoming(idx, -msg);
      edge.detection1->repam_outgoing(edge.index1, msg);
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
