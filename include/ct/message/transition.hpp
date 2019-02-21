#ifndef LIBCT_MESSAGE_TRANSITION_HPP
#define LIBCT_MESSAGE_TRANSITION_HPP

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

  void set_left_transition(index index_this, detection_type* other1, index index_other1, detection_type* other2 = nullptr, index index_other2 = 0)
  {
    set_transition(left_, index_this, other1, index_other1, other2, index_other2);
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
      if (edge.is_division() && to_right) {
        (edge.detection1->*repam_other)(edge.index1, .5 * msg);
        (edge.detection2->*repam_other)(edge.index2, .5 * msg);
      } else {
        (edge.detection1->*repam_other)(edge.index1, msg);
      }
      ++idx;
    }
  }

  bool is_primal_consistent() const
  {
    bool result = true;
    index idx;

    if (!detection_->primal_.is_incoming_set() || !detection_->primal_.is_outgoing_set())
      result &= false;

    idx = 0;
    for (auto& edge : left_) {
      result &= (detection_->primal_.incoming() == idx) == (edge.detection1->primal_.outgoing() == edge.index1);
      //assert(edge.detection2 == nullptr);
      ++idx;
    }

    idx = 0;
    for (auto& edge : right_) {
      result &= (detection_->primal_.outgoing() == idx) == (edge.detection1->primal_.incoming() == edge.index1);
      if (edge.detection2 != nullptr)
        result &= (detection_->primal_.outgoing() == idx) == (edge.detection2->primal_.incoming() == edge.index2);
      ++idx;
    }

    return result;
  }

  template<bool to_right>
  void propagate_primal()
  {
    if (detection_->primal_.is_detection_off())
      return;

    if constexpr (to_right) {
      assert(detection_->primal_.is_outgoing_set());
      if (detection_->primal_.outgoing() < detection_->outgoing_.size() - 1) {
        auto& edge = right_[detection_->primal_.outgoing()];

        edge.detection1->primal_.set_incoming(edge.index1);
        if (edge.detection2 != nullptr)
          edge.detection2->primal_.set_incoming(edge.index2);
      }
    } else {
      assert(detection_->primal_.is_incoming_set());
      if (detection_->primal_.incoming() < detection_->incoming_.size() - 1){
        auto& edge = left_[detection_->primal_.incoming()];

        edge.detection1->primal_.set_outgoing(edge.index1);
        if (edge.detection2 != nullptr)
          edge.detection2->primal_.set_incoming(edge.index2);
      }
    }
  }

  template<bool from_left, typename CONTAINER>
  void get_primal_possibilities(CONTAINER& out)
  {
    out.fill(true);

    auto get_edges = [&]() -> auto& {
      if constexpr (from_left)
        return left_;
      else
        return right_;
    };

    auto get_primal = [&](const auto* factor) {
      if constexpr (from_left)
        return factor->primal_.outgoing();
      else
        return factor->primal_.incoming();
    };

    auto get_primal2 = [&](const auto* factor) {
      if constexpr (from_left)
        return factor->primal_.incoming();
      else
        return factor->primal_.outgoing();
    };

    auto it = out.begin();
    for (auto& edge : get_edges()) {
      assert(it != out.end());

      auto helper = [&](const auto* factor, auto index, auto primal_getter) {
        const auto p = primal_getter(factor);
        if (p != detection_primal::undecided && p != index)
          *it = false;

        if (p == index) {
          bool current = *it;
          out.fill(false);
          *it = current;
        }
      };

      helper(edge.detection1, edge.index1, get_primal);
      if (edge.detection2 != nullptr)
        if constexpr (from_left)
          helper(edge.detection2, edge.index2, get_primal2);
        else
          helper(edge.detection2, edge.index2, get_primal);
      ++it;
    }

    assert(std::find(out.cbegin(), out.cend(), true) != out.cend());
  }

protected:
  template<typename CONTAINER>
  static void set_transition(CONTAINER& container, index index_this, detection_type* other1, index index_other1, detection_type* other2 = nullptr, index index_other2 = 0)
  {
    auto& edge = container[index_this];
    assert(edge.detection1 == nullptr);
    assert(edge.detection2 == nullptr);

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
