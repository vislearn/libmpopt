#ifndef LIBCT_MESSAGE_CONFLICT_HPP
#define LIBCT_MESSAGE_CONFLICT_HPP

namespace ct {

template<typename ALLOCATOR = std::allocator<cost>>
class conflict_messages {
public:
  using conflict_type = conflict_factor<ALLOCATOR>;
  using detection_type = detection_factor<ALLOCATOR>;
  using allocator_type = typename std::allocator_traits<ALLOCATOR>::template rebind_alloc<detection_type*>;

  static constexpr cost weight = 0.05;

  conflict_messages(conflict_type* conflict, const index number_of_detections, const ALLOCATOR& allocator = ALLOCATOR())
  : conflict_(conflict)
  , detections_(number_of_detections, nullptr, allocator)
  { }

  void add_link(const index slot, detection_type* detection)
  {
    assert(detections_[slot] == nullptr);
    detections_[slot] = detection;
  }

  bool is_prepared() const
  {
    for (auto* detection : detections_)
      if (detection == nullptr)
        return false;

    return true;
  }

  void send_message_to_conflict()
  {
    index i = 0;
    for (auto* detection : detections_) {
      const auto msg_on = detection->min_detection() * weight;
      detection->repam_detection_on(-msg_on);
      conflict_->repam_on(i, msg_on);

      const auto msg_off = detection->detection_off() * weight;
      detection->repam_detection_off(-msg_off);
      conflict_->repam_off(i, msg_off);

      ++i;
    }

  }

  void send_message_to_detection()
  {
    std::array<std::array<cost, 2>, max_number_of_conflict_edges> minorant;
    assert(conflict_->costs_.size() <= minorant.size());
    uniform_minorant(conflict_->costs_.cbegin(), conflict_->costs_.cend(), minorant.begin(), minorant.end());

    index i = 0;
    for (auto* detection : detections_) {
      const cost msg_on = minorant[i][1] * weight;
      conflict_->repam_on(i, -msg_on);
      detection->repam_detection_on(msg_on);

      const cost msg_off = minorant[i][0] * weight;
      conflict_->repam_off(i, -msg_off);
      detection->repam_detection_off(msg_off);

      ++i;
    }
  }

protected:
  conflict_type* conflict_;
  fixed_vector<detection_type*, allocator_type> detections_;
};

}

#endif
