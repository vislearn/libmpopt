#ifndef LIBCT_MESSAGE_CONFLICT_HPP
#define LIBCT_MESSAGE_CONFLICT_HPP

namespace ct {

template<typename DETECTION>
struct conflict_message_storage {
  conflict_message_storage()
  : detection(nullptr)
  , weight(-1)
  { }

  DETECTION* detection;
  cost weight;

  bool is_prepared() const
  {
    return detection != nullptr && weight >= 0 && weight <= 1;
  }
};

template<typename ALLOCATOR = std::allocator<cost>>
class conflict_messages {
public:
  using conflict_type = conflict_factor<ALLOCATOR>;
  using detection_type = detection_factor<ALLOCATOR>;
  using allocator_type = typename std::allocator_traits<ALLOCATOR>::template rebind_alloc<conflict_message_storage<detection_type>>;

  conflict_messages(conflict_type* conflict, const index number_of_detections, const ALLOCATOR& allocator = ALLOCATOR())
  : conflict_(conflict)
  , detections_(number_of_detections, allocator)
  { }

  void add_link(const index slot, detection_type* detection, const cost weight)
  {
    assert(!detections_[slot].is_prepared());
    detections_[slot].detection = detection;
    detections_[slot].weight = weight;
  }

  bool is_prepared() const
  {
    for (auto& d : detections_)
      if (! d.is_prepared())
        return false;

    return true;
  }

  void send_message_to_conflict()
  {
    index i = 0;
    for (auto& d : detections_) {
      const auto msg = d.detection->min_detection() * d.weight;
      d.detection->repam_detection(-msg);
      conflict_->repam(i, msg);

      ++i;
    }
  }

  void send_message_to_detection()
  {
    std::array<std::array<cost, 2>, max_number_of_conflict_edges> minorant;
    assert(conflict_->costs_.size() <= minorant.size());
    uniform_minorant(conflict_->costs_.cbegin(), conflict_->costs_.cend(), minorant.begin(), minorant.end());

    index i = 0;
    for (auto& d : detections_) {
      const cost msg = minorant[i][1] - minorant[i][0];
      conflict_->repam(i, -msg);
      d.detection->repam_detection(msg);

      ++i;
    }
  }

protected:
  conflict_type* conflict_;
  fixed_vector<conflict_message_storage<detection_type>, allocator_type> detections_;
};

}

#endif
