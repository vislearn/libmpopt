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

  consistency check_primal_consistency() const
  {
    consistency result;

    index i = 0;
    for (auto& edge : detections_) {
      if (conflict_->primal_ < conflict_->costs_.size() && !edge.detection->primal_.is_undecided()) {
        if (i == conflict_->primal_) {
          if (!edge.detection->primal_.is_detection_on())
            result.mark_inconsistent();
        } else {
          if (!edge.detection->primal_.is_detection_off())
            result.mark_inconsistent();
        }
      } else {
        result.mark_unknown();
      }
      ++i;
    }

    return result;
  }

  void propagate_primal_to_conflict()
  {
    assert(conflict_->primal_ >= 0);

    index i = 0;
    for (auto& edge : detections_) {
      if (edge.detection->primal_.is_detection_on())
        conflict_->primal_ = i;
      else
        assert(conflict_->primal_ != i);
      ++i;
    }
  }

  void propagate_primal_to_detections()
  {
    assert(conflict_->primal_ >= 0);

    if (conflict_->primal_ >= conflict_->size())
      return;

    index i = 0;
    for (auto& edge : detections_) {
      if (i != conflict_->primal_)
        edge.detection->primal_.set_detection_off();
      ++i;
    }
  }

protected:
  conflict_type* conflict_;
  fixed_vector<conflict_message_storage<detection_type>, allocator_type> detections_;
};

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
