#ifndef LIBCT_MESSAGE_CONFLICT_HPP
#define LIBCT_MESSAGE_CONFLICT_HPP

namespace ct {

struct conflict_messages {

  template<typename CONFLICT_NODE>
  static void send_messages_to_conflict(const CONFLICT_NODE& node)
  {
    index slot = 0;
    for (const auto& edge : node.detections) {
      auto& c = node.conflict;
      auto& d = edge.node->detection;

      const cost weight = 1.0d / (edge.node->conflicts.size() - edge.slot);
      const auto msg = d.min_detection() * weight;
      d.repam_detection(-msg);
      c.repam(slot, msg);
      ++slot;
    }
  }

  template<typename CONFLICT_NODE>
  static void send_messages_to_detection(const CONFLICT_NODE& node)
  {
    auto& c = node.conflict;
    auto [it1, it2] = least_two_elements(c.costs_.cbegin(), c.costs_.cend());
    const auto m = 0.5 * (*it1 + *it2);

    index slot = 0;
    for (const auto& edge : node.detections) {
      auto& d = edge.node->detection;

      const cost msg = c.costs_[slot] - m;
      c.repam(slot, -msg);
      d.repam_detection(msg);
      ++slot;
    }
  }

  template<typename CONFLICT_NODE>
  static consistency check_primal_consistency(const CONFLICT_NODE& node)
  {
    consistency result;

    index slot = 0;
    for (const auto& edge : node.detections) {
      const auto& c = node.conflict;
      const auto& d = edge.node->detection;

      if (c.primal_ < c.costs_.size() && !d.primal_.is_undecided()) {
        if (slot == c.primal_) {
          if (!d.primal_.is_detection_on())
            result.mark_inconsistent();
        } else {
          if (!d.primal_.is_detection_off())
            result.mark_inconsistent();
        }
      } else {
        result.mark_unknown();
      }
      ++slot;
    }

    return result;
  }

  template<typename CONFLICT_NODE>
  static void propagate_primal_to_conflict(const CONFLICT_NODE& node)
  {
    auto& c = node.conflict;
    assert(c.primal_ >= 0);

    index slot = 0;
    for (const auto& edge : node.detections) {
      const auto& d = edge.node->detection;

      if (d.primal_.is_detection_on()) {
        assert(c.primal_ >= c.size() || c.primal_ == slot);
        c.primal_ = slot;
      } else {
        assert(c.primal_ != slot);
      }
      ++slot;
    }
  }

  template<typename CONFLICT_NODE>
  static void propagate_primal_to_detections(const CONFLICT_NODE& node)
  {
    const auto& c = node.conflict;
    assert(c.primal_ >= 0);

    if (c.primal_ >= c.size())
      return;

    index slot = 0;
    for (const auto& edge : node.detections) {
      auto& d = edge.node->detection;

      if (slot != c.primal_)
        d.primal_.set_detection_off();
      ++slot;
    }
  }

};

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
