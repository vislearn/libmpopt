#ifndef LIBCT_MESSAGE_CONFLICT_HPP
#define LIBCT_MESSAGE_CONFLICT_HPP

namespace ct {

struct conflict_messages {

#ifndef NDEBUG
  template<typename CONFLICT_NODE>
  static cost local_lower_bound(const CONFLICT_NODE& node)
  {
    cost result = node.conflict.lower_bound();
    for (const auto& edge: node.detections)
      result += edge.node->detection.lower_bound();
    return result;
  }
#endif

  template<typename CONFLICT_NODE>
  static void send_messages_to_conflict(const CONFLICT_NODE& node)
  {
#ifndef NDEBUG
    const cost lb_before = local_lower_bound(node);
#endif

    auto& c = node.conflict;
    index slot = 0;
    for (const auto& edge : node.detections) {
      auto& d = edge.node->detection;

      const cost weight = 1.0d / (edge.node->conflicts.size() - edge.slot);
      const auto msg = d.min_detection() * weight;
      d.repam_detection(-msg);
      c.repam(slot, msg);
      ++slot;
    }

#ifndef NDEBUG
    const cost lb_after = local_lower_bound(node);
    assert(lb_before <= lb_after + epsilon);
#endif
  }

  template<typename CONFLICT_NODE>
  static void send_messages_to_detection(const CONFLICT_NODE& node)
  {
#ifndef NDEBUG
    const cost lb_before = local_lower_bound(node);
#endif

    auto& c = node.conflict;
    auto [it1, it2] = least_two_elements(c.costs_.cbegin(), c.costs_.cend());
    const auto m = std::min(0.5 * (*it1 + *it2), 0.0);

    index slot = 0;
    for (const auto& edge : node.detections) {
      auto& d = edge.node->detection;

      const cost msg = c.costs_[slot] - m;
      c.repam(slot, -msg);
      d.repam_detection(msg);
      ++slot;
    }

#ifndef NDEBUG
    const cost lb_after = local_lower_bound(node);
    assert(lb_before <= lb_after + epsilon);
#endif
  }

  template<typename CONFLICT_NODE>
  static consistency check_primal_consistency(const CONFLICT_NODE& node)
  {
    consistency result;

    index slot = 0;
    for (const auto& edge : node.detections) {
      const auto& c = node.conflict;
      const auto& d = edge.node->detection;

      if (c.primal().is_set() && !d.primal().is_undecided()) {
        if (slot == c.primal().get()) {
          if (!d.primal().is_detection_on())
            result.mark_inconsistent();
        } else {
          if (!d.primal().is_detection_off())
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

    bool all_off = true;
    index slot = 0;
    for (const auto& edge : node.detections) {
      const auto& d = edge.node->detection;

      if (d.primal_.is_detection_on())
        c.primal_.set(slot);
      else
        assert(c.primal_.get() != slot);

      if (!d.primal_.is_detection_off())
        all_off = false;

      ++slot;
    }

    if (all_off)
      c.primal_.set(c.size() - 1);
  }

  template<typename CONFLICT_NODE>
  static void propagate_primal_to_detections(const CONFLICT_NODE& node)
  {
    const auto& c = node.conflict;

    if (c.primal_.is_undecided())
      return;

    index slot = 0;
    for (const auto& edge : node.detections) {
      auto& d = edge.node->detection;

      if (slot != c.primal_.get())
        d.primal_.set_detection_off();
      ++slot;
    }
  }

};

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
