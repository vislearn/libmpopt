#ifndef LIBMPOPT_CT_CONFLICT_MESSAGES_HPP
#define LIBMPOPT_CT_CONFLICT_MESSAGES_HPP

namespace mpopt {
namespace ct {

struct conflict_messages {

  template<typename CONFLICT_NODE>
  static void send_messages_to_conflict(const CONFLICT_NODE* node)
  {
#ifndef NDEBUG
    const cost lb_before = local_lower_bound(node);
#endif

    auto& c = node->factor;
    node->traverse_detections([&](auto& edge, auto slot) {
      auto& d = edge.node->factor;
      const cost weight = 1.0 / (edge.node->conflicts.size() - edge.slot);
      const auto msg = d.min_detection() * weight;
      d.repam_detection(-msg);
      c.repam(slot, msg);
    });

#ifndef NDEBUG
    const cost lb_after = local_lower_bound(node);
    assert(lb_before <= lb_after + epsilon);
#endif
  }

  template<typename CONFLICT_NODE>
  static void send_messages_to_detection(const CONFLICT_NODE* node)
  {
#ifndef NDEBUG
    const cost lb_before = local_lower_bound(node);
#endif

    auto& c = node->factor;
    auto [first, second] = least_two_values(c.costs_.cbegin(), c.costs_.cend());
    const auto m = std::min(0.5 * (first + second), 0.0);

    node->traverse_detections([&](auto& edge, auto slot) {
      auto& d = edge.node->factor;
      const cost msg = c.costs_[slot] - m;
      c.repam(slot, -msg);
      d.repam_detection(msg);
    });

#ifndef NDEBUG
    const cost lb_after = local_lower_bound(node);
    assert(lb_before <= lb_after + epsilon);
#endif
  }

  template<typename CONFLICT_NODE>
  static consistency check_primal_consistency(const CONFLICT_NODE* node, index slot)
  {
    assert(slot >= 0 && slot < node->detections.size());
    consistency result;

    const auto& c = node->factor;
    const auto& d = node->detections[slot].node->factor;

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

    return result;
  }

  template<typename CONFLICT_NODE>
  static consistency check_primal_consistency(const CONFLICT_NODE* node)
  {
    consistency result;

    node->traverse_detections([&](auto& edge, auto slot) {
      result.merge(check_primal_consistency(node, slot));
    });

    return result;
  }

  template<typename CONFLICT_NODE>
  static void propagate_primal_to_conflict(const CONFLICT_NODE* node)
  {
    auto& c = node->factor;

    bool all_off = true;
    node->traverse_detections([&](auto& edge, auto slot) {
      const auto& d = edge.node->factor;

      if (d.primal().is_detection_on())
        c.primal().set(slot);
      else
        assert(c.primal().get() != slot);

      if (!d.primal().is_detection_off())
        all_off = false;
    });

    if (all_off)
      c.primal().set(c.size() - 1);
  }

  template<typename CONFLICT_NODE>
  static void propagate_primal_to_detections(const CONFLICT_NODE* node)
  {
    const auto& c = node->factor;

    if (c.primal().is_undecided())
      return;

    node->traverse_detections([&](auto& edge, auto slot) {
      auto& d = edge.node->factor;

      if (slot != c.primal().get())
        d.primal().set_detection_off();
    });
  }

private:

  template<typename CONFLICT_NODE>
  static cost local_lower_bound(const CONFLICT_NODE* node)
  {
    cost result = node->factor.lower_bound();
    for (const auto& edge: node->detections)
      result += edge.node->factor.lower_bound();
    return result;
  }

};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
