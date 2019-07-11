#ifndef LIBCT_MESSAGE_TRANSITION_HPP
#define LIBCT_MESSAGE_TRANSITION_HPP

namespace ct {

struct transition_messages {

  template<bool to_right, typename DETECTION_NODE>
  static void send_messages(const DETECTION_NODE& node)
  {
    auto& here = node.detection;
    using detection_type = typename DETECTION_NODE::detection_type;

#ifndef NDEBUG
    auto local_lower_bound = [&](const auto& edge) {
      cost result = here.lower_bound();
      result += edge.node1->detection.lower_bound();
      if (edge.is_division() && to_right)
        result += edge.node2->detection.lower_bound();
      return result;
    };
#endif

    const auto min_other_side   = to_right ? here.min_incoming()
                                           : here.min_outgoing();
    const auto& costs_this_side = to_right ? here.outgoing_
                                           : here.incoming_;
    const auto cost_nirvana     = to_right ? here.disappearance()
                                           : here.appearance();

    const auto constant = here.detection() + min_other_side;
    const auto [first_minimum, second_minimum] = least_two_values(costs_this_side.begin(), costs_this_side.end() - 1);

    const auto real_second_minimum = std::min(second_minimum, cost_nirvana);
    const auto set_to = std::min(constant + (first_minimum + real_second_minimum) * 0.5, 0.0);

    node.template traverse_transitions<to_right>([&](auto& edge, auto slot) {
      const auto slot_cost   = to_right ? here.outgoing(slot)
                                        : here.incoming(slot);
      const auto repam_this  = to_right ? &detection_type::repam_outgoing
                                        : &detection_type::repam_incoming;
      const auto repam_other = to_right ? &detection_type::repam_incoming
                                        : &detection_type::repam_outgoing;

#ifndef NDEBUG
      const cost lb_before = local_lower_bound(edge);
#endif
      auto msg = constant + slot_cost - set_to;
      (here.*repam_this)(slot, -msg);
      if (edge.is_division() && to_right) {
        (edge.node1->detection.*repam_other)(edge.slot1, .5 * msg);
        (edge.node2->detection.*repam_other)(edge.slot2, .5 * msg);
      } else {
        (edge.node1->detection.*repam_other)(edge.slot1, msg);
      }

#ifndef NDEBUG
      const auto lb_after = local_lower_bound(edge);
      assert(lb_before <= lb_after + epsilon);
#endif
    });
  }

  template<bool to_right, typename DETECTION_NODE>
  static consistency check_primal_consistency_impl(const DETECTION_NODE& node, index slot)
  {
    consistency result;
    const auto& here = node.detection;

    if (!here.primal().template is_transition_set<to_right>()) {
      result.mark_unknown();
      return result;
    }

    const auto p = here.primal().template transition<to_right>();
    assert(slot >= 0 && slot < node.template transitions<to_right>().size());
    const auto& edge = node.template transitions<to_right>()[slot];

    // We do not mark the current node as `unknown` if the other side is
    // currently unset. This mirrors the implementation for the conflict
    // consistency.

    const auto& there1 = edge.node1->detection;
    if (there1.primal().template is_transition_set<!to_right>()) {
      if ((p == slot) != (there1.primal().template transition<!to_right>() == edge.slot1))
        result.mark_inconsistent();
    } else {
      result.mark_unknown();
    }

    // The following looks weird, but is in fact correct. The state of
    // `to_right` does not matter, we always check the `incoming` arc of
    // the second connected factor. With `to_right == true` this is the
    // normal behavior and is also done for `there1` in the same fashion.
    //
    // For `to_right == false` the second connected factor is a factor of
    // the very same time step (so the "sibling" of `node`).
    if (edge.is_division()) {
      const auto& there2 = edge.node2->detection;
      if (there2.primal().is_incoming_set()) {
        if ((p == slot) != (there2.primal().incoming() == edge.slot2))
          result.mark_inconsistent();
      } else {
        result.mark_unknown();
      }
    }

    return result;
  }

  template<bool to_right, typename DETECTION_NODE>
  static consistency check_primal_consistency(const DETECTION_NODE& node, index slot)
  {
    consistency this_side = check_primal_consistency_impl<to_right>(node, slot);

#ifndef NDEBUG
    assert(slot >= 0 && slot < node.template transitions<to_right>().size());
    const auto& edge = node.template transitions<to_right>()[slot];

    consistency other_side1 = check_primal_consistency_impl<!to_right>(*edge.node1, edge.slot1);
    assert(this_side == other_side1);

    if (edge.is_division()) {
      consistency other_side2 = check_primal_consistency_impl<false>(*edge.node2, edge.slot2);
      assert(this_side == other_side2);
    }
#endif

    return this_side;
  }

  template<typename DETECTION_NODE>
  static consistency check_primal_consistency(const DETECTION_NODE& node)
  {
    consistency result;

    node.template traverse_transitions<false>([&](const auto& edge, auto slot) {
      result.merge(check_primal_consistency<false>(node, slot));
    });

    node.template traverse_transitions<true>([&](const auto& edge, auto slot) {
      result.merge(check_primal_consistency<true>(node, slot));
    });

    return result;
  }

  template<bool to_right, typename DETECTION_NODE>
  static void propagate_primal(const DETECTION_NODE& node)
  {
    const auto& here = node.detection;

    if (here.primal().is_detection_off())
      return;

    if constexpr (to_right) {
      assert(here.primal().is_outgoing_set());
      if (here.primal().outgoing() < here.outgoing_.size() - 1) {
        const auto& edge = node.outgoing[here.primal().outgoing()];

        edge.node1->detection.primal().set_incoming(edge.slot1);
        for (const auto& conflict_edge : edge.node1->conflicts) {
          conflict_messages::template propagate_primal_to_conflict(*conflict_edge.node);
          conflict_messages::template propagate_primal_to_detections(*conflict_edge.node);
        }

        if (edge.is_division()) {
          edge.node2->detection.primal().set_incoming(edge.slot2);
          for (const auto& conflict_edge : edge.node2->conflicts) {
            conflict_messages::template propagate_primal_to_conflict(*conflict_edge.node);
            conflict_messages::template propagate_primal_to_detections(*conflict_edge.node);
          }
        }
      }
    } else {
      assert(here.primal().is_incoming_set());
      if (here.primal().incoming() < here.incoming_.size() - 1){
        const auto& edge = node.incoming[here.primal().incoming()];

        edge.node1->detection.primal().set_outgoing(edge.slot1);
        for (const auto& conflict_edge : edge.node1->conflicts) {
          conflict_messages::template propagate_primal_to_conflict(*conflict_edge.node);
          conflict_messages::template propagate_primal_to_detections(*conflict_edge.node);
        }

        if (edge.is_division()) {
          edge.node2->detection.primal().set_incoming(edge.slot2);
          for (const auto& conflict_edge : edge.node2->conflicts) {
            conflict_messages::template propagate_primal_to_conflict(*conflict_edge.node);
            conflict_messages::template propagate_primal_to_detections(*conflict_edge.node);
          }
        }
      }
    }
  }

  template<bool from_left, typename DETECTION_NODE, typename CONTAINER>
  static void get_primal_possibilities(const DETECTION_NODE& node, CONTAINER& out)
  {
    out.fill(true);

    auto get_primal = [&](const auto& factor) {
      if constexpr (from_left)
        return factor.primal().outgoing();
      else
        return factor.primal().incoming();
    };

    [[maybe_unused]]
    auto get_primal2 = [&](const auto& factor) {
      if constexpr (from_left)
        return factor.primal().incoming();
      else
        return factor.primal().outgoing();
    };

    auto it = out.begin();
    for (const auto& edge : node.template transitions<!from_left>()) {
      assert(it != out.end());

      auto helper = [&](const auto& factor, auto slot, auto primal_getter) {
        const auto p = primal_getter(factor);
        if (p != detection_primal::undecided && p != slot)
          *it = false;

        if (p == slot) {
          bool current = *it;
          out.fill(false);
          *it = current;
        }
      };

      helper(edge.node1->detection, edge.slot1, get_primal);
      if (edge.is_division()) {
        if constexpr (from_left)
          helper(edge.node2->detection, edge.slot2, get_primal2);
        else
          helper(edge.node2->detection, edge.slot2, get_primal);
      }
      ++it;
    }

    assert(std::find(out.cbegin(), out.cend(), true) != out.cend());
  }

};

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
