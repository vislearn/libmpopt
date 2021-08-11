#ifndef LIBQAPOPT_QAP_UNIQUENESS_MESSAGES_HPP
#define LIBQAPOPT_QAP_UNIQUENESS_MESSAGES_HPP

namespace qapopt {
namespace qap {

struct uniqueness_messages {

  template<typename UNARY_NODE>
  static void send_messages_to_uniqueness(const UNARY_NODE* node)
  {
#ifndef NDEBUG
    const auto lb_before = local_unary_lower_bound(node);
#endif

    auto [it0, it1] = least_two_elements(node->factor.costs_.cbegin(), node->factor.costs_.cend());
    assert(it0 != node->factor.costs_.cend());
    assert(it1 != node->factor.costs_.cend());
    const cost target = 0.5 * (*it0 + *it1);

    node->traverse_uniqueness([&](const auto& link, index slot) {
      const auto msg = node->factor.get(slot) - target;
      node->factor.repam(slot, -msg);
      link.node->factor.repam(link.slot, msg);
    });

#ifndef NDEBUG
    const auto lb_after = local_unary_lower_bound(node);
    assert(lb_before <= lb_after + epsilon);
#endif
  }

  template<typename UNIQUENESS_NODE>
  static void send_messages_to_unaries(const UNIQUENESS_NODE* node)
  {
#ifndef NDEBUG
    const auto lb_before = local_uniqueness_lower_bound(node);
#endif

    auto [it0, it1] = least_two_elements(node->factor.costs_.cbegin(), node->factor.costs_.cend());
    assert(it0 != node->factor.costs_.cend());
    assert(it1 != node->factor.costs_.cend());
    const cost target = 0.5 * (*it0 + *it1);

    node->traverse_unaries([&](const auto& link, index slot) {
      const auto msg = node->factor.get(slot) - target;
      node->factor.repam(slot, -msg);
      link.node->factor.repam(link.slot, msg);
    });

#ifndef NDEBUG
    const auto lb_after = local_uniqueness_lower_bound(node);
    assert(lb_before <= lb_after + epsilon);
#endif
  }

  template<typename UNIQUENESS_NODE>
  static consistency check_primal_consistency(const UNIQUENESS_NODE* node)
  {
    constexpr auto uniqueness_unset = decltype(node->factor)::primal_unset;
    constexpr auto unary_unset = decltype(node->unaries[0].node->factor)::primal_unset;

    consistency result;
    if (node->factor.primal() == uniqueness_unset) {
      result.mark_unknown();
      return result;
    }

    node->traverse_unaries([&](const auto& link, index slot) {
      if (link.node->factor.primal() == unary_unset)
        result.mark_unknown();
      else if ((link.node->factor.primal() == link.slot) != (node->factor.primal() == slot))
        result.mark_inconsistent();
    });

    return result;
  }

private:

  template<typename UNARY_NODE>
  static cost local_unary_lower_bound(const UNARY_NODE* node)
  {
    cost result = node->factor.lower_bound();
    node->traverse_uniqueness([&result](const auto& link, const index slot) {
      result += link.node->factor.lower_bound();
    });
    return result;
  }

  template<typename UNIQUENESS_NODE>
  static cost local_uniqueness_lower_bound(const UNIQUENESS_NODE* node)
  {
    cost result = node->factor.lower_bound();
    node->traverse_unaries([&result](const auto& link, const index slot) {
      result += link.node->factor.lower_bound();
    });
    return result;
  }

};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
