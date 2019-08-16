#ifndef LIBMPOPT_QAP_UNIQUENESS_MESSAGES_HPP
#define LIBMPOPT_QAP_UNIQUENESS_MESSAGES_HPP

namespace mpopt {
namespace qap {

struct uniqueness_messages {

  template<typename UNARY_NODE>
  static void send_messages_to_uniqueness(const UNARY_NODE* node)
  {
    auto [it0, it1] = least_two_elements(node->factor.costs_.cbegin(), node->factor.costs_.cend());
    assert(it0 != node->factor.costs_.cend());
    assert(it1 != node->factor.costs_.cend());
    const cost target = 0.5 * (*it0 + *it1);

    // FIXME: This is not quite correct. Let's assume there is a label that has
    //        no uniqueness term connected. Then we cannot push away costs.
    //        Are messages updates still correct then?
    node->traverse_uniqueness([&](const auto& link, index slot) {
      const auto msg = node->factor.get(slot) - target;
      node->factor.repam(slot, -msg);
      link.node->factor.repam(link.slot, msg);
    });
  }

  template<typename UNIQUENESS_NODE>
  static void send_messages_to_unaries(const UNIQUENESS_NODE* node)
  {
    auto [it0, it1] = least_two_elements(node->factor.costs_.cbegin(), node->factor.costs_.cend());
    assert(it0 != node->factor.costs_.cend());
    assert(it1 != node->factor.costs_.cend());
    const cost target = 0.5 * (*it0 + *it1);

    // FIXME: The same comment as above applies.
    node->traverse_unaries([&](const auto& link, index slot) {
      const auto msg = node->factor.get(slot) - target;
      node->factor.repam(slot, -msg);
      link.node->factor.repam(link.slot, msg);
    });
  }

  template<typename UNIQUENESS_NODE>
  static consistency check_primal_consistency(const UNIQUENESS_NODE* uniqueness_node)
  {
    constexpr auto uniqueness_unset = decltype(uniqueness_node->factor)::primal_unset;
    constexpr auto unary_unset = decltype(uniqueness_node->unaries[0].node->factor)::primal_unset;

    consistency result;
    if (uniqueness_node->factor.primal() == uniqueness_unset) {
      result.mark_unknown();
      return result;
    }

    uniqueness_node->traverse_unaries([&](const auto& link, index slot) {
      if (link.node->factor.primal() == unary_unset)
        result.mark_unknown();
      else if ((link.node->factor.primal() == link.slot) != (uniqueness_node->factor.primal() == slot))
        result.mark_inconsistent();
    });

    return result;
  }

};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
