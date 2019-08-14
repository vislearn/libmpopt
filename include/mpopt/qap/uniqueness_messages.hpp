#ifndef LIBMPOPT_QAP_UNIQUENESS_MESSAGES_HPP
#define LIBMPOPT_QAP_UNIQUENESS_MESSAGES_HPP

namespace mpopt {
namespace qap {

struct uniqueness_messages {

  template<typename UNARY_NODE>
  static void send_messages_to_uniqueness(const UNARY_NODE* node)
  {
    auto [it0, it1] = least_two_elements(node->unary.costs_.cbegin(), node->unary.costs_.cend());
    assert(it0 != node->unary.costs_.cend());
    assert(it1 != node->unary.costs_.cend());
    const cost target = 0.5 * (*it0 + *it1);

    // FIXME: This is not quite correct. Let's assume there is a label that has
    //        no uniqueness term connected. Then we cannot push away costs.
    //        Are messages updates still correct then?
    node->traverse_uniqueness([&](const auto& link, index slot) {
      const auto msg = node->unary.get(slot) - target;
      node->unary.repam(slot, -msg);
      link.node->uniqueness.repam(link.slot, msg);
    });
  }

  template<typename UNIQUENESS_NODE>
  static void send_messages_to_unaries(const UNIQUENESS_NODE* node)
  {
    auto [it0, it1] = least_two_elements(node->uniqueness.costs_.cbegin(), node->uniqueness.costs_.cend());
    assert(it0 != node->uniqueness.costs_.cend());
    assert(it1 != node->uniqueness.costs_.cend());
    const cost target = 0.5 * (*it0 + *it1);

    // FIXME: The same comment as above applies.
    node->traverse_unaries([&](const auto& link, index slot) {
      const auto msg = node->uniqueness.get(slot) - target;
      node->uniqueness.repam(slot, -msg);
      link.node->unary.repam(link.slot, msg);
    });
  }

  template<typename UNIQUENESS_NODE>
  static consistency check_primal_consistency(const UNIQUENESS_NODE* uniqueness_node)
  {
    constexpr auto uniqueness_unset = decltype(uniqueness_node->uniqueness)::primal_unset;
    constexpr auto unary_unset = decltype(uniqueness_node->unaries[0].node->unary)::primal_unset;

    consistency result;
    if (uniqueness_node->uniqueness.primal() == uniqueness_unset) {
      result.mark_unknown();
      return result;
    }

    uniqueness_node->traverse_unaries([&](const auto& link, index slot) {
      if (link.node->unary.primal() == unary_unset)
        result.mark_unknown();
      else if ((link.node->unary.primal() == link.slot) != (uniqueness_node->uniqueness.primal() == slot))
        result.mark_inconsistent();
    });

    return result;
  }

};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
