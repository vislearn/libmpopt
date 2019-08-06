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

};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
