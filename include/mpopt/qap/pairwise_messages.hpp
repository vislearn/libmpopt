#ifndef LIBMPOPT_QAP_PAIRWISE_MESSAGES_HPP
#define LIBMPOPT_QAP_PAIRWISE_MESSAGES_HPP

namespace mpopt {
namespace qap {

struct pairwise_messages {

  template<typename UNARY_NODE>
  static void send_messages_to_pairwise(const UNARY_NODE* node)
  {
    // get number of forward and backward edges
    int number_of_neighbours = node->forward.size() + node->backward.size();

    // The code below unconditionally substracts the message value from the
    // unary node and distributes it among all connected pairwise nodes. If no
    // pairwise nodes exists, the substractions is still performed which would
    // result in a corrupt reparametrization. Guard this case here.
    if (number_of_neighbours == 0)
      return;

    // send full cost evenly to all edges (so remaining cost in node is 0.0)
    for (index l = 0; l < node->factor.size(); ++l) {
      const cost msg = node->factor.get(l);
      node->factor.repam(l, -msg);
      for (const auto* pairwise_node : node->forward)
        pairwise_node->factor.repam0(l, msg/number_of_neighbours);
      for (const auto* pairwise_node : node->backward)
        pairwise_node->factor.repam1(l, msg/number_of_neighbours);
      assert(node->factor.get(l) == 0.0);
    }
  }

  template<typename PAIRWISE_NODE>
  static void receive_pairwise_message(const PAIRWISE_NODE* pairwise_node)
  {
    // move all costs to pairwise node
    helper_move_to_pairwise(pairwise_node);
  }

  // mplp++ edge update (from pairwise to both unaries)
  template<typename PAIRWISE_NODE>
  static void send_messages_to_unaries(const PAIRWISE_NODE* pairwise_node)
  {
    // move half of what is possible to left node (unary0)
    helper_move_to_unary<false, true>(pairwise_node);
    // move as much as possible to right node (unary1)
    helper_move_to_unary<true, false>(pairwise_node);
    // move as much as possible to left node (unary0)
    helper_move_to_unary<false, false>(pairwise_node);
  }

  // full mplp++ edge update
  template<typename PAIRWISE_NODE>
  static void full_mplp_update(const PAIRWISE_NODE* pairwise_node)
  {
    receive_pairwise_message(pairwise_node);
    send_messages_to_unaries(pairwise_node);
  }

  template<typename PAIRWISE_NODE>
  static consistency check_primal_consistency(const PAIRWISE_NODE* pairwise_node)
  {
    constexpr auto pw_unset = decltype(pairwise_node->factor)::primal_unset;
    constexpr auto un_unset = decltype(pairwise_node->unary0->factor)::primal_unset;

    const auto [pw0, pw1] = pairwise_node->factor.primal();
    const auto un0 = pairwise_node->unary0->factor.primal();
    const auto un1 = pairwise_node->unary1->factor.primal();

    consistency result;
    if (pw0 == pw_unset || pw1 == pw_unset) {
      result.mark_unknown();
      return result;
    }

    if (un0 == un_unset)
      result.mark_unknown();
    else if (un0 != pw0)
      result.mark_inconsistent();

    if (un1 == un_unset)
      result.mark_unknown();
    else if (un1 != pw1)
      result.mark_inconsistent();

    return result;
  }

private:

  template<typename PAIRWISE_NODE>
  static void helper_move_to_pairwise(const PAIRWISE_NODE* pairwise_node)
  {
    const auto* unary0 = pairwise_node->unary0;
    const auto* unary1 = pairwise_node->unary1;

    for (index l = 0; l < unary0->factor.size(); ++l) {
      const cost msg = unary0->factor.get(l);
      unary0->factor.repam(l, -msg);
      pairwise_node->factor.repam0(l, msg);
    }
    for (index l = 0; l < unary1->factor.size(); ++l) {
      const cost msg = unary1->factor.get(l);
      unary1->factor.repam(l, -msg);
      pairwise_node->factor.repam1(l, msg);
    }
  }

  template<bool forward, bool half, typename PAIRWISE_NODE>
  static void helper_move_to_unary(const PAIRWISE_NODE* pairwise_node)
  {
    const auto* unary_node = pairwise_node->template unary<forward>();
    for (index l = 0; l < unary_node->factor.size(); ++l) {
      const cost msg = (half ? 0.5 : 1.0) * pairwise_node->factor.template min_marginal<forward>(l);
      pairwise_node->factor.template repam<forward>(l, -msg);
      unary_node->factor.repam(l, msg);
    }
  }

};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
