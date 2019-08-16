#ifndef LIBMPOPT_QAP_PAIRWISE_MESSAGES_HPP
#define LIBMPOPT_QAP_PAIRWISE_MESSAGES_HPP

namespace mpopt {
namespace qap {

struct pairwise_messages {

  // complete mplp++ edge update
  template<typename PAIRWISE_NODE>
  static void update(const PAIRWISE_NODE* pairwise_node)
  {
    // move all costs to pairwise node
    helper_move_to_pairwise(pairwise_node);
    // move half of what is possible to left node (unary0)
    helper_move_to_unary<false, true>(pairwise_node);
    // move as much as possible to right node (unary1)
    helper_move_to_unary<true, false>(pairwise_node);
    // move as much as possible to left node (unary0)
    helper_move_to_unary<false, false>(pairwise_node);
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
