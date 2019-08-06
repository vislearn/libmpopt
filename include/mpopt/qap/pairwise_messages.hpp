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

private:

  template<typename PAIRWISE_NODE>
  static void helper_move_to_pairwise(const PAIRWISE_NODE* pairwise_node)
  {
    const auto* unary0 = pairwise_node->unary0;
    const auto* unary1 = pairwise_node->unary1;

    for (index l = 0; l < unary0->unary.size(); ++l) {
      const cost msg = unary0->unary.get(l);
      unary0->unary.repam(l, -msg);
      pairwise_node->pairwise.repam0(l, msg);
    }
    for (index l = 0; l < unary1->unary.size(); ++l) {
      const cost msg = unary1->unary.get(l);
      unary1->unary.repam(l, -msg);
      pairwise_node->pairwise.repam1(l, msg);
    }
  }

  template<bool forward, bool half, typename PAIRWISE_NODE>
  static void helper_move_to_unary(const PAIRWISE_NODE* pairwise_node)
  {
    const auto* unary_node = pairwise_node->template unary<forward>();
    for (index l = 0; l < unary_node->unary.size(); ++l) {
      const cost msg = (half ? 0.5 : 1.0) * pairwise_node->pairwise.template min_marginal<forward>(l);
      pairwise_node->pairwise.template repam<forward>(l, -msg);
      unary_node->unary.repam(l, msg);
    }
  }

};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
