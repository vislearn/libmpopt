#ifndef LIBMPOPT_GM_MESSAGES_HPP
#define LIBMPOPT_GM_MESSAGES_HPP

namespace mpopt {
namespace gm {

struct messages {

  template<bool forward, typename UNARY_NODE>
  static void receive(const UNARY_NODE* unary_node)
  {
    auto& edges = unary_node->template edges<!forward>();
    for (const auto* pairwise_node : edges) {
      for (index l = 0; l < unary_node->unary.size(); ++l) {
        const cost msg = pairwise_node->pairwise.template min_marginal<forward>(l);
        pairwise_node->pairwise.template repam<forward>(l, -msg);
        unary_node->unary.repam(l, msg);
      }
    }
  }

  template<bool forward, typename UNARY_NODE>
  static void send(const UNARY_NODE* unary_node)
  {
    auto& edges = unary_node->template edges<forward>();
    index split = std::max(unary_node->forward.size(), unary_node->backward.size());
    for (const auto* pairwise_node : edges) {
      assert(split >= 1);
      for (index l = 0; l < unary_node->unary.size(); ++l) {
        const cost msg = unary_node->unary.get(l) / split;
        unary_node->unary.repam(l, -msg);
        pairwise_node->pairwise.template repam<!forward>(l, msg);
      }
      --split;
    }
  }

  template<typename UNARY_NODE>
  static void propagate_primals(const UNARY_NODE* unary_node)
  {
    assert(unary_node->unary.primal() != decltype(unary_node->unary)::primal_unset);
    index primal = unary_node->unary.primal();

    for (const auto* pairwise_node : unary_node->forward)
      pairwise_node->pairwise.primal0_ = primal;

    for (const auto* pairwise_node : unary_node->backward)
      pairwise_node->pairwise.primal1_ = primal;
  }

};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
