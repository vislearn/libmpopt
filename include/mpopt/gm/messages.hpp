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

  template<bool forward, typename UNARY_NODE>
  static void trws_style_rounding(const UNARY_NODE* unary_node)
  {
    std::tuple<index, cost> best(0, std::numeric_limits<cost>::infinity());
    for (index i = 0; i < unary_node->unary.size(); ++i) {
      cost value = unary_node->unary.get(i);
      for (auto* edge : unary_node->template edges<!forward>()) {
        const index j = std::get<forward ? 0 : 1>(edge->pairwise.primal());
        assert(j != decltype(edge->pairwise)::primal_unset);
        value += edge->pairwise.get(forward ? j : i, forward ? i : j);
      }

      if (value < std::get<0>(best))
        best = {i, value};
    }
    unary_node->unary.primal() = std::get<1>(best);
  }

  template<typename UNARY_NODE>
  static void propagate_primal(const UNARY_NODE* unary_node)
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
