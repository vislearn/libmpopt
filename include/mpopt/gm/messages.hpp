#ifndef LIBMPOPT_GM_MESSAGES_HPP
#define LIBMPOPT_GM_MESSAGES_HPP

namespace mpopt {
namespace gm {

struct messages {

  template<bool forward, typename UNARY_NODE>
  static void receive(const UNARY_NODE* unary_node, double beta=1.0)
  {
    auto& edges = unary_node->template edges<!forward>();
    for (const auto* pairwise_node : edges) {
      for (index l = 0; l < unary_node->unary.size(); ++l) {
        const cost msg = pairwise_node->pairwise.template min_marginal<forward>(l) * beta;
        pairwise_node->pairwise.template repam<forward>(l, -msg);
        unary_node->unary.repam(l, msg);
      }
    }
  }

  template<bool forward, typename UNARY_NODE>
  static void send(const UNARY_NODE* unary_node, double beta=1.0)
  {
    auto& edges = unary_node->template edges<forward>();
    index split = std::max(unary_node->forward.size(), unary_node->backward.size());
    for (const auto* pairwise_node : edges) {
      assert(split >= 1);
      directed_send_helper<forward>(unary_node, pairwise_node, beta * 1.0 / split);
      --split;
    }
  }

  template<typename UNARY_NODE>
  static void diffusion(const UNARY_NODE* unary_node)
  {
    index split = unary_node->forward.size() + unary_node->backward.size();
    for (const auto* pairwise_node : unary_node->backward) {
      assert(split >= 1);
      directed_send_helper<false>(unary_node, pairwise_node, 1.0 / split);
      --split;
    }
    for (const auto* pairwise_node : unary_node->forward) {
      assert(split >= 1);
      directed_send_helper<true>(unary_node, pairwise_node, 1.0 / split);
      --split;
    }
    assert(split == 0);

#ifndef NDEBUG
    for (index i = 0; i < unary_node->unary.size(); ++i)
      assert(dbg::are_identical(unary_node->unary.get(i), 0.0));
#endif
  }

  template<bool forward, typename PAIRWISE_NODE>
  static void send_pairwise(const PAIRWISE_NODE* pairwise_node)
  {
    const auto no_labels_to = std::get<forward ? 1 : 0>(pairwise_node->pairwise.size());
    for (index l = 0; l < no_labels_to; ++l) {
      const auto msg = pairwise_node->pairwise.template min_marginal<forward>(l);
      pairwise_node->pairwise.template repam<forward>(l, -msg);
      pairwise_node->template unary<forward>()->unary.repam(l, msg);
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

      if (value < std::get<1>(best))
        best = {i, value};
    }
    unary_node->unary.primal() = std::get<0>(best);
  }

  template<typename UNARY_NODE>
  static consistency check_unary_primal_consistency(const UNARY_NODE* unary_node)
  {
    constexpr auto un_unset = decltype(unary_node->unary)::primal_unset;
    constexpr auto pw_unset = decltype(unary_node->forward[0]->pairwise)::primal_unset;

    consistency result;
    if (unary_node->unary.primal() == un_unset) {
      result.mark_unknown();
      return result;
    }

    for (const auto* pairwise_node : unary_node->forward) {
      if (pairwise_node->pairwise.primal0_ == pw_unset)
        result.mark_unknown();
      else if (unary_node->unary.primal() != pairwise_node->pairwise.primal0_)
        result.mark_inconsistent();
    }

    for (const auto* pairwise_node : unary_node->backward) {
      if (pairwise_node->pairwise.primal1_ == pw_unset)
        result.mark_unknown();
      else if (unary_node->unary.primal() != pairwise_node->pairwise.primal1_)
        result.mark_inconsistent();
    }

    return result;
  }

  template<typename PAIRWISE_NODE>
  static consistency check_pairwise_primal_consistency(const PAIRWISE_NODE* pairwise_node)
  {
    constexpr auto pw_unset = decltype(pairwise_node->pairwise)::primal_unset;
    constexpr auto un_unset = decltype(pairwise_node->unary0->unary)::primal_unset;

    consistency result;
    if (pairwise_node->pairwise.primal0_ == pw_unset ||
        pairwise_node->pairwise.primal1_ == pw_unset)
    {
      result.mark_unknown();
      return result;
    }

    if (pairwise_node->unary0->unary.primal_ == un_unset)
      result.mark_unknown();
    else if (pairwise_node->unary0->unary.primal_ != pairwise_node->pairwise.primal0_)
      result.mark_inconsistent();

    if (pairwise_node->unary1->unary.primal_ == un_unset)
      result.mark_unknown();
    else if (pairwise_node->unary1->unary.primal_ != pairwise_node->pairwise.primal1_)
      result.mark_inconsistent();

    return result;
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

private:

  template<bool forward, typename UNARY_NODE, typename PAIRWISE_NODE>
  static void directed_send_helper(const UNARY_NODE* unary_node, const PAIRWISE_NODE* pairwise_node, double fraction)
  {
    assert(fraction > 0 && fraction <= 1);
    for (index l = 0; l < unary_node->unary.size(); ++l) {
      const cost msg = unary_node->unary.get(l) * fraction;
      unary_node->unary.repam(l, -msg);
      pairwise_node->pairwise.template repam<!forward>(l, msg);
    }
  }

};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
