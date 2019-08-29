#ifndef LIBMPOPT_GM_MESSAGES_HPP
#define LIBMPOPT_GM_MESSAGES_HPP

namespace mpopt {
namespace gm {

struct messages {

  template<bool forward, typename UNARY_NODE>
  static void receive(const UNARY_NODE* unary_node, double beta=1.0)
  {
#ifndef NDEBUG
    const auto lb_before = local_lower_bound(unary_node);
#endif

    auto& edges = unary_node->template edges<!forward>();
    for (const auto* pairwise_node : edges) {
      for (index l = 0; l < unary_node->factor.size(); ++l) {
        const cost msg = pairwise_node->factor.template min_marginal<forward>(l) * beta;
        pairwise_node->factor.template repam<forward>(l, -msg);
        unary_node->factor.repam(l, msg);
      }
    }

#ifndef NDEBUG
    const auto lb_after = local_lower_bound(unary_node);
    assert(lb_before <= lb_after + epsilon);
#endif
  }

  template<bool forward, typename UNARY_NODE>
  static void send(const UNARY_NODE* unary_node, double beta=1.0)
  {
#ifndef NDEBUG
    const auto lb_before = local_lower_bound(unary_node);
#endif

    auto& edges = unary_node->template edges<forward>();
    index split = std::max(unary_node->forward.size(), unary_node->backward.size());
    for (const auto* pairwise_node : edges) {
      assert(split >= 1);
      directed_send_helper<forward>(unary_node, pairwise_node, beta * 1.0 / split);
      --split;
    }

#ifndef NDEBUG
    const auto lb_after = local_lower_bound(unary_node);
    assert(lb_before <= lb_after + epsilon);
#endif
  }

  template<typename UNARY_NODE>
  static void diffusion(const UNARY_NODE* unary_node)
  {
#ifndef NDEBUG
    const auto lb_before = local_lower_bound(unary_node);
#endif

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
    for (index i = 0; i < unary_node->factor.size(); ++i)
      assert(dbg::are_identical(unary_node->factor.get(i), 0.0));

    const auto lb_after = local_lower_bound(unary_node);
    assert(lb_before <= lb_after + epsilon);
#endif
  }

  template<bool forward, typename PAIRWISE_NODE>
  static void send_pairwise(const PAIRWISE_NODE* pairwise_node)
  {
    const auto no_labels_to = std::get<forward ? 1 : 0>(pairwise_node->factor.size());
    for (index l = 0; l < no_labels_to; ++l) {
      const auto msg = pairwise_node->factor.template min_marginal<forward>(l);
      pairwise_node->factor.template repam<forward>(l, -msg);
      pairwise_node->template unary<forward>()->factor.repam(l, msg);
    }
  }

  template<bool forward, typename UNARY_NODE>
  static void trws_style_rounding(const UNARY_NODE* unary_node)
  {
    std::tuple<index, cost> best(0, std::numeric_limits<cost>::infinity());
    for (index i = 0; i < unary_node->factor.size(); ++i) {
      cost value = unary_node->factor.get(i);
      for (auto* edge : unary_node->template edges<!forward>()) {
        const index j = std::get<forward ? 0 : 1>(edge->factor.primal());
        assert(j != decltype(edge->factor)::primal_unset);
        value += edge->factor.get(forward ? j : i, forward ? i : j);
      }

      if (value < std::get<1>(best))
        best = {i, value};
    }
    unary_node->factor.primal() = std::get<0>(best);
  }

  template<typename UNARY_NODE>
  static consistency check_unary_primal_consistency(const UNARY_NODE* unary_node)
  {
    constexpr auto un_unset = decltype(unary_node->factor)::primal_unset;
    constexpr auto pw_unset = decltype(unary_node->forward[0]->factor)::primal_unset;

    consistency result;
    if (unary_node->factor.primal() == un_unset) {
      result.mark_unknown();
      return result;
    }

    for (const auto* pairwise_node : unary_node->forward) {
      auto [pw0, pw1] = pairwise_node->factor.primal();
      if (pw0 == pw_unset)
        result.mark_unknown();
      else if (pw0 != unary_node->factor.primal())
        result.mark_inconsistent();
    }

    for (const auto* pairwise_node : unary_node->backward) {
      auto [pw0, pw1] = pairwise_node->factor.primal();
      if (pw1 == pw_unset)
        result.mark_unknown();
      else if (pw1 != unary_node->factor.primal())
        result.mark_inconsistent();
    }

    return result;
  }

  template<typename PAIRWISE_NODE>
  static consistency check_pairwise_primal_consistency(const PAIRWISE_NODE* pairwise_node)
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

  template<typename UNARY_NODE>
  static void propagate_primal(const UNARY_NODE* unary_node)
  {
    assert(unary_node->factor.primal() != decltype(unary_node->factor)::primal_unset);
    index primal = unary_node->factor.primal();

    for (const auto* pairwise_node : unary_node->forward) {
      auto [pw_primal0, pw_primal1] = pairwise_node->factor.primal();
      pw_primal0 = primal;
    }

    for (const auto* pairwise_node : unary_node->backward) {
      auto [pw_primal0, pw_primal1] = pairwise_node->factor.primal();
      pw_primal1 = primal;
    }
  }

private:

  template<typename UNARY_NODE>
  static cost local_lower_bound(const UNARY_NODE* unary_node)
  {
    cost lb = unary_node->factor.lower_bound();

    for (const auto* pairwise_node : unary_node->backward)
      lb += pairwise_node->factor.lower_bound();

    for (const auto* pairwise_node: unary_node->forward)
      lb += pairwise_node->factor.lower_bound();

    return lb;
  }

  template<bool forward, typename UNARY_NODE, typename PAIRWISE_NODE>
  static void directed_send_helper(const UNARY_NODE* unary_node, const PAIRWISE_NODE* pairwise_node, double fraction)
  {
    assert(fraction > 0 && fraction <= 1);
    for (index l = 0; l < unary_node->factor.size(); ++l) {
      const cost msg = unary_node->factor.get(l) * fraction;
      unary_node->factor.repam(l, -msg);
      pairwise_node->factor.template repam<!forward>(l, msg);
    }
  }

};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
