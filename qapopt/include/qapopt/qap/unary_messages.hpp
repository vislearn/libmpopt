#ifndef LIBQAPOPT_QAP_UNARY_MESSAGES_HPP
#define LIBQAPOPT_QAP_UNARY_MESSAGES_HPP

namespace qapopt {
namespace qap {

struct unary_messages {

  template<typename UNARY_NODE>
  static void diffusion(const UNARY_NODE* unary_node)
  {
    unary_node->traverse_uniqueness([unary_node](const auto& link, const index slot) {
      const auto msg = unary_node->factor.get(slot) * 0.25d;
      unary_node->factor.repam(slot, -msg);
      link.node->factor.repam(link.slot, msg);
    });

    // This is only the split factor for all connected pairwise factors.
    // We later add one if there is also a uniqueness factor for a given label.
    index split = unary_node->forward.size() + unary_node->backward.size();
    for (const auto* pairwise_node : unary_node->backward) {
      assert(split >= 1);
      directed_send_helper<false>(unary_node, pairwise_node, 1.0d / split);
      --split;
    }
    for (const auto* pairwise_node : unary_node->forward) {
      assert(split >= 1);
      directed_send_helper<true>(unary_node, pairwise_node, 1.0d / split);
      --split;
    }
    assert(split == 0);

#ifndef NDEBUG
    for (index i = 0; i < unary_node->factor.size(); ++i)
      assert(dbg::are_identical(unary_node->factor.get(i), 0.0));
#endif
  }

  template<typename UNARY_NODE>
  static consistency check_primal_consistency(const UNARY_NODE* unary_node)
  {
    constexpr auto unary_unset = decltype(unary_node->factor)::primal_unset;
    constexpr auto pairwise_unset = decltype(unary_node->forward[0]->factor)::primal_unset;
    constexpr auto uniqueness_unset = decltype(unary_node->uniqueness[0].node->factor)::primal_unset;
    static_assert(unary_unset == pairwise_unset && unary_unset == uniqueness_unset);
    constexpr auto unset = unary_unset;

    consistency result;
    if (unary_node->factor.primal() == unset) {
      result.mark_unknown();
      return result;
    }

    for (const auto* pairwise_node : unary_node->forward) {
      const auto pw_p = std::get<0>(pairwise_node->factor.primal());
      if (pw_p == unset)
        result.mark_unknown();
      else if (pw_p != unary_node->factor.primal())
        result.mark_inconsistent();
    }

    for (const auto* pairwise_node : unary_node->backward) {
      const auto pw_p = std::get<1>(pairwise_node->factor.primal());
      if (pw_p == unset)
        result.mark_unknown();
      else if (pw_p != unary_node->factor.primal())
        result.mark_inconsistent();
    }

    unary_node->traverse_uniqueness([&](const auto& link, const index slot) {
      if (link.node->factor.primal() == unset)
        result.mark_unknown();
      else if ((unary_node->factor.primal() == slot) != (link.node->factor.primal() == link.slot))
        result.mark_inconsistent();
    });

    return result;
  }

private:

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
