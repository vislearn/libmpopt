#ifndef LIBMPOPT_GM_COMBILP_HPP
#define LIBMPOPT_GM_COMBILP_HPP

namespace mpopt {
namespace gm {

template<typename ALLOCATOR>
class combilp {
public:
  using allocator_type = ALLOCATOR;
  using graph_type = graph<allocator_type>;
  using unary_node_type = typename graph_type::unary_node_type;
  using pairwise_node_type = typename graph_type::pairwise_node_type;

  combilp(const graph_type& graph)
  : graph_(&graph)
  { }

  void run()
  {
    preprocess();

    int iterations = 0;
    bool changed = true;
    while (changed) {
      reparametrize_border();

      GRBEnv env;
      size_t inconsistent = 0;
      gurobi_model_builder<allocator_type> builder(env);
      for (const auto* node : graph_->unaries()) {
        if (!mask_sac_.at(node)) {
          builder.add_factor(node);
          ++inconsistent;
        }
      }
      for (const auto* node : graph_->pairwise()) {
        if (!mask_sac_.at(node->unary0) && !mask_sac_.at(node->unary1))
          builder.add_factor(node);
      }

      std::cout << "\n== CombiLP iteration " << (iterations+1)
                << " (" << inconsistent << "/" << graph_->unaries().size() << ", "
                << (100.0f * inconsistent / graph_->unaries().size()) << "%)" << std::endl;

      builder.finalize();
      builder.optimize();
      builder.update_primals();

#ifndef NDEBUG
      for (const auto* node : graph_->unaries())
        if (mask_sac_.at(node))
          assert(messages::check_unary_primal_consistency(node));
        else
          assert(node->unary.primal() != decltype(node->unary)::primal_unset);
#endif

      changed = process_mismatches();
      ++iterations;
    }

#ifndef NDEBUG
    for (const auto* node : graph_->pairwise())
      assert(messages::check_pairwise_primal_consistency(node));

    for (const auto* node : graph_->unaries())
      assert(messages::check_unary_primal_consistency(node));
#endif
  }

protected:
  void preprocess()
  {
    for (const auto* unary_node : graph_->unaries()) {
      messages::receive<true>(unary_node, 0.9);
      messages::send<true>(unary_node, 0.9);
    }

    for (const auto* unary_node : graph_->unaries()) {
      messages::diffusion(unary_node);
    }

    for (const auto* pairwise_node : graph_->pairwise()) {
      pairwise_node->pairwise.reset_primal();
      pairwise_node->pairwise.round_independently();
    }

    mask_sac_.clear();
    for (const auto* unary_node : graph_->unaries())
      mask_sac_[unary_node] = set_consistent_unary_primal_if_possible(unary_node);
  }

  bool set_consistent_unary_primal_if_possible(const unary_node_type* unary_node)
  {
      unary_node->unary.reset_primal();

      bool is_sac = false;
      bool is_first = true;

      auto process_primal = [&](const index label) {
        if (is_first) {
          is_first = false;
          unary_node->unary.primal() = label;
        } else {
          if (unary_node->unary.primal() != label)
            unary_node->unary.reset_primal();
        }
      };

      for (const auto* pairwise_node : unary_node->backward) {
        auto [p0, p1] = pairwise_node->pairwise.primal();
        assert(p1 != decltype(pairwise_node->pairwise)::primal_unset);
        process_primal(p1);
      }

      for (const auto* pairwise_node : unary_node->forward) {
        auto [p0, p1] = pairwise_node->pairwise.primal();
        assert(p0 != decltype(pairwise_node->pairwise)::primal_unset);
        process_primal(p0);
      }

      if (is_first) {
        unary_node->unary.round_independently();
        is_sac = true;
      }

      is_sac = unary_node->unary.primal() != decltype(unary_node->unary)::primal_unset;
      assert(!is_sac || messages::check_unary_primal_consistency(unary_node));
      return is_sac;
  }

  void reparametrize_border()
  {
    for (const auto* node : graph_->pairwise()) {
      const bool sac0 = mask_sac_.at(node->unary0);
      const bool sac1 = mask_sac_.at(node->unary1);

      if (sac0 != sac1) {
        if (sac0)
          messages::send_pairwise<true>(node);
        else
          messages::send_pairwise<false>(node);
      }
    }
  }

  bool process_mismatches()
  {
    bool result = false;
    for (const auto* pairwise_node : graph_->pairwise()) {
      if (!messages::check_pairwise_primal_consistency(pairwise_node)) {
        if (!try_switch_pairwise_primal(pairwise_node)) {
          result = true;
          mask_sac_.at(pairwise_node->unary0) = false;
          mask_sac_.at(pairwise_node->unary1) = false;
        }
      }
    }

    return result;
  }

  template<typename PAIRWISE_NODE>
  static bool try_switch_pairwise_primal(const PAIRWISE_NODE* pairwise_node)
  {
    constexpr auto unary_unset = unary_factor<allocator_type>::primal_unset;
    constexpr auto pairwise_unset = pairwise_factor<allocator_type>::primal_unset;

    auto [p0, p1] = pairwise_node->pairwise.primal();
    assert(p0 != pairwise_unset && p1 != pairwise_unset);

    const auto* unary0 = pairwise_node->unary0;
    const auto* unary1 = pairwise_node->unary1;
    assert(unary0->unary.primal() != unary_unset);
    assert(unary1->unary.primal() != unary_unset);

    const cost current = pairwise_node->pairwise.get(p0, p1);
    const cost switched = pairwise_node->pairwise.get(unary0->unary.primal(), unary1->unary.primal());

    if (switched <= current + epsilon) {
      pairwise_node->pairwise.primal() = std::tuple(unary0->unary.primal(), unary1->unary.primal());
      assert(messages::check_pairwise_primal_consistency(pairwise_node));
      return true;
    }

    return false;
  }

  const graph_type* graph_;
  std::map<const unary_node_type*, bool> mask_sac_;
};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
