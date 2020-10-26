#ifndef LIBMPOPT_QAP_COMBILP_HPP
#define LIBMPOPT_QAP_COMBILP_HPP

namespace mpopt {
namespace qap {

#ifdef ENABLE_GUROBI

template<typename ALLOCATOR>
class combilp {
public:
  using allocator_type = ALLOCATOR;
  using graph_type = graph<allocator_type>;
  using unary_node_type = typename graph_type::unary_node_type;
  using pairwise_node_type = typename graph_type::pairwise_node_type;

  combilp(const graph_type& graph, cost constant = 0)
  : graph_(&graph)
  , iterations_(0)
  , ilp_time_(0)
  , constant_(constant)
  { }

  auto ilp_time() const { return ilp_time_; }

  void run()
  {
    dbg::timer t;
    gurobi_model_builder<allocator_type> builder(gurobi_env_);
    preprocess();

    bool changed = true;
    while (changed) {
      const auto inconsistent = populate_builder(builder);

      std::cout << "\n== CombiLP iteration " << (iterations_+1)
                << " (" << inconsistent << "/" << graph_->pairwise().size() << ", "
                << (100.0f * inconsistent / graph_->pairwise().size()) << "%)" << std::endl;

      builder.finalize();
      t.start();
      builder.optimize();
      t.stop();
      builder.update_primals();

      auto [lb, ub] = compute_bounds();
      std::cout << "clp-it=" << iterations_ << " "
                << "lb=" << lb << " "
                << "ub=" << ub << " "
                << "gap=" << (100.0 * (ub-lb) / std::abs(lb)) << "%" << std::endl;

      changed = process_mismatches();
      ++iterations_;
      ilp_time_ += t.seconds();
    }

#ifndef NDEBUG
    for (const auto* node : graph_->unaries())
      assert(unary_messages::check_primal_consistency(node));

    for (const auto* node : graph_->pairwise())
      assert(pairwise_messages::check_primal_consistency(node));

    for (const auto* node : graph_->uniqueness())
      assert(uniqueness_messages::check_primal_consistency(node));
#endif
  }

protected:
  void preprocess()
  {
    for (const auto* node : graph_->pairwise())
      pairwise_messages::update(node);

    for (const auto* node : graph_->uniqueness())
      uniqueness_messages::send_messages_to_unaries(node);

    mask_sac_.clear();
    for (const auto* node : graph_->pairwise()) {
      node->factor.reset_primal();
      node->factor.round_independently();
      mask_sac_[node] = true;
    }
  }

  bool process_mismatches()
  {
    size_t violation_counter = 0;

    for (const auto* node : graph_->pairwise()) {
      if (mask_sac_.at(node) && !try_switch_pairwise_primal(node)) {
        ++violation_counter;
        mask_sac_.at(node) = false;
      }
    }

    std::cout << "Found " << violation_counter << " pairwise violations" << std::endl;
    return violation_counter > 0;
  }

  template<typename GUROBI_WRAPPER>
  size_t populate_builder(GUROBI_WRAPPER& builder)
  {
    size_t inconsistent = 0;

    for (const auto* node : graph_->unaries()) {
      node->factor.reset_primal();
      builder.add_factor(node);
    }

    for (const auto* node : graph_->uniqueness()) {
      node->factor.reset_primal();
      builder.add_factor(node);
    }

    for (const auto* node : graph_->pairwise()) {
      if (!mask_sac_.at(node)) {
        node->factor.reset_primal();
        builder.add_factor(node);
        ++inconsistent;
      }
    }

    return inconsistent;
  }

  std::tuple<cost, cost> compute_bounds() const
  {
    cost lb = constant_, ub = constant_;

    for (const auto* node : graph_->unaries()) {
      assert(node->factor.primal() != decltype(node->factor)::primal_unset);
      const auto c = node->factor.evaluate_primal();
      lb += c;
      ub += c;
    }

    for (const auto* node : graph_->uniqueness()) {
      assert(node->factor.primal() != decltype(node->factor)::primal_unset);
      const auto c = node->factor.evaluate_primal();
      lb += c;
      ub += c;
    }

    for (const auto* node : graph_->pairwise()) {
      if (!mask_sac_.at(node)) {
        assert(pairwise_messages::check_primal_consistency(node));
        const auto c = node->factor.evaluate_primal();
        lb += c;
        ub += c;
      } else {
        lb += node->factor.lower_bound();

        const auto p0 = node->unary0->factor.primal();
        const auto p1 = node->unary1->factor.primal();
        ub += node->factor.get(p0, p1);
      }
    }

    return {lb, ub};
  }

  template<typename PAIRWISE_NODE>
  static bool try_switch_pairwise_primal(const PAIRWISE_NODE* pairwise_node)
  {
    constexpr auto unary_unset = unary_factor<allocator_type>::primal_unset;
    constexpr auto pairwise_unset = pairwise_factor<allocator_type>::primal_unset;

    auto [p0, p1] = pairwise_node->factor.primal();
    assert(p0 != pairwise_unset && p1 != pairwise_unset);

    const auto* unary0 = pairwise_node->unary0;
    const auto* unary1 = pairwise_node->unary1;
    assert(unary0->factor.primal() != unary_unset);
    assert(unary1->factor.primal() != unary_unset);

    const cost current = pairwise_node->factor.get(p0, p1);
    const cost switched = pairwise_node->factor.get(unary0->factor.primal(), unary1->factor.primal());

    if (switched <= current + epsilon) {
      pairwise_node->factor.primal() = std::tuple(unary0->factor.primal(), unary1->factor.primal());
      assert(pairwise_messages::check_primal_consistency(pairwise_node));
      return true;
    }

    return false;
  }

  const graph_type* graph_;
  std::map<const pairwise_node_type*, bool> mask_sac_;
  int iterations_;
  double ilp_time_;
  cost constant_;
  GRBEnv gurobi_env_;
};

#endif

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
