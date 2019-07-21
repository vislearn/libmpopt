#ifndef LIBMPOPT_GM_SOLVER_HPP
#define LIBMPOPT_GM_SOLVER_HPP

namespace mpopt {
namespace gm {

template<typename ALLOCATOR>
class solver {
public:
  using allocator_type = ALLOCATOR;
  using graph_type = graph<allocator_type>;

  solver(const ALLOCATOR& allocator = ALLOCATOR())
  : graph_(allocator)
  , iterations_(0)
  , constant_(0)
  { }

  auto& get_graph() { return graph_; }
  const auto& get_graph() const { return graph_; }

  cost lower_bound() const
  {
    assert(graph_.is_prepared());
    cost result = constant_;

    for (const auto* node : graph_.unaries())
      result += node->unary.lower_bound();

    for (const auto* node : graph_.pairwise())
      result += node->pairwise.lower_bound();

    return result;
  }

  cost evaluate_primal() const
  {
    assert(graph_.is_prepared());
    const cost inf = std::numeric_limits<cost>::infinity();
    cost result = constant_;

    for (const auto* node : graph_.unaries()) {
      if (!messages::check_unary_primal_consistency(node))
        result += inf;
      result += node->unary.evaluate_primal();
    }

    for (const auto* node : graph_.pairwise()) {
      if (!messages::check_pairwise_primal_consistency(node))
        result += inf;
      result += node->pairwise.evaluate_primal();
    }

    return result;
  }

  cost upper_bound() const { return evaluate_primal(); }

  void reset_primal()
  {
    for (const auto* node : graph_.unaries())
      node->unary.reset_primal();

    for (const auto* node : graph_.pairwise())
      node->pairwise.reset_primal();
  }

  void run(const int max_batches = 1000 / batch_size)
  {
    assert(graph_.is_prepared());
    cost best_ub = std::numeric_limits<cost>::infinity();

    signal_handler h;
    using clock_type = std::chrono::high_resolution_clock;
    const auto clock_start = clock_type::now();
    std::cout.precision(std::numeric_limits<cost>::max_digits10);
    for (int i = 0; i < max_batches && !h.signaled(); ++i) {
      for (int j = 0; j < batch_size-1; ++j) {
        forward_pass<false>();
        backward_pass<false>();
      }

      reset_primal();
      forward_pass<true>();
      best_ub = std::min(best_ub, evaluate_primal());

      reset_primal();
      backward_pass<true>();
      best_ub = std::min(best_ub, evaluate_primal());

      const auto clock_now = clock_type::now();
      const std::chrono::duration<double> seconds = clock_now - clock_start;

      const auto lb = lower_bound();
      iterations_ += batch_size;
      std::cout << "it=" << iterations_ << " "
                << "lb=" << lb << " "
                << "ub=" << best_ub << " "
                << "gap=" << static_cast<float>(100.0 * (best_ub - lb) / std::abs(lb)) << "% "
                << "t=" << seconds.count() << std::endl;
    }
  }

  void solve_ilp()
  {
    reset_primal();

    GRBEnv env;
    gurobi_model_builder<allocator_type> builder(env);
    builder.set_constant(constant_);

    for (const auto* node : graph_.unaries())
      builder.add_factor(node);

    for (const auto* node : graph_.pairwise())
      builder.add_factor(node);

    builder.finalize();
    builder.optimize();
    builder.update_primals();
    std::cout << "final objective: " << evaluate_primal() << std::endl;
  }

  void execute_combilp()
  {
    reset_primal();
    for (auto* unary_node : graph_.unaries()) {
      for (auto* pairwise_node : unary_node->backward)
        pairwise_node->pairwise.round_independently();
      messages::receive<true>(unary_node);
      unary_node->unary.round_independently();
      messages::send<true>(unary_node);
    }
    backward_pass<false>();

    GRBEnv env;
    sac_detector sac(graph_);
    sac.update();

    // FIXME: The current implementation of CombiLP is still faulty.
    //
    // The old implementation of CombiLP was purely working unary nodes. We
    // should compare to the old results and only count the unary factors. As
    // we handle pairwise terms separately the behaviour might have also
    // changed. This has to get verified.
    //
    // The other problem is that the current reparametrization has basically
    // moved all the relevant costs to the end of the structures (so
    // potentially out of the ILP). This is obviously a problem that has to get
    // fixed.
    //
    // Apart from that, it should be working.

    int iterations = 0;
    while (true) {
      const auto [inconsistent, total] = sac.get_stats();
      std::cout << "== CombiLP iteration " << (iterations+1) << " (inconsistency: " << inconsistent << "/" << total << ")" << std::endl;

      gurobi_model_builder<allocator_type> builder(env);
      for (const auto* node : graph_.unaries())
        if (!sac.get(node))
          builder.add_factor(node);

      for (const auto* node : graph_.pairwise())
        if (!sac.get(node))
          builder.add_factor(node);

      builder.finalize();
      builder.optimize();
      builder.update_primals();

      sac_detector sac2(graph_);
      sac2.update();
      sac.merge(sac2);

      if (std::get<0>(sac2.get_stats()) == 0)
        break;

      ++iterations;
    }
  }

protected:

  template<bool rounding> void forward_pass() { single_pass<true, rounding>(); }
  template<bool rounding> void backward_pass() { single_pass<false, rounding>(); }

  template<bool forward, bool rounding>
  void single_pass()
  {
    auto helper = [&](auto begin, auto end) {
      for (auto it = begin; it != end; ++it) {
        messages::receive<forward>(*it);
        constant_ += (*it)->unary.normalize();

        if constexpr (rounding) {
          messages::trws_style_rounding<forward>(*it);
          messages::propagate_primal(*it);
        }

        messages::send<forward>(*it);
      }
    };

    if constexpr (forward)
      helper(graph_.unaries().begin(), graph_.unaries().end());
    else
      helper(graph_.unaries().rbegin(), graph_.unaries().rend());
  }

  graph_type graph_;
  int iterations_;
  cost constant_;
};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
