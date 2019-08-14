#ifndef LIBMPOPT_QAP_SOLVER_HPP
#define LIBMPOPT_QAP_SOLVER_HPP

namespace mpopt {
namespace qap {

template<typename ALLOCATOR>
class solver {
public:
  using allocator_type = ALLOCATOR;
  using graph_type = graph<allocator_type>;

  solver(const ALLOCATOR& allocator = ALLOCATOR())
  : graph_(allocator)
  , iterations_(0)
  { }

  auto& get_graph() { return graph_; }
  const auto& get_graph() const { return graph_; }

  cost lower_bound() const
  {
    assert(graph_.is_prepared());
    cost result = 0;

    for (const auto* node : graph_.unaries())
      result += node->unary.lower_bound();

    for (const auto* node : graph_.uniqueness())
      result += node->uniqueness.lower_bound();

    for (const auto* node : graph_.pairwise())
      result += node->pairwise.lower_bound();

    return result;
  }

  cost evaluate_primal() const
  {
    assert(graph_.is_prepared());
    const cost inf = std::numeric_limits<cost>::infinity();
    cost result = 0;

    for (const auto* node : graph_.unaries())
      result += node->unary.evaluate_primal();

    for (const auto* node : graph_.uniqueness()) {
      if (!uniqueness_messages::check_primal_consistency(node))
        result += inf;
      result += node->uniqueness.evaluate_primal();
    }

    for (const auto* node : graph_.pairwise()) {
      if (!pairwise_messages::check_primal_consistency(node))
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

  void run(const int max_iterations = 1000)
  {
    const int max_batches = (max_iterations + batch_size - 1) / batch_size;
    assert(graph_.is_prepared());
    cost best_ub = std::numeric_limits<cost>::infinity();

    signal_handler h;
    using clock_type = std::chrono::high_resolution_clock;
    const auto clock_start = clock_type::now();
    std::cout.precision(std::numeric_limits<cost>::max_digits10);
    for (int i = 0; i < max_batches && !h.signaled(); ++i) {
      for (int j = 0; j < batch_size-1; ++j)
        single_pass<false>();

      single_pass<true>();
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

    gurobi_model_builder<allocator_type> builder(env_);

    for (const auto* node : graph_.unaries())
      builder.add_factor(node);

    for (const auto* node : graph_.uniqueness())
      builder.add_factor(node);

    for (const auto* node : graph_.pairwise())
      builder.add_factor(node);

    builder.finalize();
    builder.optimize();
    builder.update_primals();
    std::cout << "final objective: " << evaluate_primal() << std::endl;
  }

  void solve_lap_as_ilp()
  {
    reset_primal();

    gurobi_model_builder<allocator_type> builder(env_);

    for (const auto* node : graph_.unaries())
      builder.add_factor(node);

    for (const auto* node : graph_.uniqueness())
      builder.add_factor(node);

    builder.finalize();
    builder.optimize();
    builder.update_primals();

    for (const auto* node : graph_.pairwise())
      node->pairwise.primal() = std::tuple(
        node->unary0->unary.primal(),
        node->unary1->unary.primal());
  }

protected:

  template<bool rounding>
  void single_pass()
  {
    for (const auto* node : graph_.pairwise())
      pairwise_messages::update(node);

    if constexpr (rounding)
      solve_lap_as_ilp();

    for (const auto* node : graph_.unaries())
      uniqueness_messages::send_messages_to_uniqueness(node);

    for (const auto* node : graph_.uniqueness())
      uniqueness_messages::send_messages_to_unaries(node);
  }

  graph_type graph_;
  int iterations_;
  GRBEnv env_;
};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
