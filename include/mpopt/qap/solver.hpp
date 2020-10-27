#ifndef LIBMPOPT_QAP_SOLVER_HPP
#define LIBMPOPT_QAP_SOLVER_HPP

namespace mpopt {
namespace qap {

template<typename ALLOCATOR>
class solver : public ::mpopt::solver<solver<ALLOCATOR>> {
public:
  using allocator_type = ALLOCATOR;
  using graph_type = graph<allocator_type>;
  using unary_node_type = typename graph_type::unary_node_type;
  using uniqueness_node_type = typename graph_type::uniqueness_node_type;
  using pairwise_node_type = typename graph_type::pairwise_node_type;

#ifdef ENABLE_GUROBI
  using gurobi_model_builder_type = gurobi_model_builder<allocator_type>;
#endif

  // import from base class
  using typename ::mpopt::solver<solver<ALLOCATOR>>::clock_type;

  solver(const ALLOCATOR& allocator = ALLOCATOR())
  : graph_(allocator)
  { }

  auto& get_graph() { return graph_; }
  const auto& get_graph() const { return graph_; }

  void run(const int max_iterations = 1000)
  {
    graph_.check_structure();
    const int max_batches = (max_iterations + batch_size - 1) / batch_size;
    cost best_ub = infinity;

    signal_handler h;
    std::cout.precision(std::numeric_limits<cost>::max_digits10);
    for (int i = 0; i < max_batches && !h.signaled(); ++i) {
      const auto clock_start = clock_type::now();

      for (int j = 0; j < batch_size-1; ++j)
        single_pass<false>();

      single_pass<true>();

      best_ub = std::min(best_ub, this->evaluate_primal());
      const auto lb = this->lower_bound();
      this->iterations_ += batch_size;

      const auto clock_end = clock_type::now();
      this->duration_ += clock_end - clock_start;

      std::cout << "it=" << this->iterations_ << " "
                << "lb=" << lb << " "
                << "ub=" << best_ub << " "
                << "gap=" << static_cast<float>(100.0 * (best_ub - lb) / std::abs(lb)) << "% "
                << "t=" << this->runtime() << std::endl;
    }
  }

  void execute_combilp()
  {
#ifdef ENABLE_GUROBI
    this->reset_primal();
    combilp subsolver(graph_, this->constant_);
    subsolver.run();
#else
    abort_on_disabled_gurobi();
#endif
  }

  void solve_lap_as_ilp()
  {
#ifdef ENABLE_GUROBI
    // We do not reset the primals and use the currently set ones as MIP start.
    gurobi_model_builder<allocator_type> builder(this->gurobi_env());

    for (const auto* node : graph_.unaries())
      builder.add_factor(node);

    for (const auto* node : graph_.uniqueness())
      builder.add_factor(node);

#ifndef NDEBUG
    for (const auto* node : graph_.pairwise())
      assert(dbg::are_identical(node->factor.lower_bound(), 0.0));
#endif

    builder.finalize();
    builder.optimize();
    builder.update_primals();

    for (const auto* node : graph_.pairwise())
      node->factor.primal() = std::tuple(
        node->unary0->factor.primal(),
        node->unary1->factor.primal());
#endif
  }

protected:

  template<typename FUNCTOR>
  void for_each_node(FUNCTOR f) const
  {
    for (const auto* node : graph_.unaries())
      f(node);

    for (const auto* node : graph_.uniqueness())
      f(node);

    for (const auto* node : graph_.pairwise())
      f(node);
  }

  template<typename NODE_TYPE>
  bool check_primal_consistency(const NODE_TYPE* node) const
  {
    if constexpr (std::is_same_v<NODE_TYPE, unary_node_type>)
      return unary_messages::check_primal_consistency(node);
    else if constexpr (std::is_same_v<NODE_TYPE, uniqueness_node_type>)
      return uniqueness_messages::check_primal_consistency(node);
    else if constexpr (std::is_same_v<NODE_TYPE, pairwise_node_type>)
      return pairwise_messages::check_primal_consistency(node);
    return false;
  }

  template<bool rounding>
  void single_pass()
  {
#ifndef NDEBUG
    auto lb_before = this->lower_bound();
#endif

    for (const auto* node : graph_.pairwise())
      pairwise_messages::update(node);

    if constexpr (rounding)
      solve_lap_as_ilp();

    for (const auto* node : graph_.unaries()) {
      this->constant_ += node->factor.normalize();
      uniqueness_messages::send_messages_to_uniqueness(node);
    }

    for (const auto* node : graph_.uniqueness()) {
      this->constant_ += node->factor.normalize();
      uniqueness_messages::send_messages_to_unaries(node);
    }

#ifndef NDEBUG
    auto lb_after = this->lower_bound();
    assert(lb_before <= lb_after + epsilon);
#endif
  }

  graph_type graph_;
  friend class ::mpopt::solver<solver<ALLOCATOR>>;
};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
