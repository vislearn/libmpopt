#ifndef LIBQAPOPT_QAP_SOLVER_HPP
#define LIBQAPOPT_QAP_SOLVER_HPP

namespace qapopt {
namespace qap {

constexpr int default_greedy_generations = 10;

template<typename ALLOCATOR>
class solver : public ::qapopt::solver<solver<ALLOCATOR>> {
public:
  using base_type = ::qapopt::solver<solver<ALLOCATOR>>;
  using allocator_type = ALLOCATOR;
  using graph_type = graph<allocator_type>;
  using unary_node_type = typename graph_type::unary_node_type;
  using uniqueness_node_type = typename graph_type::uniqueness_node_type;
  using pairwise_node_type = typename graph_type::pairwise_node_type;
#ifdef ENABLE_GUROBI
  using gurobi_model_builder_type = gurobi_model_builder<allocator_type>;
#endif

  // import from base class
  using typename base_type::clock_type;

  solver(const ALLOCATOR& allocator = ALLOCATOR())
  : graph_(allocator)
  , fm_ms_build(0)
  , fm_ms_solve(0)
  {
#ifndef ENABLE_QPBO
    std::cerr << "!!!!!!!!!!\n"
              << "ENABLE_QPBO was not deactivated during configuration of libqapopt.\n"
              << "No fusion moves are performed and the the quality of the computed upper bound is degraded.\n"
              << "!!!!!!!!!!\n" << std::endl;
#endif
  }

  auto& get_graph() { return graph_; }
  const auto& get_graph() const { return graph_; }

  cost evaluate_lap() const
  {
    graph_.check_structure();
    const cost inf = std::numeric_limits<cost>::infinity();
    cost result = 0;

    for (const auto* node : graph_.unaries())
      result += node->factor.evaluate_primal();

    for (const auto* node : graph_.uniqueness()) {
      if (!uniqueness_messages::check_primal_consistency(node))
        result += inf;
      result += node->factor.evaluate_primal();
    }

    return result;
  }

  template<bool verbose=true, bool enable_rounding=true>
  void run(const int batch_size=default_batch_size, const int max_batches=default_max_batches)
  {
    graph_.check_structure();

    signal_handler h;
    std::cout.precision(std::numeric_limits<cost>::max_digits10);
    for (int i = 0; i < max_batches && !h.signaled(); ++i) {
      const auto clock_start = clock_type::now();

      for (int j = 0; j < batch_size-1; ++j)
        single_pass<false>();

      single_pass<enable_rounding>();

      this->iterations_ += batch_size;

      const auto clock_end = clock_type::now();
      this->duration_ += clock_end - clock_start;

      if constexpr (verbose)
        output_status();
    }
  }

  template<bool verbose=true>
  void run_rounding_only()
  {
    graph_.check_structure();

    auto clock_tick = clock_type::now();
    solve_lap_as_ilp();
    this->duration_ += clock_type::now() - clock_tick;

    if constexpr (verbose)
      output_status();
  }

  void execute_fusion_move(std::vector<index> solution0, std::vector<index> solution1)
  {
#ifdef ENABLE_GUROBI
    using clock = std::chrono::steady_clock;
    const auto build_start = clock::now();
    gurobi_fusion_model_builder<allocator_type> builder(this->gurobi_env());
    assert(solution0.size() == graph_.unaries().size());
    assert(solution1.size() == graph_.unaries().size());

    for (index i = 0; i < graph_.unaries().size(); ++i)
      builder.add_factor(graph_.unaries()[i], solution0[i], solution1[i]);

    for (const auto* node : graph_.uniqueness())
      builder.add_factor(node);

    for (const auto* node : graph_.pairwise())
      builder.add_factor(node);

    builder.finalize();
    const auto build_end = clock::now();
    const auto solve_start = clock::now();
    builder.optimize();
    const auto solve_end = clock::now();
    builder.update_primals();

    assert(this->check_primal_consistency());

    using ms = std::chrono::duration<double, std::milli>;
    fm_ms_build += std::chrono::duration_cast<ms>(build_end - build_start).count();
    fm_ms_solve += std::chrono::duration_cast<ms>(solve_end - solve_start).count();
#else
    abort_on_disabled_gurobi();
#endif
  }

  void execute_qpbo(std::vector<index> solution0, std::vector<index> solution1, bool enable_weak_persistency, bool enable_probe, bool enable_improve)
  {
#ifdef ENABLE_QPBO
    using clock = std::chrono::steady_clock;
    const auto build_start = clock::now();
    qpbo_model_builder<allocator_type> builder(graph_);
    assert(solution0.size() == graph_.unaries().size());
    assert(solution1.size() == graph_.unaries().size());

    builder.enable_weak_persistency(enable_weak_persistency);
    builder.enable_probe(enable_probe);
    builder.enable_improve(enable_improve);

    for (index i = 0; i < graph_.unaries().size(); ++i)
      builder.add_factor(graph_.unaries()[i], solution0[i], solution1[i]);

    for (const auto* node : graph_.uniqueness())
      builder.add_factor(node);

    for (const auto* node : graph_.pairwise())
      builder.add_factor(node);

    builder.finalize();
    const auto build_end = clock::now();
    const auto solve_start = clock::now();
    builder.optimize();
    const auto solve_end = clock::now();
    builder.update_primals();

    assert(this->check_primal_consistency());

    using ms = std::chrono::duration<double, std::milli>;
    fm_ms_build += std::chrono::duration_cast<ms>(build_end - build_start).count();
    fm_ms_solve += std::chrono::duration_cast<ms>(solve_end - solve_start).count();
#else
    std::abort();
#endif
  }

  void execute_lsatr(std::vector<index> solution0, std::vector<index> solution1)
  {
#ifdef ENABLE_LSATR
    using clock = std::chrono::steady_clock;
    const auto build_start = clock::now();
    lsatr_model_builder<allocator_type> builder(graph_);
    assert(solution0.size() == graph_.unaries().size());
    assert(solution1.size() == graph_.unaries().size());

    for (index i = 0; i < graph_.unaries().size(); ++i)
      builder.add_factor(graph_.unaries()[i], solution0[i], solution1[i]);

    for (const auto* node : graph_.uniqueness())
      builder.add_factor(node);

    for (const auto* node : graph_.pairwise())
      builder.add_factor(node);

    builder.finalize();
    const auto build_end = clock::now();
    const auto solve_start = clock::now();
    builder.optimize();
    const auto solve_end = clock::now();
    builder.update_primals();

    assert(this->check_primal_consistency());

    using ms = std::chrono::duration<double, std::milli>;
    fm_ms_build += std::chrono::duration_cast<ms>(build_end - build_start).count();
    fm_ms_solve += std::chrono::duration_cast<ms>(solve_end - solve_start).count();
#else
    std::abort();
#endif
  }

  void compute_greedy_assignment()
  {
    greedy g(graph_);
    g.run();
    assert(this->check_primal_consistency());
  }

  void solve_lap_as_ilp()
  {
#ifdef ENABLE_GUROBI
    // We do not reset the primals and use the currently set ones as MIP start.
    gurobi_model_builder<allocator_type> builder(this->gurobi_env(), false);

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
#else
    abort_on_disabled_gurobi();
#endif
  }

  double get_fm_ms_build() const { return fm_ms_build; }
  double get_fm_ms_solve() const { return fm_ms_solve; }

protected:
  void output_status() const
  {
    const cost lb = this->lower_bound();
    const cost ub = this->evaluate_primal();
    const cost lap_lb = evaluate_lap();
    const auto gap = 100.0d * (ub - lb) / std::abs(lb);
    const std::chrono::duration<double> seconds = this->duration_;

    std::cout << "it=" << this->iterations_ << " "
              << "lb=" << lb << " "
              << "ub=" << ub << " "
              << "gap=" << gap << "% "
              << "lap=" << lap_lb << " "
              << "t=" << seconds.count() << std::endl;
  }

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

  using base_type::check_primal_consistency;

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

    for (const auto* node : graph_.unaries())
      pairwise_messages::send_messages_to_pairwise(node);

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
  friend class ::qapopt::solver<solver<ALLOCATOR>>;
  double fm_ms_build;
  double fm_ms_solve;
};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
