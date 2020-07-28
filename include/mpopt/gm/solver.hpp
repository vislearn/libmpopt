#ifndef LIBMPOPT_GM_SOLVER_HPP
#define LIBMPOPT_GM_SOLVER_HPP

namespace mpopt {
namespace gm {

template<typename ALLOCATOR>
class solver : public ::mpopt::solver<solver<ALLOCATOR>> {
public:
  using allocator_type = ALLOCATOR;
  using graph_type = graph<allocator_type>;
  using unary_node_type = typename graph_type::unary_node_type;
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
    cost best_ub = std::numeric_limits<cost>::infinity();

    signal_handler h;
    std::cout.precision(std::numeric_limits<cost>::max_digits10);
    for (int i = 0; i < max_batches && !h.signaled(); ++i) {
      const auto clock_start = clock_type::now();

      for (int j = 0; j < batch_size-1; ++j) {
        forward_pass<false>();
        backward_pass<false>();
      }

      this->reset_primal();
      forward_pass<true>();
      best_ub = std::min(best_ub, this->evaluate_primal());

      this->reset_primal();
      backward_pass<true>();
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
    combilp subsolver(graph_);
    subsolver.run();
    backward_pass<false>();
#else
    abort_on_disabled_gurobi();
#endif
  }

protected:

  template<typename FUNCTOR>
  void for_each_node(FUNCTOR f) const
  {
    for (const auto* node : graph_.unaries())
      f(node);

    for (const auto* node : graph_.pairwise())
      f(node);
  }

  bool check_primal_consistency(const unary_node_type* node) const
  {
    return messages::check_unary_primal_consistency(node);
  }

  bool check_primal_consistency(const pairwise_node_type* node) const
  {
    return messages::check_pairwise_primal_consistency(node);
  }

  template<bool rounding> void forward_pass() { single_pass<true, rounding>(); }
  template<bool rounding> void backward_pass() { single_pass<false, rounding>(); }

  template<bool forward, bool rounding>
  void single_pass()
  {
#ifndef NDEBUG
    auto lb_before = this->lower_bound();
#endif

    auto helper = [&](auto begin, auto end) {
      for (auto it = begin; it != end; ++it) {
        messages::receive<forward>(*it);
        this->constant_ += (*it)->factor.normalize();

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
