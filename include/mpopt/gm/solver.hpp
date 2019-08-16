#ifndef LIBMPOPT_GM_SOLVER_HPP
#define LIBMPOPT_GM_SOLVER_HPP

namespace mpopt {
namespace gm {

template<typename ALLOCATOR>
class solver : public ::mpopt::solver<solver<ALLOCATOR>> {
public:
  using allocator_type = ALLOCATOR;
  using graph_type = graph<allocator_type>;
  using gurobi_model_builder_type = gurobi_model_builder<allocator_type>;

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
    using clock_type = std::chrono::high_resolution_clock;
    const auto clock_start = clock_type::now();
    std::cout.precision(std::numeric_limits<cost>::max_digits10);
    for (int i = 0; i < max_batches && !h.signaled(); ++i) {
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

      const auto clock_now = clock_type::now();
      const std::chrono::duration<double> seconds = clock_now - clock_start;

      const auto lb = this->lower_bound();
      this->iterations_ += batch_size;
      std::cout << "it=" << this->iterations_ << " "
                << "lb=" << lb << " "
                << "ub=" << best_ub << " "
                << "gap=" << static_cast<float>(100.0 * (best_ub - lb) / std::abs(lb)) << "% "
                << "t=" << seconds.count() << std::endl;
    }
  }

  void execute_combilp()
  {
    this->reset_primal();
    combilp subsolver(graph_);
    subsolver.run();
    backward_pass<false>();
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

  template<bool rounding> void forward_pass() { single_pass<true, rounding>(); }
  template<bool rounding> void backward_pass() { single_pass<false, rounding>(); }

  template<bool forward, bool rounding>
  void single_pass()
  {
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
  }

  graph_type graph_;

  friend class ::mpopt::solver<solver<ALLOCATOR>>;
};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
