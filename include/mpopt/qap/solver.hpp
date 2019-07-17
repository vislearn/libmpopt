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

    for (const auto* node : graph_.pairwise())
      result += node->pairwise.lower_bound();

    return result;
  }

  cost evaluate_primal() const
  {
    assert(graph_.is_prepared());
    const cost inf = std::numeric_limits<cost>::infinity();
    cost result = inf;

    // TODO: Implement this.
    assert(false);

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

    signal_handler h;
    using clock_type = std::chrono::high_resolution_clock;
    const auto clock_start = clock_type::now();
    std::cout.precision(std::numeric_limits<cost>::max_digits10);
    for (int i = 0; i < max_batches && !h.signaled(); ++i) {
      for (int j = 0; j < batch_size-1; ++j)
        single_pass<false>();

      single_pass<true>();

      const auto clock_now = clock_type::now();
      const std::chrono::duration<double> seconds = clock_now - clock_start;

      const auto lb = lower_bound();
      iterations_ += batch_size;
      std::cout << "it=" << iterations_ << " "
                << "lb=" << lb << " "
                << "t=" << seconds.count() << std::endl;
    }
  }

protected:

  template<bool rounding>
  void single_pass()
  {
    // TODO: Implement this.
    assert(false);
  }

  graph_type graph_;
  int iterations_;
};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
