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
    cost result = constant_;

    // FIXME: Check that primals are really consistent!

    for (const auto* node : graph_.unaries())
      result += node->unary.evaluate_primal();

    for (const auto* node : graph_.pairwise())
      result += node->pairwise.evaluate_primal();

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
          index best_label;
          cost best_value = std::numeric_limits<cost>::infinity();
          for (index i = 0; i < (*it)->unary.size(); ++i) {
            cost current = (*it)->unary.get(i);
            for (auto* edge : (*it)->template edges<!forward>()) {
              const index j = std::get<forward ? 0 : 1>(edge->pairwise.primal());
              assert(j != decltype(edge->pairwise)::primal_unset);
              if constexpr (forward)
                current += edge->pairwise.get(j, i);
              else
                current += edge->pairwise.get(i, j);
            }

            if (current < best_value) {
              best_value = current;
              best_label = i;
            }
          }
          (*it)->unary.primal() = best_label;
          messages::propagate_primals(*it);
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
