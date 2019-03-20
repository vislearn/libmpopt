#ifndef LIBCT_TRACKER_HPP
#define LIBCT_TRACKER_HPP

namespace ct {

template<typename ALLOCATOR = std::allocator<cost>>
class tracker {
public:
  using allocator_type = ALLOCATOR;
  using graph_type = graph<allocator_type>;
  using detection_type = typename graph_type::detection_type;
  using conflict_type = typename graph_type::conflict_type;
  using timestep_type = typename graph_type::timestep_type;

  static constexpr int config_batch = 100;

  tracker(const ALLOCATOR& allocator = ALLOCATOR())
  : graph_(allocator)
  , iterations_(0)
  { }

  auto& get_graph() { return graph_; }
  const auto& get_graph() const { return graph_; }

  cost lower_bound() const
  {
    cost result = 0;
    for (const auto& timestep : graph_.timesteps()) {
      for (const auto* node : timestep.detections)
        result += node->detection.lower_bound();

      for (const auto* node : timestep.conflicts)
        result += node->conflict.lower_bound();
    }

    return result;
  }

  cost evaluate_primal() const
  {
    const cost inf = std::numeric_limits<cost>::infinity();
    cost result = 0;

    for (const auto& timestep : graph_.timesteps()) {
      for (const auto* node : timestep.detections) {
        if (!transition_messages::check_primal_consistency(*node))
          result += inf;
        result += node->detection.evaluate_primal();
      }

      for (const auto* node : timestep.conflicts) {
        if (!conflict_messages::check_primal_consistency(*node))
          result += inf;
        result += node->conflict.evaluate_primal();
      }
    }

    return result;
  }

  cost upper_bound() const { return evaluate_primal(); }

  void reset_primal()
  {
    for (const auto& timestep : graph_.timesteps()) {
      for (const auto* node : timestep.detections)
        node->detection.reset_primal();

      for (const auto* node : timestep.conflicts)
        node->conflict.reset_primal();
    }
  }

  // This method is only needed for external rounding code.
  template<bool forward>
  void single_step(const index timestep_idx)
  {
    const auto& timesteps = graph_.timesteps();
    assert(timestep_idx >= 0 && timestep_idx < timesteps.size());
    single_step<forward, false>(timesteps[timestep_idx]); // Rounding is disabled here.
  }

  template<bool forward, bool rounding>
  void single_pass()
  {
#ifndef NDEBUG
    auto lb_before = lower_bound();
#endif

    auto runner = [&](auto begin, auto end) {
      for (auto it = begin; it != end; ++it) {
        this->single_step<forward, rounding>(*it);
      }
    };

    const auto& timesteps = graph_.timesteps();
    if constexpr (forward)
      runner(timesteps.begin(), timesteps.end());
    else
      runner(timesteps.rbegin(), timesteps.rend());

    if constexpr (rounding)
      for (const auto& timestep : timesteps)
        for (const auto* node : timestep.detections)
          node->detection.fix_primal();

#ifndef NDEBUG
    auto lb_after = lower_bound();
    assert(lb_before <= lb_after + epsilon);
#endif
  }

  template<bool rounding=false> void forward_pass() { single_pass<true, rounding>(); }
  template<bool rounding=false> void backward_pass() { single_pass<false, rounding>(); }

  void run(const int max_batches = 1000 / config_batch)
  {
    assert(graph_.is_prepared());
    const auto& timesteps = graph_.timesteps();

    std::vector<detection_primal> best_detection_primals(graph_.number_of_detections());
    std::vector<conflict_primal> best_conflict_primals(graph_.number_of_conflicts());
    cost best_ub = std::numeric_limits<cost>::infinity();

    auto visit_primal_storage = [&](auto functor) {
      auto it_detection = best_detection_primals.begin();
      auto it_conflict = best_conflict_primals.begin();
      for (const auto& t : timesteps) {
        for (const auto* node : t.detections) {
          assert(it_detection != best_detection_primals.end());
          functor(it_detection++, node->detection);
        }
        for (const auto* node : t.conflicts) {
          assert(it_conflict != best_conflict_primals.end());
          functor(it_conflict++, node->conflict);
        }
      }
      assert(it_detection == best_detection_primals.end());
      assert(it_conflict == best_conflict_primals.end());
    };

    auto remember_best_primals = [&]() {
      auto ub = evaluate_primal();
      if (ub < best_ub) {
        best_ub = ub;
        visit_primal_storage([&](auto it, const auto& f) { *it = f.primal(); });
      }
    };

    auto restore_best_primals = [&] () {
      visit_primal_storage([&](auto it, auto& f) { f.primal() = *it++; });
    };

    signal_handler h;
    using clock_type = std::chrono::high_resolution_clock;
    const auto clock_start = clock_type::now();
    std::cout.precision(std::numeric_limits<cost>::max_digits10);
    for (int i = 0; i < max_batches && !h.signaled(); ++i) {
      for (int j = 0; j < config_batch-1; ++j) {
        forward_pass<false>();
        backward_pass<false>();
      }

      reset_primal();
      forward_pass<true>();
      remember_best_primals();

      reset_primal();
      backward_pass<true>();
      remember_best_primals();

      const auto clock_now = clock_type::now();
      const std::chrono::duration<double> seconds = clock_now - clock_start;

      const auto lb = lower_bound();
      iterations_ += config_batch;
      std::cout << "it=" << iterations_ << " "
                << "lb=" << lb << " "
                << "ub=" << best_ub << " "
                << "gap=" << static_cast<float>(100.0 * (best_ub - lb) / std::abs(lb)) << "% "
                << "t=" << seconds.count() << std::endl;
    }

    restore_best_primals();
  }

protected:
  template<bool forward, bool rounding>
  void single_step(const timestep_type& t)
  {
    for (const auto* node : t.conflicts)
      conflict_messages::send_messages_to_conflict(*node);

    for (const auto* node : t.conflicts)
      conflict_messages::send_messages_to_detection(*node);

    if constexpr (rounding) {
      // FIXME: Pre-allocate scratch space and do not resort to dynamic
      // memory allocation.
      std::vector<typename graph_type::detection_node_type*> sorted_detections(t.detections.cbegin(), t.detections.cend());
      std::sort(sorted_detections.begin(), sorted_detections.end(),
        [](const auto* a, const auto* b) {
          const auto va = a->detection.min_detection();
          const auto vb = b->detection.min_detection();
          return va < vb;
        });

      for (const auto* node : sorted_detections) {
        // Checks that all messages are either consistent or unknown but not
        // inconsitent. This property must be invariant during rounding, so we
        // verify that it is the case.
        auto check_messages = [&]() {
#ifndef NDEBUG
          assert(transition_messages::check_primal_consistency(*node).is_not_inconsistent());
          for (const auto& edge : node->conflicts)
            assert(conflict_messages::check_primal_consistency(*edge.node).is_not_inconsistent());
#endif
        };

        std::array<bool, max_number_of_detection_edges + 1> possible;
        transition_messages::get_primal_possibilities<forward>(*node, possible);

        node->detection.template round_primal<forward>(possible); check_messages();
        transition_messages::propagate_primal<!forward>(*node); check_messages();
        for (const auto& edge : node->conflicts) {
          conflict_messages::propagate_primal_to_conflict(*edge.node); check_messages();
          conflict_messages::propagate_primal_to_detections(*edge.node); check_messages();
        }
      }
    }

    for (const auto* node : t.detections)
      transition_messages::send_messages<forward>(*node);
  }

  graph_type graph_;
  factor_counter factor_counter_;
  int iterations_;
};

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
