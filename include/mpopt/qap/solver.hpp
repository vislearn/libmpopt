#ifndef LIBMPOPT_QAP_SOLVER_HPP
#define LIBMPOPT_QAP_SOLVER_HPP

namespace mpopt {
namespace qap {

constexpr int default_greedy_generations = 10;

enum class pairwise_update_kind { normal_mplp_plus_plus, diffused_mplp_plus_plus };
constexpr auto PAIRWISE_UPDATE_KIND = pairwise_update_kind::diffused_mplp_plus_plus;

template<typename ALLOCATOR>
class solver : public ::mpopt::solver<solver<ALLOCATOR>> {
public:
  using base_type = ::mpopt::solver<solver<ALLOCATOR>>;
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
  , local_search_(graph_)
  , qpbo_(graph_)
  {
    greedy_ = std::make_unique<greedy<ALLOCATOR>>(graph_);
#ifndef ENABLE_QPBO
    std::cerr << "!!!!!!!!!!\n"
              << "ENABLE_QPBO was not activated during configuration of libmpopt.\n"
              << "No fusion moves are performed and the the quality of the computed upper bound is degraded.\n"
              << "!!!!!!!!!!\n" << std::endl;
#endif
  }

  auto& get_graph() { return graph_; }
  const auto& get_graph() const { return graph_; }

  void set_fusion_moves_enabled(bool enabled) {
    fusion_moves_enabled_ = enabled;
  }

  void set_dual_updates_enabled(bool enabled) {
    dual_updates_enabled_ = enabled;
  }

  void set_local_search_enabled(bool enabled) {
    local_search_enabled_ = enabled;
  }

  void set_grasp_alpha(double alpha) {
    assert(0 < alpha && alpha <= 1);
    grasp_alpha_ = alpha;
  }

  void use_grasp() {
    if (grasp_ == nullptr) {
      greedy_.reset();
      grasp_ = std::make_unique<grasp<ALLOCATOR>>(graph_, grasp_alpha_);
    }
  }

  void use_greedy() {
    if (greedy_ == nullptr) {
      grasp_.reset();
      greedy_ = std::make_unique<greedy<ALLOCATOR>>(graph_);
    }
  }

  void run(const int batch_size=default_batch_size, const int max_batches=default_max_batches, int greedy_generations=default_greedy_generations)
  {
    graph_.check_structure();
    cost best_ub = infinity;

    signal_handler h;
    std::cout.precision(std::numeric_limits<cost>::max_digits10);

    if (!dual_updates_enabled_) {
      for (const auto* node : graph_.pairwise()) {
        pairwise_messages::send_messages_to_unaries(node);
      }
    }

    for (int i = 0; i < max_batches && !h.signaled(); ++i) {
      const auto clock_start = clock_type::now();

      for (int j = 0; j < batch_size-1; ++j)
        single_pass<false>(greedy_generations);

      single_pass<true>(greedy_generations);

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

  void compute_greedy_assignment()
  {
    if (greedy_ != nullptr) {
      greedy_->run();
    } else {
      grasp_->run();
    }

    if (local_search_enabled_) {
      local_search_.run();
    }
    assert(this->check_primal_consistency());
  }

  void execute_combilp()
  {
    if (std::abs(this->evaluate_primal() - this->lower_bound()) < epsilon) {
      std::cout << "Not starting CombiLP: Problem is tight." << std::endl;
      return;
    }

#ifdef ENABLE_GUROBI
    this->reset_primal();
    combilp subsolver(graph_, graph_.constant());
    subsolver.run();
#else
    abort_on_disabled_gurobi();
#endif
  }

protected:

  template<bool rounding>
  void single_pass(int greedy_generations)
  {
#ifndef NDEBUG
    auto lb_before = this->lower_bound();
#endif

    if (dual_updates_enabled_) {
      if constexpr (PAIRWISE_UPDATE_KIND == pairwise_update_kind::normal_mplp_plus_plus) {
        for (const auto* node : graph_.pairwise())
          pairwise_messages::full_mplp_update(node);
      }

      if constexpr (PAIRWISE_UPDATE_KIND == pairwise_update_kind::diffused_mplp_plus_plus) {
        for (const auto* node : graph_.unaries())
          pairwise_messages::send_messages_to_pairwise(node);

        for (const auto* node : graph_.pairwise())
          pairwise_messages::send_messages_to_unaries(node);
      }
    }

    if constexpr (rounding) {
      primal_storage previous(graph_), current(graph_);
      for (int i = 0; i < greedy_generations; ++i) {
        previous.save();
        const auto previous_ub = this->evaluate_primal();

        compute_greedy_assignment();
        current.save();
        auto current_ub = this->evaluate_primal();

#ifdef ENABLE_QPBO
        if (fusion_moves_enabled_ && previous_ub != infinity) {
          qpbo_.reset();
          index idx = 0;
          for (const auto* node : graph_.unaries()) {
            qpbo_.add_factor(node, previous.get(idx), current.get(idx));
            ++idx;
          }

          for (const auto* node : graph_.uniqueness())
            qpbo_.add_factor(node);

          for (const auto* node : graph_.pairwise())
            qpbo_.add_factor(node);

          qpbo_.enable_improve(true);
          qpbo_.finalize();
          qpbo_.optimize();
          qpbo_.update_primals();
          assert(this->check_primal_consistency());
          const auto fused_ub = this->evaluate_primal();

          if (fused_ub < current_ub)
            current_ub = fused_ub;
          else
            current.restore();
        }
#endif

        if (previous_ub < current_ub)
          previous.restore();
      }
    }

    if (dual_updates_enabled_) {
      for (const auto* node : graph_.unaries()) {
        graph_.add_to_constant(node->factor.normalize());
        uniqueness_messages::send_messages_to_uniqueness(node);
      }

      for (const auto* node : graph_.uniqueness()) {
        graph_.add_to_constant(node->factor.normalize());
        uniqueness_messages::send_messages_to_unaries(node);
      }
    }

#ifndef NDEBUG
    auto lb_after = this->lower_bound();
    assert(lb_before <= lb_after + epsilon);
#endif
  }

  bool fusion_moves_enabled_ = true;
  bool dual_updates_enabled_ = true;
  bool local_search_enabled_ = true;
  double grasp_alpha_ = 0.25;
  graph_type graph_;
  std::unique_ptr<grasp<ALLOCATOR>> grasp_{};
  std::unique_ptr<greedy<ALLOCATOR>> greedy_;
  local_search<ALLOCATOR> local_search_;
#ifdef ENABLE_QPBO
  qpbo_model_builder<ALLOCATOR> qpbo_;
#endif
  friend class ::mpopt::solver<solver<ALLOCATOR>>;
};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
