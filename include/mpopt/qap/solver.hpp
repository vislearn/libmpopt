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
#ifdef ENABLE_QPBO
  , qpbo_(graph_)
#endif
  , primals_best_(graph_)
  , primals_candidate_(graph_)
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

  void set_random_seed(const unsigned long seed) {
    if (greedy_ == nullptr) {
      grasp_->set_random_seed(seed); 
    } else {
      greedy_->set_random_seed(seed);
    }
  }

  void run(const int batch_size=default_batch_size, const int max_batches=default_max_batches, int greedy_generations=default_greedy_generations)
  {
    graph_.check_structure();
    primals_best_.resize();
    primals_candidate_.resize();
    ub_best_ = ub_candidate_ = infinity;

    auto dump_assignment = [this]() {
      bool first=true;
      std::cout << "[";
      for (const auto a : primals_best_.assignment()) {
        if (!first)
          std::cout << " ";
        std::cout << a;
        first = false;
      }
      std::cout << "]";
    };

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

      const auto lb = this->lower_bound();
      this->iterations_ += batch_size;

      const auto clock_end = clock_type::now();
      this->duration_ += clock_end - clock_start;

      std::cout << "it=" << this->iterations_ << " "
                << "lb=" << lb << " "
                << "ub=" << ub_best_ << " "
                << "gap=" << static_cast<float>(100.0 * (ub_best_ - lb) / std::abs(lb)) << "% "
                << "t=" << this->runtime() << " "
                << "a=";
      dump_assignment();
      std::cout << "\n";
    }

    // If max_batches is zero the caller does not want to run any dual
    // iterations at all. In those cases we run the greedy heurisitic the
    // specified number of times and fuse the solutions together.
    if (max_batches == 0) {
      const auto lb = this->lower_bound();
      for (int i = 0; i < greedy_generations; ++i) {
        const auto clock_start = clock_type::now();
        primal_step();
        const auto clock_end = clock_type::now();
        this->duration_ += clock_end - clock_start;

        std::cout << "greedy=" << (i+1) << " "
                  << "lb=" << lb << " "
                  << "ub=" << ub_best_ << " "
                  << "gap=" << static_cast<float>(100.0 * (ub_best_ - lb) / std::abs(lb)) << "% "
                  << "t=" << this->runtime() << " "
                  << "a=";
        dump_assignment();
        std::cout << "\n";
      }
    }

    primals_best_.restore();
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

    if constexpr (rounding)
      for (int i = 0; i < greedy_generations; ++i)
        primal_step();

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

  void primal_step() {
    compute_greedy_assignment();
    ub_candidate_ = this->evaluate_primal();

#ifdef ENABLE_QPBO
    primals_candidate_.save();
    if (fusion_moves_enabled_ && ub_best_ != infinity) {
      qpbo_.reset();
      index idx = 0;
      for (const auto* node : graph_.unaries()) {
        qpbo_.add_factor(node, primals_best_.get(idx), primals_candidate_.get(idx));
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
      const auto ub_fused = this->evaluate_primal();

      // Check if one of candidate or fused solution is in fact better.
      if (ub_best_ > std::min(ub_candidate_, ub_fused)) {
        // Now check which one is better, for the candidate-case we have less
        // work to do, check it first.
        if (ub_candidate_ <= ub_fused) {
          primals_best_ = primals_candidate_;
          ub_best_ = ub_candidate_;
        } else {
          // QPBO solver wrote solution into the individual factors, we read
          // them off here.
          primals_best_.save();
          ub_best_ = ub_fused;
        }
      }
    } else if (ub_candidate_ < ub_best_) {
      primals_best_ = primals_candidate_;
      ub_best_ = ub_candidate_;
    }
#else
    if (ub_candidate_ < ub_best_) {
      primals_best_.save();
      ub_best_ = ub_candidate_;
    }
#endif
  }

  graph_type graph_;
  std::unique_ptr<grasp<ALLOCATOR>> grasp_{};
  std::unique_ptr<greedy<ALLOCATOR>> greedy_;
  local_search<ALLOCATOR> local_search_;
#ifdef ENABLE_QPBO
  qpbo_model_builder<ALLOCATOR> qpbo_;
#endif
  primal_storage<ALLOCATOR> primals_best_, primals_candidate_;
  cost ub_best_, ub_candidate_;
  friend class ::mpopt::solver<solver<ALLOCATOR>>;

  bool fusion_moves_enabled_ = true;
  bool dual_updates_enabled_ = true;
  bool local_search_enabled_ = true;
  double grasp_alpha_ = 0.25;
};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
