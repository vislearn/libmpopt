#ifndef LIBMPOPT_CT_COMBILP_HPP
#define LIBMPOPT_CT_COMBILP_HPP

namespace mpopt {
namespace ct {

template<typename ALLOCATOR>
class combilp {
public:
  using allocator_type = ALLOCATOR;
  using graph_type = graph<allocator_type>;
  using detection_node_type = typename graph_type::detection_node_type;
  using conflict_node_type = typename graph_type::conflict_node_type;

  combilp(const graph_type& graph)
  : graph_(&graph)
  , iterations_(0)
  , ilp_time_(0)
  {
  }

  auto ilp_time() const { return ilp_time_; }

  void run()
  {
    dbg::timer t;
    GRBEnv env;
    gurobi_model_builder<allocator_type> builder(env);
    preprocess();

    bool changed = true;
    while (changed) {
      populate_builder(builder);

      std::cout << "\n== CombiLP iteration " << (iterations_+1) << " ==" << std::endl;
      output_mask_size();

      builder.finalize();
      t.start();
      builder.optimize();
      t.stop();
      builder.update_primals();

      changed = process_mismatches();
      ++iterations_;
      ilp_time_ += t.seconds();
    }

#ifndef NDEBUG
    for (const auto& timestep : graph_->timesteps()) {
      for (const auto* node : timestep.detections)
        assert(transition_messages::check_primal_consistency(node));

      for (const auto* node : timestep.conflicts)
        assert(conflict_messages::check_primal_consistency(node));
    }
#endif
  }

protected:
  void output_mask_size() const
  {
    size_t inconsistent_detections=0, total_detections=0;
    size_t inconsistent_conflicts=0, total_conflicts=0;
    for (const auto& timestep : graph_->timesteps()) {
      for (const auto* node : timestep.detections) {
        ++total_detections;
        if (!mask_sac_d_.at(node))
          ++inconsistent_detections;
      }

      for (const auto* node : timestep.conflicts) {
        ++total_conflicts;
        if (!mask_sac_c_.at(node))
          ++inconsistent_conflicts;
      }
    }

    std::cout << "clp detections: " << inconsistent_detections << " / " << total_detections << " (" << (100.0 * inconsistent_detections / total_detections) << "%)\n"
              << "clp conflicts: " << inconsistent_conflicts << " / " << total_conflicts << " (" << (100.0 * inconsistent_conflicts / total_conflicts) << "%)" << std::endl;
  }

  template<typename BUILDER>
  void populate_builder(BUILDER& builder) const
  {
    for (const auto& timestep : graph_->timesteps()) {
      for (const auto* node : timestep.detections) {
        if (!mask_sac_d_.at(node)) {
          node->factor.reset_primal();
          builder.add_factor(node);
        }
      }

      for (const auto* node : timestep.conflicts) {
        if (!mask_sac_c_.at(node)) {
          node->factor.reset_primal();
          builder.add_factor(node);
        }
      }
    }
  }

  void preprocess()
  {
    for (const auto& timestep : graph_->timesteps())
      for (const auto* node : timestep.conflicts)
        conflict_messages::send_messages_to_detection(node);

    const auto& ts = graph_->timesteps();
    for (int iteration = 0; iteration < 10; ++iteration) {
      for (auto it = ts.begin(); it != ts.end(); ++it)
        for (const auto* node : it->detections)
          transition_messages::send_messages<true>(node, 0.9);

      for (auto it = ts.rbegin(); it != ts.rend(); ++it)
        for (const auto* node : it->detections)
          transition_messages::send_messages<false>(node, 0.9);
    }

    for (const auto& timestep : graph_->timesteps()) {
      for (const auto* node : timestep.detections) {
        node->factor.reset_primal();
        node->factor.round_independently();
      }
    }

    mask_sac_d_.clear();
    for (const auto& timestep : graph_->timesteps())
      for (const auto* node : timestep.detections)
        mask_sac_d_[node] = transition_messages::check_primal_consistency(node);

    mask_sac_c_.clear();
    for (const auto& timestep : graph_->timesteps()) {
      for (const auto* node : timestep.conflicts) {
        const bool is_sac = set_consistent_conflict_primal_if_possible(node);
        mask_sac_c_[node] = is_sac;

        if (!is_sac) {
          for (const auto& edge : node->detections)
            mask_sac_d_.at(edge.node) = false;
        }
      }
    }
  }

  bool set_consistent_conflict_primal_if_possible(const conflict_node_type* node)
  {
    node->factor.reset_primal();
    bool is_first = true;

    auto process_primal = [&](const index label) {
      if (is_first) {
        is_first = false;
        node->factor.primal().set(label);
      } else {
        if (node->factor.primal().get() != label)
          node->factor.reset_primal();
      }
    };

    node->traverse_detections([&process_primal](const auto& edge, const index slot) {
      auto& f = edge.node->factor;
      assert(f.primal().is_incoming_set());
      assert(f.primal().is_outgoing_set());
      if (f.primal().is_detection_on())
        process_primal(slot);
    });

    // Set it to the dummy label if no detection voted for being active.
    if (is_first) {
      node->factor.primal().set(node->factor.size() - 1);
    }

    // Verify that the current configuration is in fact a local minimizer.
    // If not we mark it as inconsistent.
    if (node->factor.primal().is_set())
      if (node->factor.evaluate_primal() > node->factor.lower_bound() + epsilon)
        node->factor.reset_primal();

    bool is_sac = node->factor.primal().is_set();
    assert(is_sac == conflict_messages::check_primal_consistency(node));
    return is_sac;
  }

  bool process_mismatches()
  {
    bool changed = false;

    for (const auto& timestep : graph_->timesteps()) {
      for (const auto* node : timestep.detections) {
        if (mask_sac_d_.at(node) && !transition_messages::check_primal_consistency(node)) {
          mask_sac_d_.at(node) = false;
          changed = true;
        }
      }

      for (const auto* node : timestep.conflicts) {
        if (mask_sac_c_.at(node) && !conflict_messages::check_primal_consistency(node)) {
          mask_sac_c_.at(node) = false;
          for (const auto& edge : node->detections)
            mask_sac_d_.at(edge.node) = false;
          changed = true;
        }
      }
    }

    return changed;
  }

  const graph_type* graph_;
  int iterations_;
  double ilp_time_;
  std::map<const detection_node_type*, bool> mask_sac_d_;
  std::map<const conflict_node_type*, bool> mask_sac_c_;
};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
