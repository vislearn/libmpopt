#ifndef LIBMPOPT_CT_GUROBI_HPP
#define LIBMPOPT_CT_GUROBI_HPP

namespace mpopt {
namespace ct {

#ifdef ENABLE_GUROBI

template<typename ALLOCATOR = std::allocator<cost>>
class gurobi_model_builder
{
public:
  using allocator_type = ALLOCATOR;
  using graph_type = graph<allocator_type>;
  using detection_node_type = typename graph_type::detection_node_type;
  using conflict_node_type = typename graph_type::conflict_node_type;
  using gurobi_detection_type = gurobi_detection_factor<allocator_type>;
  using gurobi_conflict_type = gurobi_conflict_factor<allocator_type>;

  gurobi_model_builder(GRBEnv& env)
  : model_(env)
  , finalized_(false)
  { }

  void set_constant(const cost constant)
  {
    model_.set(GRB_DoubleAttr_ObjCon, constant);
  }

  void add_factor(const detection_node_type* node)
  {
    assert(node != nullptr);
    if (detections_.try_emplace(node, node->factor, model_).second)
      new_detections_.push_back(node);
    finalized_ = false;
  }

  void add_factor(const conflict_node_type* node)
  {
    assert(node != nullptr);
    if (conflicts_.try_emplace(node, node->factor, model_).second)
      new_conflicts_.push_back(node);
    finalized_ = false;
  }

  void finalize()
  {
    if (!finalized_) {
      for (const auto* node : new_detections_)
        add_linear_constraints(node);
      new_detections_.clear();

      for (const auto* node : new_conflicts_)
        add_linear_constraints(node);
      new_conflicts_.clear();

      finalized_ = true;
    } else {
      assert(new_detections_.size() == 0);
      assert(new_conflicts_.size() == 0);
    }
  }

  void optimize()
  {
    assert(finalized_);
    std::cout << "Going to optimize an ILP with " << detections_.size() << " + " << conflicts_.size() << " factors" << std::endl;

    model_.set(GRB_IntParam_Threads, 1); // single thread
    model_.set(GRB_IntParam_Method, 1); // dual simplex
    model_.optimize();
    assert(model_.get(GRB_IntAttr_Status) == GRB_OPTIMAL);
  }

  void update_primals()
  {
    for (auto& x : detections_)
      x.second.update_primal();

    for (auto& x : conflicts_)
      x.second.update_primal();
  }

protected:
  void add_linear_constraints(const detection_node_type* node)
  {
    // FIXME: As this function gets called for both factors if two neighboring
    // factors are in the `new_detections_` vector, the associated constraints
    // will be inserted twice.
    //
    // This should not be a big problem: Gurobi should eliminate these
    // duplicated constraints in the preprocessing pass.
    add_detection_linear_constraints_impl<false>(node);
    add_detection_linear_constraints_impl<true>(node);
  }

  void add_linear_constraints(const conflict_node_type* node)
  {
    auto& conf = conflicts_.at(node);
    node->traverse_detections([&](const auto& edge, auto slot) {
      auto it = detections_.find(edge.node);
      if (it != detections_.end()) {
        auto& det = it->second;
        model_.addConstr(conf.variable(slot) == det.detection());
      }
    });
  }

  template<bool forward>
  void add_detection_linear_constraints_impl(const detection_node_type* node)
  {
    auto& det = detections_.at(node);

    auto process_neighboring_node = [&](const index this_slot, const detection_node_type* neighbor, const index neighbor_slot) {
      auto it = detections_.find(neighbor);
      if (it != detections_.end()) {
        auto& this_var = det.template transition<forward>(this_slot);
        auto& neighbor_var = it->second.template transition<!forward>(neighbor_slot);
        model_.addConstr(this_var == neighbor_var);
      }
    };

    node->template traverse_transitions<forward>([&](const auto& edge, auto slot) {
      process_neighboring_node(slot, edge.node1, edge.slot1);
      if (forward && edge.is_division())
        process_neighboring_node(slot, edge.node2, edge.slot2);
    });
  }

  GRBModel model_;
  bool finalized_;
  std::map<const detection_node_type*, gurobi_detection_type> detections_;
  std::map<const conflict_node_type*, gurobi_conflict_type> conflicts_;
  std::vector<const detection_node_type*> new_detections_;
  std::vector<const conflict_node_type*> new_conflicts_;
};

#endif

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
