#ifndef LIBMPOPT_CT_CONFLICT_SUBSOLVER_HPP
#define LIBMPOPT_CT_CONFLICT_SUBSOLVER_HPP

namespace mpopt {
namespace ct {

class conflict_subsolver {
public:
  size_t add_node(cost c)
  {
    auto idx = nodes_.size();
    nodes_.push_back(c);
    conflicts_.emplace_back();
    assert(nodes_.size() == conflicts_.size());
    return idx;
  }

  void add_conflict(size_t node1, size_t node2)
  {
    assert(node1 >= 0 && node1 < nodes_.size());
    assert(node2 >= 0 && node2 < nodes_.size());
    //assert(node1 < node2);
    assert(nodes_.size() == conflicts_.size());
    conflicts_[node1].push_back(node2);
    conflicts_[node2].push_back(node1);
  }

  auto size() const { return nodes_.size(); }

  void optimize(GRBEnv& env)
  {
#if 0
    std::cout << "Optimizing model with " << nodes_.size() << " nodes." << std::endl;
#endif
    assert(nodes_.size() == conflicts_.size());

#if 0
    objective_ = std::numeric_limits<cost>::infinity();
    std::vector<bool> assignment(nodes_.size(), true);
    exhaustive_recurse(assignment, 0, 0);
#else
    std::vector<GRBVar> vars(nodes_.size());
    GRBModel model(env);
    model.set(GRB_IntParam_OutputFlag, 0);

    for (size_t i = 0; i < nodes_.size(); ++i) {
      vars[i] = model.addVar(0, 1, nodes_[i], GRB_BINARY);
    }

    for (size_t i = 0; i < conflicts_.size(); ++i) {
      for (auto j : conflicts_[i]) {
        model.addConstr(vars[i] + vars[j] <= 1);
      }
    }

    model.optimize();
    assignment_.resize(nodes_.size());
    for (size_t i = 0; i < nodes_.size(); ++i) {
      assignment_[i] = vars[i].get(GRB_DoubleAttr_X) > .5;
    }
#endif
  }

  bool assignment(size_t node) const
  {
    assert(node >= 0 && node < nodes_.size());
    assert(nodes_.size() == assignment_.size());
    return assignment_[node];
  }

protected:
  void exhaustive_recurse(std::vector<bool>& assignment, size_t cur, cost objective)
  {
    if (cur == assignment.size()) {
#if 0
      for (auto x : assignment) { std::cout << x << " "; }
      std::cout << std::endl;
#endif
      if (objective < objective_) {
        objective_ = objective;
        assignment_ = assignment;
      }
      return;
    }

    if (assignment[cur]) {
      auto new_assignment = assignment;
      for (auto conflict : conflicts_[cur])
        new_assignment[conflict] = false;

      exhaustive_recurse(new_assignment, cur+1, objective + nodes_[cur]);
      assignment[cur] = false;
    }

    exhaustive_recurse(assignment, cur+1, objective);
  };

  std::vector<cost> nodes_;
  std::vector<std::vector<size_t>> conflicts_;
  std::vector<bool> assignment_;
  cost objective_;
};


template<typename GRAPH_TYPE>
class conflict_walker {
public:
  using graph_type = GRAPH_TYPE;
  using detection_node_type = typename graph_type::detection_node_type;
  using conflict_node_type = typename graph_type::conflict_node_type;

  void walk(const conflict_node_type* node, conflict_subsolver& subsolver)
  {
    if (visited_.find(node) == visited_.end()) {
      visited_.insert(node);
      for (const auto& edge : node->detections) {
        if (edge.node->factor.primal().is_detection_off())
          continue;

        if (factor_to_variable_.find(edge.node) == factor_to_variable_.end()) {
          auto var = subsolver.add_node(edge.node->factor.min_detection());
          factor_to_variable_.insert(std::pair(edge.node, var));
          variable_to_factor_.insert(std::pair(var, edge.node));
        }
      }

      for (size_t i = 0; i < node->detections.size(); ++i) {
        if (node->detections[i].node->factor.primal().is_detection_off())
          continue;

        for (size_t j = i+1; j < node->detections.size(); ++j) {
          if (node->detections[j].node->factor.primal().is_detection_off())
            continue;

          assert(factor_to_variable_.find(node->detections[i].node) != factor_to_variable_.end());
          assert(factor_to_variable_.find(node->detections[j].node) != factor_to_variable_.end());

          subsolver.add_conflict(
            factor_to_variable_[node->detections[i].node],
            factor_to_variable_[node->detections[j].node]);
        }
      }

      for (const auto& edge : node->detections) {
        if (edge.node->factor.primal().is_detection_off())
          continue;

        for (const auto& edge2 : edge.node->conflicts)
          walk(edge2.node, subsolver);
      }
    }
  }

  const detection_node_type* variable_to_factor(size_t variable)
  {
    return variable_to_factor_[variable];
  }

private:
  std::set<const void*> visited_;
  std::map<const detection_node_type*, size_t> factor_to_variable_;
  std::map<size_t, const detection_node_type*> variable_to_factor_;
};

template<typename GRAPH_TYPE>
class conflict_subsolver2 {
public:
  using graph_type = GRAPH_TYPE;
  using detection_node_type = typename graph_type::detection_node_type;
  using conflict_node_type = typename graph_type::conflict_node_type;

  conflict_subsolver2(GRBEnv& env)
  : model_(env)
  {
    model_.set(GRB_IntParam_OutputFlag, 0);
  }

  void add_detection(const detection_node_type* node)
  {
    assert(factor_to_variable_.find(node) == factor_to_variable_.end());
    factor_to_variable_[node] = model_.addVar(0, 1, node->factor.min_detection(), GRB_BINARY);
  }

  void add_conflict(const conflict_node_type* node)
  {
    GRBLinExpr expr;
    for (const auto& edge : node->detections)
      expr += factor_to_variable_.at(edge.node);
    model_.addConstr(expr <= 1);
  }

  void optimize()
  {
    model_.optimize();
  }

  bool assignment(const detection_node_type* node)
  {
    return factor_to_variable_.at(node).get(GRB_DoubleAttr_X) > .5;
  }

private:
  GRBModel model_;
  std::map<const detection_node_type*, GRBVar> factor_to_variable_;
};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
