#ifndef LIBMPOPT_CT_CONFLICT_SUBSOLVER_HPP
#define LIBMPOPT_CT_CONFLICT_SUBSOLVER_HPP

namespace mpopt {
namespace ct {

template<typename GRAPH_TYPE>
class conflict_subsolver {
public:
  using graph_type = GRAPH_TYPE;
  using detection_node_type = typename graph_type::detection_node_type;
  using conflict_node_type = typename graph_type::conflict_node_type;

  conflict_subsolver(GRBEnv& env)
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
