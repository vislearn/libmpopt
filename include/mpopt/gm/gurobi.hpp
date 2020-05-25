#ifndef LIBMPOPT_GM_GUROBI_HPP
#define LIBMPOPT_GM_GUROBI_HPP

namespace mpopt {
namespace gm {

#ifdef ENABLE_GUROBI

template<typename ALLOCATOR>
class gurobi_model_builder {
public:
  using allocator_type = ALLOCATOR;
  using graph_type = graph<allocator_type>;
  using unary_node_type = typename graph_type::unary_node_type;
  using pairwise_node_type = typename graph_type::pairwise_node_type;
  using gurobi_unary_type = gurobi_unary_factor<allocator_type>;
  using gurobi_pairwise_type = gurobi_pairwise_factor<allocator_type>;

  gurobi_model_builder(GRBEnv& env)
  : model_(env)
  { }

  void set_constant(const cost constant)
  {
    model_.set(GRB_DoubleAttr_ObjCon, constant);
  }

  void add_factor(const unary_node_type* node)
  {
    assert(node != nullptr);
    unaries_.try_emplace(node, node->factor, model_);
  }

  void add_factor(const pairwise_node_type* node)
  {
    assert(node != nullptr);
    if (pairwise_.try_emplace(node, node->factor, model_).second)
      add_linear_constraint(node);
  }

  void finalize()
  {
  }

  void optimize()
  {
    std::cout << "Going to optimize an ILP with " << unaries_.size() << " + " << pairwise_.size() << " factors" << std::endl;

    model_.set(GRB_IntParam_Threads, 1); // single thread
    model_.set(GRB_IntParam_Method, 1); // dual simplex
    model_.optimize();
    assert(model_.get(GRB_IntAttr_Status) == GRB_OPTIMAL);
  }

  void update_primals()
  {
    for (auto& x : unaries_)
      x.second.update_primal();

    for (auto& x : pairwise_)
      x.second.update_primal();
  }

protected:
  void add_linear_constraint(const pairwise_node_type* node)
  {
    // We assume that both unaries of a pairwise edge are present in the ILP.
    // Otherwise we can not mark the variables for the pairwise edge as
    // `GRB_CONTINUOUS`.
    auto& pairwise = pairwise_.at(node);
    const auto [size0, size1] = node->factor.size();
    two_dimension_array_accessor a(size0, size1);

    auto& unary0 = unaries_.at(node->unary0);
    for (index idx0 = 0; idx0 < size0; ++idx0) {
      GRBLinExpr expr;
      for (index idx1 = 0; idx1 < size1; ++idx1)
        expr += pairwise.variable(a.to_linear(idx0, idx1));
      model_.addConstr(expr == unary0.variable(idx0));
    }

    auto& unary1 = unaries_.at(node->unary1);
    for (index idx1 = 0; idx1 < size1; ++idx1) {
      GRBLinExpr expr;
      for (index idx0 = 0; idx0 < size0; ++idx0)
        expr += pairwise.variable(a.to_linear(idx0, idx1));
      model_.addConstr(expr == unary1.variable(idx1));
    }
  }

  GRBModel model_;
  std::map<const unary_node_type*, gurobi_unary_type> unaries_;
  std::map<const pairwise_node_type*, gurobi_pairwise_type> pairwise_;
};

#endif

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
