#ifndef LIBMPOPT_QAP_GUROBI_HPP
#define LIBMPOPT_QAP_GUROBI_HPP

namespace mpopt {
namespace qap {

template<typename ALLOCATOR>
class gurobi_unary_factor {
public:
  using allocator_type = ALLOCATOR;
  using factor_type = unary_factor<allocator_type>;

  gurobi_unary_factor(factor_type& factor, GRBModel& model)
  : factor_(&factor)
  , vars_(factor.size())
  {
    std::vector<double> coeffs(vars_.size(), 1.0);
    for (size_t i = 0; i < factor_->size(); ++i)
      vars_[i] = model.addVar(0.0, 1.0, factor_->get(i), GRB_BINARY);

    GRBLinExpr expr;
    expr.addTerms(coeffs.data(), vars_.data(), vars_.size());
    model.addConstr(expr == 1);
  }

  void update_primal() const
  {
    auto& p = factor_->primal();
    p = factor_type::primal_unset;
    for (size_t i = 0; i < vars_.size(); ++i) {
      if (vars_[i].get(GRB_DoubleAttr_X) >= 0.5) {
        assert(p == factor_type::primal_unset);
        p = i;
      }
    }
    assert(p != factor_type::primal_unset);
  }

  auto& variable(index i) { assert(i >= 0 && i < vars_.size()); return vars_[i]; }
  const auto& factor() const { assert(factor_ != nullptr); return *factor_; }

protected:
  factor_type* factor_;
  std::vector<GRBVar> vars_;
};


template<typename ALLOCATOR>
class gurobi_uniqueness_factor {
public:
  using allocator_type = ALLOCATOR;
  using factor_type = uniqueness_factor<allocator_type>;

  gurobi_uniqueness_factor(factor_type& factor, GRBModel& model)
  : factor_(&factor)
  , vars_(factor.costs_.size())
  {
    std::vector<double> coeffs(vars_.size(), 1.0);
    for (size_t i = 0; i < vars_.size(); ++i)
      vars_[i] = model.addVar(0.0, 1.0, factor_->costs_[i], GRB_BINARY);

    GRBLinExpr expr;
    expr.addTerms(coeffs.data(), vars_.data(), vars_.size());
    model.addConstr(expr == 1);
  }

  void update_primal() const
  {
    auto& p = factor_->primal();
    p = factor_type::primal_unset;
    for (size_t i = 0; i < vars_.size(); ++i) {
      if (vars_[i].get(GRB_DoubleAttr_X) >= 0.5) {
        assert(p == factor_type::primal_unset);
        p = i;
      }
    }
    assert(p != factor_type::primal_unset);
  }

  auto& variable(index i) { assert(i >= 0 && i < vars_.size()); return vars_[i]; }
  const auto& factor() const { assert(factor_ != nullptr); return *factor_; }

protected:
  factor_type* factor_;
  std::vector<GRBVar> vars_;
};


template<typename ALLOCATOR>
class gurobi_pairwise_factor {
public:
  using allocator_type = ALLOCATOR;
  using factor_type = pairwise_factor<allocator_type>;

  gurobi_pairwise_factor(factor_type& factor, GRBModel& model)
  : factor_(&factor)
  , vars_(factor.no_labels0_ * factor.no_labels1_)
  {
    std::vector<double> coeffs(vars_.size(), 1.0);
    for (size_t i = 0; i < vars_.size(); ++i)
      vars_[i] = model.addVar(0.0, 1.0, factor_->costs_[i], GRB_CONTINUOUS);

    GRBLinExpr expr;
    expr.addTerms(coeffs.data(), vars_.data(), vars_.size());
    model.addConstr(expr == 1);
  }

  void update_primal() const
  {
    auto& p0 = factor_->primal0_;
    auto& p1 = factor_->primal1_;
    p0 = p1 = factor_type::primal_unset;

    for (size_t i = 0; i < vars_.size(); ++i) {
      if (vars_[i].get(GRB_DoubleAttr_X) >= 0.5) {
        assert(p0 == factor_type::primal_unset && p1 == factor_type::primal_unset);
        const auto indices = factor_->to_nonlinear(i);
        p0 = std::get<0>(indices);
        p1 = std::get<1>(indices);
      }
    }

    assert(p0 != factor_type::primal_unset && p1 != factor_type::primal_unset);
  }

  auto& variable(index i) { assert(i >= 0 && i < vars_.size()); return vars_[i]; }
  const auto& factor() const { assert(factor_ != nullptr); return *factor_; }

protected:
  factor_type* factor_;
  std::vector<GRBVar> vars_;
};


template<typename ALLOCATOR>
class gurobi_model_builder {
public:
  using allocator_type = ALLOCATOR;
  using graph_type = graph<allocator_type>;
  using unary_node_type = typename graph_type::unary_node_type;
  using uniqueness_node_type = typename graph_type::uniqueness_node_type;
  using pairwise_node_type = typename graph_type::pairwise_node_type;
  using gurobi_unary_type = gurobi_unary_factor<allocator_type>;
  using gurobi_uniqueness_type = gurobi_uniqueness_factor<allocator_type>;
  using gurobi_pairwise_type = gurobi_pairwise_factor<allocator_type>;

  gurobi_model_builder(GRBEnv& env)
  : model_(env)
  , finalized_(false)
  { }

  void set_constant(const cost constant)
  {
    model_.set(GRB_DoubleAttr_ObjCon, constant);
  }

  void add_factor(const unary_node_type* node)
  {
    assert(node != nullptr);
    assert(!finalized_);
    // TODO: Do we really want this check? We already use try_emplace.
    assert(unaries_.find(node) == unaries_.end());
    unaries_.try_emplace(node, node->unary, model_);
  }

  void add_factor(const uniqueness_node_type* node)
  {
    assert(node != nullptr);
    assert(!finalized_);
    // TODO: Do we really want this check? We already use try_emplace.
    assert(uniqueness_.find(node) == uniqueness_.end());
    uniqueness_.try_emplace(node, node->uniqueness, model_);
  }

  void add_factor(const pairwise_node_type* node)
  {
    assert(node != nullptr);
    assert(!finalized_);
    // TODO: Do we really want this check? We already use try_emplace.
    assert(pairwise_.find(node) == pairwise_.end());
    pairwise_.try_emplace(node, node->pairwise, model_);
  }

  void finalize()
  {
    if (!finalized_) {
      add_linear_constraints();
      finalized_ = true;
    }
  }

  void optimize()
  {
    assert(finalized_);
    std::cout << "Going to optimize an ILP with " << unaries_.size() << " + " << uniqueness_.size() << " + " << pairwise_.size() << " factors" << std::endl;

    model_.set(GRB_IntParam_Threads, 1); // single thread
    model_.set(GRB_IntParam_Method, 1); // dual simplex
    model_.optimize();
    assert(model_.get(GRB_IntAttr_Status) == GRB_OPTIMAL);
  }

  void update_primals()
  {
    for (auto& x : unaries_)
      x.second.update_primal();

    for (auto& x : uniqueness_)
      x.second.update_primal();

    for (auto& x : pairwise_)
      x.second.update_primal();
  }

protected:
  void add_linear_constraints()
  {
    // We assume that both unaries of a pairwise edge are present in the ILP.
    // Otherwise we can not mark the variables for the pairwise edge
    // `GRB_CONTINUOUS`.
    for (auto& pair : pairwise_) {
      const auto* pairwise_node = pair.first;
      auto& pairwise = pair.second;
      assert(unaries_.count(pairwise_node->unary0) == 1 && unaries_.count(pairwise_node->unary1) == 1);

      auto& unary0 = unaries_.at(pairwise_node->unary0);
      for (index idx0 = 0; idx0 < pairwise_node->pairwise.no_labels0_; ++idx0) {
        GRBLinExpr expr;
        for (index idx1 = 0; idx1 < pairwise_node->pairwise.no_labels1_; ++idx1)
          expr += pairwise.variable(pairwise_node->pairwise.to_linear(idx0, idx1));
        model_.addConstr(expr == unary0.variable(idx0));
      }

      auto& unary1 = unaries_.at(pairwise_node->unary1);
      for (index idx1 = 0; idx1 < pairwise_node->pairwise.no_labels1_; ++idx1) {
        GRBLinExpr expr;
        for (index idx0 = 0; idx0 < pairwise_node->pairwise.no_labels0_; ++idx0)
          expr += pairwise.variable(pairwise_node->pairwise.to_linear(idx0, idx1));
        model_.addConstr(expr == unary1.variable(idx1));
      }
    }

    // Not all unaries of a uniqueness factor have to be present in the ILP.
    for (auto& pair : uniqueness_) {
      const auto* uniqueness_node = pair.first;
      auto& uniqueness = pair.second;

      uniqueness_node->traverse_unaries([&](const auto& link, index slot) {
        auto it = unaries_.find(link.node);
        if (it != unaries_.end())
          model_.addConstr(uniqueness.variable(slot) == it->second.variable(link.slot));
      });
    }
  }

  GRBModel model_;
  bool finalized_;
  std::map<const unary_node_type*, gurobi_unary_type> unaries_;
  std::map<const uniqueness_node_type*, gurobi_uniqueness_type> uniqueness_;
  std::map<const pairwise_node_type*, gurobi_pairwise_type> pairwise_;
};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
