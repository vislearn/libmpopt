#ifndef LIBQAPOPT_QAP_GUROBI_HPP
#define LIBQAPOPT_QAP_GUROBI_HPP

namespace qapopt {
namespace qap {

#ifdef ENABLE_GUROBI

namespace gurobi_helper {

void setup_gurobi_parameters(GRBModel& model, bool verbose=true) {
  model.set(GRB_IntParam_Threads, 1); // single thread
  model.set(GRB_IntParam_Method, 1); // dual simplex
  model.set(GRB_DoubleParam_MIPGap, 0);
  model.set(GRB_DoubleParam_MIPGapAbs, 0);
  model.set(GRB_IntParam_OutputFlag, verbose ? 1 : 0);
}

}


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

  gurobi_model_builder(GRBEnv& env, bool verbose=true)
  : model_(env)
  {
    gurobi_helper::setup_gurobi_parameters(model_, verbose);
  }

  void set_constant(const cost constant)
  {
    model_.set(GRB_DoubleAttr_ObjCon, constant);
  }

  void add_factor(const unary_node_type* node)
  {
    assert(node != nullptr);
    unaries_.try_emplace(node, node->factor, model_);
  }

  void add_factor(const uniqueness_node_type* node)
  {
    assert(node != nullptr);
    bool did_insert = uniqueness_.try_emplace(node, node->factor, model_).second;
    if (did_insert)
      add_linear_constraint(node);
  }

  void add_factor(const pairwise_node_type* node)
  {
    assert(node != nullptr);
    bool did_insert = pairwise_.try_emplace(node, node->factor, model_).second;
    if (did_insert)
      add_linear_constraint(node);
  }

  void finalize()
  {
  }

  void optimize()
  {
#ifndef NDEBUG
    std::cout << "gurobi_model_builder: Going to optimize an ILP with " << unaries_.size() << " + " << uniqueness_.size() << " + " << pairwise_.size() << " factors" << std::endl;
#endif

    model_.optimize();
    const auto status = model_.get(GRB_IntAttr_Status);

    assert(status == GRB_OPTIMAL);

    if (status != GRB_OPTIMAL) {
      std::ostringstream s;
      s << "Gurobi terminated with status " << status;
      throw std::runtime_error(s.str());
    }
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
  void add_linear_constraint(const pairwise_node_type* node)
  {
    // We assume that both unaries of a pairwise edge are present in the ILP.
    // Otherwise we can not mark the variables for the pairwise edge
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

  void add_linear_constraint(const uniqueness_node_type* node)
  {
    // As for the pairwise terms we assume that all unaries of the given
    // uniqueness factor are present in the ILP.
    auto& uniqueness = uniqueness_.at(node);
    node->traverse_unaries([&](const auto& link, index slot) {
      auto& unary = unaries_.at(link.node);
      model_.addConstr(uniqueness.variable(slot) == unary.variable(link.slot));
    });
  }

  GRBModel model_;
  std::map<const unary_node_type*, gurobi_unary_type> unaries_;
  std::map<const uniqueness_node_type*, gurobi_uniqueness_type> uniqueness_;
  std::map<const pairwise_node_type*, gurobi_pairwise_type> pairwise_;
};


template<typename ALLOCATOR>
class gurobi_fusion_model_builder {
public:
  using allocator_type = ALLOCATOR;
  using graph_type = graph<allocator_type>;
  using unary_node_type = typename graph_type::unary_node_type;
  using uniqueness_node_type = typename graph_type::uniqueness_node_type;
  using pairwise_node_type = typename graph_type::pairwise_node_type;

  struct gurobi_unary_type {
    index label0, label1;
    cost cost0, cost1;
    GRBVar var0, var1;

    gurobi_unary_type(index label0, index label1)
    : label0(label0)
    , label1(label1)
    {
    }

    bool has_two() const {
      return label0 != label1;
    }

    void set_costs(cost cost0, cost cost1) {
      this->cost0 = cost0;
      this->cost1 = cost1;
    }
  };

  gurobi_fusion_model_builder(GRBEnv& env, bool verbose=false)
  : model_(env)
  {
    gurobi_helper::setup_gurobi_parameters(model_, verbose);
  }

  void set_constant(const cost constant)
  {
    model_.set(GRB_DoubleAttr_ObjCon, constant);
  }

  void add_factor(const unary_node_type* node, index label0, index label1)
  {
    assert(node != nullptr);
    assert(label0 >= 0 && label0 < node->factor.size());
    assert(label1 >= 0 && label1 < node->factor.size());

    gurobi_unary_type unary(label0, label1);
    unary.set_costs(node->factor.get(label0), node->factor.get(label1));

    if (unary.has_two()) {
      unary.var0 = add_var(unary.cost0);
      unary.var1 = add_var(unary.cost1);
      model_.addConstr(unary.var0 + unary.var1 == 1);
    } else {
      unary.var0 = add_var(unary.cost0);
      model_.addConstr(unary.var0 == 1);
    }

    const auto result = unaries_.emplace(std::make_pair(node, unary));
    assert(result.second);
  }

  void add_factor(const uniqueness_node_type* node)
  {
    bool did_insert = uniqueness_.insert(node).second;
    assert(did_insert);
    add_linear_constraint(node);
  }

  void add_factor(const pairwise_node_type* node)
  {
    auto did_insert = pairwise_.insert(node).second;
    assert(did_insert);

    const auto& left = unaries_.at(node->unary0);
    const auto& right = unaries_.at(node->unary1);

    const auto cost00 = node->factor.get(left.label0, right.label0);
    const auto cost01 = node->factor.get(left.label0, right.label1);
    const auto cost10 = node->factor.get(left.label1, right.label0);
    const auto cost11 = node->factor.get(left.label1, right.label1);

    auto add_edge_constr = [this](GRBVar e, GRBVar l, GRBVar r) {
      model_.addConstr(e <= l);
      model_.addConstr(e <= r);
      model_.addConstr(e >= l + r - 1);
    };

    // edge 0 - 0
    if (cost00 != 0) {
      GRBVar var00 = add_var(cost00);
      add_edge_constr(var00, left.var0, right.var0);
    }

    // edge 0 - 1
    if (right.has_two() && cost01 != 0) {
      GRBVar var01 = add_var(cost01);
      add_edge_constr(var01, left.var0, right.var1);
    }

    // edge 1 - 0
    if (left.has_two() && cost10 != 0) {
      GRBVar var10 = add_var(cost10);
      add_edge_constr(var10, left.var1, right.var0);
    }

    // edge 1 - 1
    if (left.has_two() && right.has_two() && cost11 != 0) {
      GRBVar var11 = add_var(cost11);
      add_edge_constr(var11, left.var1, right.var1);
    }
  }

  void finalize()
  {
  }

  void optimize()
  {
    model_.optimize();

    const auto status = model_.get(GRB_IntAttr_Status);

    assert(status == GRB_OPTIMAL);

    if (status != GRB_OPTIMAL) {
      std::ostringstream s;
      s << "Gurobi terminated with status " << status;
      throw std::runtime_error(s.str());
    }
  }

  void update_primals()
  {
    for (const auto& [node, info] : unaries_) {
      node->factor.reset_primal();
      if (info.has_two()) {
        const bool val0 = info.var0.get(GRB_DoubleAttr_X) > 0.1;
        const bool val1 = info.var1.get(GRB_DoubleAttr_X) > 0.1;
        assert((val0 && !val1) || (!val0 && val1));
        node->factor.primal() = val0 ? info.label0: info.label1;
      } else {
        node->factor.primal() = info.label0;
        assert(info.var0.get(GRB_DoubleAttr_X) > .1);
      }
    }

    for (const auto* node : uniqueness_) {
      node->factor.primal() = node->unaries.size();
      node->traverse_unaries([&](const auto& edge, const index slot) {
        if (edge.node->factor.primal() == edge.slot)
          node->factor.primal() = slot;
      });
    }

    for (const auto* node : pairwise_) {
      node->factor.reset_primal();
      const auto* left = node->unary0;
      const auto* right = node->unary1;
      node->factor.primal() = std::tuple(left->factor.primal(), right->factor.primal());
    }
  }

protected:
  auto add_var(cost c)
  {
    return model_.addVar(0.0, 1.0, c, GRB_BINARY);
  }

  void add_linear_constraint(const uniqueness_node_type* node)
  {
    gurobi_unary_type* first = nullptr;
    index first_label;
    gurobi_unary_type* second = nullptr;
    index second_label;

    node->traverse_unaries([&](const auto edge, const index slot) {
      auto it = unaries_.find(edge.node);
      if (it != unaries_.end()) {
        if (it->second.label0 == edge.slot || it->second.label1 == edge.slot) {
          if (first == nullptr) {
            first = &it->second;
            first_label = edge.slot;
          } else {
            assert(second == nullptr);
            second = &it->second;
            second_label = edge.slot;
          }
        }
      }
    });

    if (first != nullptr && second != nullptr) {
      bool handled = false;

      if (first->label0 == first_label) {
        if (second->label0 == second_label) {
          model_.addConstr(first->var0 + second->var0 <= 1);
          handled = true;
        }

        if (second->has_two() && second->label1 == second_label) {
          model_.addConstr(first->var0 + second->var1 <= 1);
          handled = true;
        }
      }

      if (first->has_two() && first->label1 == first_label) {
        if (second->label0 == second_label) {
          model_.addConstr(first->var1 + second->var0 <= 1);
          handled = true;
        }

        if (second->has_two() && second->label1 == second_label) {
          model_.addConstr(first->var1 + second->var1 <= 1);
          handled = true;
        }
      }

      assert(handled);
    }
  }

  GRBModel model_;
  std::unordered_map<const unary_node_type*, gurobi_unary_type> unaries_;
  std::unordered_set<const uniqueness_node_type*> uniqueness_;
  std::unordered_set<const pairwise_node_type*> pairwise_;
};

#endif

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
