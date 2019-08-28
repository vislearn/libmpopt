#ifndef foo
#define foo

namespace mpopt {

template<typename DERIVED_TYPE>
class solver {
public:
  solver()
  : iterations_(0)
  , constant_(0)
  { }

  cost lower_bound() const
  {
    static_cast<const DERIVED_TYPE*>(this)->graph_.check_structure();
    cost result = constant_;

    static_cast<const DERIVED_TYPE*>(this)->for_each_node([&result](const auto* node) {
      result += node->factor.lower_bound();
    });

    return result;
  }

  cost evaluate_primal() const
  {
    const auto* derived = static_cast<const DERIVED_TYPE*>(this);

    derived->graph_.check_structure();
    const cost inf = std::numeric_limits<cost>::infinity();
    cost result = constant_;

    derived->for_each_node(
      [&](const auto* node) {
        if (!derived->check_primal_consistency(node))
          result += inf;
        result += node->factor.evaluate_primal();
      });

    return result;
  }

  cost upper_bound() const { return evaluate_primal(); }

  void reset_primal()
  {
    static_cast<DERIVED_TYPE*>(this)->for_each_node([](const auto* node) {
      node->factor.reset_primal();
    });
  }

  void run(const int max_iterations)
  {
    assert(false && "Not implemented!");
  }

  void solve_ilp()
  {
    // We do not reset the primal as they will be used as a MIP start.
    typename DERIVED_TYPE::gurobi_model_builder_type builder(gurobi_env_);
    builder.set_constant(constant_);

    static_cast<DERIVED_TYPE*>(this)->for_each_node([&builder](const auto* node) {
      builder.add_factor(node);
    });

    builder.finalize();
    builder.optimize();
    builder.update_primals();
  }

protected:
  int iterations_;
  cost constant_;
  GRBEnv gurobi_env_;
};

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
