#ifndef LIBMPOPT_COMMON_GRAPH_HPP
#define LIBMPOPT_COMMON_GRAPH_HPP

namespace mpopt {

template<typename DERIVED_TYPE>
class graph {
public:

  graph()
  : constant_(0)
  { }

  void check_structure() const
  {
    derived_this()->for_each_node([](const auto* node) {
      node->check_structure();
    });
  }

  bool check_primal_consistency() const
  {
    bool result = true;
    derived_this()->for_each_node([this, &result](const auto* node) {
      result = result && derived_this()->check_primal_consistency(node);
    });
    return result;
  }

  auto constant() const { return constant_; }
  void add_to_constant(cost v) { constant_ += v; }

  cost lower_bound() const
  {
    check_structure();
    cost result = constant_;

    derived_this()->for_each_node([&result](const auto* node) {
      result += node->factor.lower_bound();
    });

    return result;
  }

  cost evaluate_primal() const
  {
    check_structure();
    cost result = constant_;

    derived_this()->for_each_node([&](const auto* node) {
      if (!derived_this()->check_primal_consistency(node))
        result += infinity;
      result += node->factor.evaluate_primal();
    });

    return result;
  }

  cost upper_bound() const { return evaluate_primal(); }

  void reset_primal() const
  {
    derived_this()->for_each_node([](const auto* node) {
      node->factor.reset_primal();
    });
  }

protected:
  cost constant_;

private:
  DERIVED_TYPE* derived_this() { return static_cast<DERIVED_TYPE*>(this); }
  const DERIVED_TYPE* derived_this() const { return static_cast<const DERIVED_TYPE*>(this); }
};

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
