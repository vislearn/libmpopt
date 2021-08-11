#ifndef LIBQAPOPT_COMMON_FACTORS_UNARY_HPP
#define LIBQAPOPT_COMMON_FACTORS_UNARY_HPP

namespace qapopt {

template<typename ALLOCATOR = std::allocator<cost>>
class unary_factor {
public:
  using allocator_type = ALLOCATOR;
  static constexpr cost initial_cost = std::numeric_limits<cost>::signaling_NaN();
  static constexpr index primal_unset = std::numeric_limits<index>::max();

  unary_factor(index number_of_labels, const ALLOCATOR& allocator = ALLOCATOR())
  : costs_(number_of_labels, initial_cost, allocator)
  , primal_(primal_unset)
#ifndef NDEBUG
  , index_(-1)
#endif
  {
  }

  unary_factor(const unary_factor& other) = delete;
  unary_factor& operator=(const unary_factor& other) = delete;

#ifndef NDEBUG
  void set_debug_info(index idx) { index_ = idx; }

  std::string dbg_info() const
  {
    std::ostringstream s;
    s << "v(" << index_ << ")";
    return s.str();
  }
#endif

  auto size() const { return costs_.size(); }

  bool is_prepared() const
  {
    bool result = true;
    for (const auto& x : costs_)
        result = result && !std::isnan(x);
    return result;
  }

  void set(const index idx, cost c) { assert_index(idx); costs_[idx] = c; }
  cost get(const index idx) { assert_index(idx); return costs_[idx]; }

  cost normalize()
  {
    const auto lb = lower_bound();
    for (auto& x : costs_) x -= lb;
    return lb;
  }

  cost lower_bound() const
  {
    return *std::min_element(costs_.begin(), costs_.end());
  }

  void repam(const index idx, const cost msg)
  {
    assert_index(idx);
    costs_[idx] += msg;
  }

  void reset_primal() { primal_ = primal_unset; }

  cost evaluate_primal() const
  {
    if (primal_ != primal_unset)
      return costs_[primal_];
    else
      return infinity;
  }

  void round_independently()
  {
    auto it = std::min_element(costs_.cbegin(), costs_.cend());
    primal_ = it - costs_.cbegin();
    assert(primal_ >= 0 && primal_ < costs_.size());

#ifndef NDEBUG
    size_t counter = 0;
    for (auto x : costs_)
      if (dbg::are_identical(x, costs_[primal_]))
        ++counter;

    if (counter > 1)
      std::cout << "highly similar values: " << dbg_info() << std::endl;
#endif
  }

  auto& primal() { return primal_; }
  const auto& primal() const { return primal_; }
  bool is_primal_set() const { return primal_ != primal_unset; }

protected:
  void assert_index(const index idx) const { assert(idx >= 0 && idx < costs_.size()); }

  fixed_vector_alloc_gen<cost, ALLOCATOR> costs_;
  index primal_;

#ifndef NDEBUG
  index index_;
#endif
};


#ifdef ENABLE_GUROBI

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
    for (size_t i = 0; i < factor_->size(); ++i) {
      vars_[i] = model.addVar(0.0, 1.0, factor_->get(i), GRB_BINARY);
      if (factor_->is_primal_set())
        vars_[i].set(GRB_DoubleAttr_Start, factor_->primal() == i ? 1.0 : 0.0);
    }

    GRBLinExpr expr;
    expr.addTerms(coeffs.data(), vars_.data(), vars_.size());
    model.addConstr(expr == 1);
  }

  void update_primal() const
  {
    update_primal([](GRBVar v) { return v.get(GRB_DoubleAttr_X); });
  }

  template<typename FUNCTOR>
  void update_primal(FUNCTOR f) const
  {
    auto& p = factor_->primal();
    p = factor_type::primal_unset;
    double max = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < vars_.size(); ++i) {
      const auto val = f(vars_[i]);
      if (val > max) {
        max = val;
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

#endif

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
