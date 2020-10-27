#ifndef LIBMPOPT_COMMON_FACTORS_PAIRWISE_HPP
#define LIBMPOPT_COMMON_FACTORS_PAIRWISE_HPP

namespace mpopt {

template<typename ALLOCATOR = std::allocator<cost>>
class pairwise_factor {
public:
  using allocator_type = ALLOCATOR;
  static constexpr cost initial_cost = std::numeric_limits<cost>::signaling_NaN();
  static constexpr index primal_unset = std::numeric_limits<index>::max();

  pairwise_factor(index number_of_labels0, index number_of_labels1, const ALLOCATOR& allocator = ALLOCATOR())
  : costs_(number_of_labels0 * number_of_labels1, initial_cost, allocator)
  , no_labels0_(number_of_labels0)
  , no_labels1_(number_of_labels1)
  , primal0_(primal_unset)
  , primal1_(primal_unset)
#ifndef NDEBUG
  , index0_(-1)
  , index1_(-1)
#endif
  {
  }

  pairwise_factor(const pairwise_factor& other) = delete;
  pairwise_factor& operator=(const pairwise_factor& other) = delete;

#ifndef NDEBUG
  void set_debug_info(index idx0, index idx1)
  {
    index0_ = idx0;
    index1_ = idx1;
  }

  std::string dbg_info() const
  {
    std::ostringstream s;
    s << "uv(" << index0_ << "," << index1_ << ")";
    return s.str();
  }
#endif

  auto size() const { return std::tuple(no_labels0_, no_labels1_); }

  bool is_prepared() const
  {
    bool result = true;
    for (const auto& x : costs_)
        result = result && !std::isnan(x);
    return result;
  }

  void set(const index idx0, const index idx1, cost c)
  {
    two_dimension_array_accessor a(no_labels0_, no_labels1_);
    const index linear_idx = a.to_linear(idx0, idx1);
    costs_[linear_idx] = c;
  }

  cost get(const index idx0, const index idx1)
  {
    two_dimension_array_accessor a(no_labels0_, no_labels1_);
    const index linear_idx = a.to_linear(idx0, idx1);
    return costs_[linear_idx];
  }

  cost lower_bound() const
  {
    return *std::min_element(costs_.begin(), costs_.end());
  }

  cost min_marginal0(const index idx0) const
  {
    two_dimension_array_accessor a(no_labels0_, no_labels1_);
    cost minimum = infinity;
    for (index idx1 = 0; idx1 < no_labels1_; ++idx1) {
      minimum = std::min(minimum, costs_[a.to_linear(idx0, idx1)]);
    }
    return minimum;
  }

  cost min_marginal1(const index idx1) const
  {
    two_dimension_array_accessor a(no_labels0_, no_labels1_);
    cost minimum = infinity;
    for (index idx0 = 0; idx0 < no_labels0_; ++idx0) {
      minimum = std::min(minimum, costs_[a.to_linear(idx0, idx1)]);
    }
    return minimum;
  }

  template<bool right>
  cost min_marginal(const index idx) const
  {
    if constexpr (right)
      return min_marginal1(idx);
    else
      return min_marginal0(idx);
  }

  void repam0(const index idx0, const cost msg)
  {
    two_dimension_array_accessor a(no_labels0_, no_labels1_);
    for (index idx1 = 0; idx1 < no_labels1_; ++idx1) {
      assert_index(idx0, idx1);
      auto linear_idx = a.to_linear(idx0, idx1);
      costs_[linear_idx] += msg;
    }
  }

  void repam1(const index idx1, const cost msg)
  {
    two_dimension_array_accessor a(no_labels0_, no_labels1_);
    for (index idx0 = 0; idx0 < no_labels0_; ++idx0) {
      assert_index(idx0, idx1);
      auto linear_idx = a.to_linear(idx0, idx1);
      costs_[linear_idx] += msg;
    }
  }

  template<bool right>
  void repam(const index idx, const cost msg)
  {
    if constexpr (right)
      repam1(idx, msg);
    else
      repam0(idx, msg);
  }

  void reset_primal() { primal0_ = primal_unset; primal1_ = primal_unset; }

  cost evaluate_primal() const
  {
    two_dimension_array_accessor a(no_labels0_, no_labels1_);
    if (primal0_ != primal_unset && primal1_ != primal_unset)
      return costs_[a.to_linear(primal0_, primal1_)];
    else
      return infinity;
  }

  void round_independently()
  {
    two_dimension_array_accessor a(no_labels0_, no_labels1_);
    auto it = std::min_element(costs_.cbegin(), costs_.cend());
    auto linear_idx = it - costs_.cbegin();

    const auto indices = a.to_nonlinear(linear_idx);
    primal0_ = std::get<0>(indices);
    primal1_ = std::get<1>(indices);

#ifndef NDEBUG
    size_t counter = 0;
    for (auto x : costs_)
      if (dbg::are_identical(x, costs_[linear_idx]))
        ++counter;

    if (counter > 1)
      std::cout << "highly similar values: " << dbg_info() << std::endl;
#endif
  }

  auto primal() { return std::tie(primal0_, primal1_); }
  const auto primal() const { return std::tuple(primal0_, primal1_); }
  bool is_primal_set() const { return primal0_ != primal_unset && primal1_ != primal_unset; }

protected:
  void assert_index(const index idx0, const index idx1) const
  {
    assert(idx0 >= 0 && idx0 < no_labels0_);
    assert(idx1 >= 0 && idx1 < no_labels1_);
    assert(idx0 * idx1 < costs_.size());
  }

  fixed_vector_alloc_gen<cost, ALLOCATOR> costs_;
  index no_labels0_, no_labels1_;
  index primal0_, primal1_;

#ifndef NDEBUG
  index index0_, index1_;
#endif

  template<typename> friend class gurobi_pairwise_factor;
};


#ifdef ENABLE_GUROBI

template<typename ALLOCATOR>
class gurobi_pairwise_factor {
public:
  using allocator_type = ALLOCATOR;
  using factor_type = pairwise_factor<allocator_type>;

  gurobi_pairwise_factor(factor_type& factor, GRBModel& model)
  : factor_(&factor)
  , vars_(factor.no_labels0_ * factor.no_labels1_)
  {
    std::optional<size_t> linear_primal_idx;
    if (factor_->is_primal_set()) {
      two_dimension_array_accessor a(factor_->no_labels0_, factor_->no_labels1_);
      const auto [p0, p1] = factor_->primal();
      linear_primal_idx = a.to_linear(p0, p1);
    }

    std::vector<double> coeffs(vars_.size(), 1.0);
    for (size_t i = 0; i < vars_.size(); ++i) {
      vars_[i] = model.addVar(0.0, 1.0, factor_->costs_[i], GRB_CONTINUOUS);
      if (linear_primal_idx.has_value())
        vars_[i].set(GRB_DoubleAttr_Start, *linear_primal_idx == i ? 1.0 : 0.0);
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
    auto [size0, size1] = factor_->size();
    two_dimension_array_accessor a(size0, size1);

    auto [p0, p1] = factor_->primal();
    p0 = p1 = factor_type::primal_unset;
    double max = -std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < vars_.size(); ++i) {
      const auto val = f(vars_[i]);
      if (val > max) {
        max = val;
        const auto indices = a.to_nonlinear(i);
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

#endif

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
