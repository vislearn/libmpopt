#ifndef LIBMPOPT_COMMON_SOLVER_HPP
#define LIBMPOPT_COMMON_SOLVER_HPP

namespace mpopt {

constexpr int default_batch_size = 10;
constexpr int default_max_batches = 100;

template<typename DERIVED_TYPE>
class solver {
public:
  using clock_type = std::chrono::steady_clock;

  solver()
  : iterations_(0)
  , duration_(0)
  { }

  auto lower_bound() const { return derived_this()->graph_.lower_bound(); }
  auto evaluate_primal() const { return derived_this()->graph_.evaluate_primal(); }
  auto check_primal_consistency() const { return derived_this()->graph_.check_primal_consistency(); }
  auto upper_bound() const { return derived_this()->graph_.upper_bound(); }
  auto reset_primal() const { return derived_this()->graph_.reset_primal(); }

  void run(const int batch_size=default_batch_size, const int max_batches=default_max_batches)
  {
    assert(false && "Not implemented!");
  }

  double runtime() const
  {
    using seconds = std::chrono::duration<double>;
    return std::chrono::duration_cast<seconds>(duration_).count();
  }

  void solve_ilp()
  {
#ifdef ENABLE_GUROBI
    // We do not reset the primal as they will be used as a MIP start.
    typename DERIVED_TYPE::gurobi_model_builder_type builder(gurobi_env());
    builder.set_constant(derived_this()->graph_.constant());

    derived_this()->graph_.for_each_node([&builder](const auto* node) {
      builder.add_factor(node);
    });

    builder.finalize();
    builder.optimize();
    builder.update_primals();
#else
    std::cerr << "Abort: ENABLE_GUROBI was unset during configuration of libmpopt.";
    std::abort();
#endif
  }

protected:
  int iterations_;
  clock_type::duration duration_;

#ifdef ENABLE_GUROBI
  std::optional<GRBEnv> gurobi_env_;

  GRBEnv& gurobi_env()
  {
    if (gurobi_env_)
      return *gurobi_env_;
    else
      return gurobi_env_.emplace();
  }
#endif

private:
  DERIVED_TYPE* derived_this() { return static_cast<DERIVED_TYPE*>(this); }
  const DERIVED_TYPE* derived_this() const { return static_cast<const DERIVED_TYPE*>(this); }
};

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
