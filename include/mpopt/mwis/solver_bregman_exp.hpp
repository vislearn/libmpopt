#ifndef LIBMPOPT_MWIS_SOLVER_BREGMAN_EXP_HPP
#define LIBMPOPT_MWIS_SOLVER_BREGMAN_EXP_HPP

namespace mpopt {
namespace mwis {
namespace bregman_exp {

// Can be switch to float for possibly faster dual updates. Loss of precision
// will not affect the solution quality, because the exp-costs are used to
// compute a equivalance transformation (reparametrization) at full precsision.
using cost_exp = cost;

constexpr bool enable_stabilization = false;
constexpr cost_exp get_default_threshold_stability() {
  if constexpr (std::is_same_v<cost_exp, float>) {
    return 1e30f;
  } else {
    return 1e300;
  }
}

struct range {
  index begin;
  index end;
  index size;
};

constexpr int default_greedy_generations = 10;
constexpr bool initial_reparametrization = true;

template<typename T> bool feasibility_check(const T   sum) { return std::abs(sum - 1.0) < 1e-8; }
template<>           bool feasibility_check(const int sum) { return sum == 1; }

class solver {
public:

  solver()
  : finalized_graph_(false)
  , finalized_costs_(false)
  , constant_(0.0)
  , scaling_(1.0)
  , gen_(std::random_device()())
#ifdef ENABLE_QPBO
  , qpbo_(0, 0)
#endif
  , threshold_feasibility_(1e-2)
  , threshold_stability_(get_default_threshold_stability())
  , temperature_drop_factor_(0.5)
  {
#ifndef ENABLE_QPBO
    std::cerr << "!!!!!!!!!!\n"
              << "ENABLE_QPBO was not activated during configuration of libmpopt.\n"
              << "No fusion moves are performed and the the quality of the computed assignment is degraded.\n"
              << "!!!!!!!!!!\n" << std::endl;
#endif
  }

  index add_node(cost cost)
  {
    assert(!finalized_graph_);
    assert(no_cliques() == 0);
    costs_.push_back(cost / scaling_);
    orig_.push_back(cost);
    return costs_.size() - 1;
  }

  index add_clique(const std::vector<index>& indices)
  {
    assert(!finalized_graph_);

    range cl;
    cl.begin = clique_index_data_.size();
    cl.size = indices.size() + 1;
    cl.end = cl.begin + cl.size;
    clique_indices_.push_back(cl);

    for (auto index : indices)
      clique_index_data_.push_back(index);
    clique_index_data_.push_back(costs_.size());
    costs_.push_back(0.0);

    assert(cl.end == clique_index_data_.size());
    return clique_indices_.size() - 1;
  }

  void finalize() {
    temperature_ = 1;
    finalize_graph();
    finalize_costs();
  }

  bool finalized() const { return finalized_graph_ && finalized_costs_; }

  cost constant() const { return constant_ * scaling_; }
  void constant(cost c) { constant_ = c / scaling_; }

  template<bool reduced=false>
  cost node_cost(index i) const {
    assert(finalized_graph_);
    assert(i < no_nodes() && i < no_orig());
    return reduced ? costs_[i] * scaling_: orig_[i];
  }

  void node_cost(index i, cost c)
  {
    assert(finalized_graph_);
    assert(i < no_nodes() && i < no_orig());
    const auto shift = c - orig_[i];
    orig_[i] += shift;
    costs_[i] += shift / scaling_;
    finalized_costs_ = false;
  }

  template<bool reduced=false>
  cost clique_cost(index i) const
  {
    assert(finalized());
    assert(i < no_cliques());
    const auto j = no_orig() + i;
    assert(j < no_nodes());
    return reduced ? costs_[j] * scaling_ : 0.0;
  }

  cost dual_relaxed() const
  {
    // If alphas are non-zero this computation fails. Reparametrization would be required before calling this
    // function then.
    assert_unset_alphas();

    // Compute $D(\lambda) = \sum_i \lambda_i + \sum_i \max_{x \in [0, 1]^N} <c^\lambda, x>$.
    // Note that the sum of all lambdas = constant_.
    if constexpr (initial_reparametrization) {
      // We now that the second sum is always zero, because this is ensured after the first
      // reparametrization run.
      return constant_ * scaling_;
    } else {
      // Compute $\sum_i \max_{x \in [0, 1]^N} <c^\lambda, x>$.
      const auto f = [](const auto a, const auto b) {
        return a + std::max(b, cost{0});
      };
      const auto sum_max_i = std::accumulate(costs_.cbegin(), costs_.cend(), cost{0}, f);
      return (constant_ + sum_max_i) * scaling_;
    }
  }

  cost primal() const { return value_best_ * scaling_; }

  cost primal_relaxed() const { return value_relaxed_best_ * scaling_; }

  template<typename OUTPUT_ITERATOR>
  void assignment(OUTPUT_ITERATOR begin, OUTPUT_ITERATOR end) const
  {
    assert(finalized_graph_);
    assert(end - begin == orig_.end() - orig_.begin());
    assert(assignment_best_.size() >= orig_.size());
    std::copy(assignment_best_.begin(), assignment_best_.begin() + orig_.size(), begin);
  }

  bool assignment(index node_idx) const
  {
    assert(finalized_graph_);
    assert(node_idx >= 0 && node_idx < orig_.size());
    return assignment_best_[node_idx];
  }

  int iterations() const { return iterations_; }

  void run(const int batch_size=default_batch_size,
           const int max_batches=default_max_batches,
           const int greedy_generations=default_greedy_generations)
  {
    std::cout.precision(std::numeric_limits<cost>::max_digits10);
    assert(finalized());
    signal_handler h;
    dbg::timer t_total;

    while (true) {
      const auto d = dual_relaxed();
      const auto p = primal();
      const auto p_relaxed = primal_relaxed();
      const auto gap = (d - p) / d * 100.0;
      const auto gap_relaxed = (d - p_relaxed) / d * 100.0;
      std::cout << "it=" << iterations_ << " "
                << "d=" << d << " "
                << "p=" << p << " "
                << "gap=" << gap << "% "
                << "p_relaxed=" << p_relaxed << " "
                << "gap_relaxed=" << gap_relaxed << "% "
                << "t=" << t_total.seconds<true>() << "s "
                << "T=" << temperature_ << " "
                << "total=" << t_total.milliseconds<true>() / iterations_ << "ms/it" << std::endl;

      init_exponential_domain();

      if (h.signaled()) {
        break;
      }

      if (temperature_ < 1e-16) {
        std::cout << "Temperature limit reached." << std::endl;
        break;
      }

      if (gap < 1e-2) {
        std::cout << "Gap limit reached." << std::endl;
        break;
      }

      t_total.start();
      bool is_optimal = false;
      while (!is_optimal && !h.signaled()) {
        for (int j = 0; j < batch_size; ++j) {
          size_t count_stabilization = 0;
          foreach_clique([&](const auto clique_idx) {
            update_lambda(clique_idx);

            if constexpr (enable_stabilization) {
              const auto& alpha = alphas_[clique_idx];
              if (alpha + 1/alpha > threshold_stability_) {
                reparametrize();
                temperature_ /= 0.5 * temperature_drop_factor_;
                init_exponential_domain();
                count_stabilization += 1;
              }
            }
          });

          std::cout << count_stabilization << " " << std::flush;
        }

        is_optimal = true;
        foreach_clique([&](const auto clique_idx) {
          cost_exp sum = 0.0;
          foreach_node_in_clique(clique_idx, [&](const auto node_idx) {
            sum += assignment_relaxed_[node_idx];
          });
          is_optimal &= std::abs(sum - 1) <= threshold_feasibility_;
        });

        iterations_ += batch_size;
      }

      reparametrize();

      // We do not use the relaxed primal, we only output it. This allows to
      // read off the progress of the dual optimization directly.
      compute_relaxed_truncated_projection();

      temperature_ *= temperature_drop_factor_;

      update_integer_assignment(greedy_generations);
      t_total.stop();

      std::cout << std::endl;
    }

    std::cout << "Optimization stopped: "
              << "d=" << dual_relaxed() << " "
              << "p=" << primal() << std::endl;
  }

  index no_nodes() const { return costs_.size(); }
  index no_orig() const { return orig_.size(); }
  index no_cliques() const { return clique_indices_.size(); }

  double threshold_feasibility() const { return threshold_feasibility_; }
  void threshold_feasibility(const double v) { threshold_feasibility_ = v; }

  double threshold_stability() const { return threshold_stability_; }
  void threshold_stability(const double v) { threshold_stability_ = v; }

  double temperature_drop_factor() const { return temperature_drop_factor_; }
  void temperature_drop_factor(const double v) { temperature_drop_factor_ = v; }

  double temperature() const { return temperature_; }
  void temperature(const double v) { temperature_ = v; }

protected:

  template<typename F>
  void foreach_clique(F f) const
  {
    for (index clique_idx = 0; clique_idx < no_cliques(); ++clique_idx)
      f(clique_idx);
  }

  template<bool only_orig=false, typename F>
  void foreach_node(F f) const
  {
    for (index node_idx = 0; node_idx < (only_orig ? no_orig() : no_nodes()); ++node_idx)
      f(node_idx);
  }

  template<typename F>
  void foreach_node_in_clique(const index clique_idx, F f) const
  {
    const auto& r = clique_indices_[clique_idx];
    for (index idx = r.begin; idx < r.end; ++idx) {
      const index node_idx = clique_index_data_[idx];
      f(node_idx);
    }
  }

  template<typename F>
  void foreach_clique_of_node(const index node_idx, F f) const
  {
    const auto& r = node_cliques_[node_idx];
    for (index idx = r.begin; idx < r.end; ++idx) {
      const index clique_idx = node_cliques_data_[idx];
      f(clique_idx);
    }
  }

  template<typename F>
  void foreach_node_neigh(const index node_idx, F f) const
  {
    const auto& r = node_neighs_[node_idx];
    for (index idx = r.begin; idx < r.end; ++idx) {
      const index other_node_idx = node_neigh_data_[idx];
      f(other_node_idx);
    }
  }

  template<typename T>
  bool compute_feasibility(const std::vector<T>& assignment) const
  {
    for (const auto& cl : clique_indices_) {
      T sum = 0;
      for (index idx = cl.begin; idx < cl.end; ++idx)
        sum += assignment[clique_index_data_[idx]];
      if (!feasibility_check(sum))
        return false;
    }
    return true;
  }

  cost compute_primal(const std::vector<int>& assignment) const
  {
    // Same as relaxed objective, we just check that $x_i \in {0, 1}$.
#ifndef NDEBUG
    for (auto a : assignment)
      assert(a == 0 || a == 1);
#endif

    return compute_primal_relaxed(assignment);
  }

  template<typename T>
  cost compute_primal_relaxed(const std::vector<T>& assignment) const
  {
    // Compute $<c, x>$ s.t. uniqueness constraints.
    assert(assignment.size() == no_nodes());
    cost result = constant_;

    for (index node_idx = 0; node_idx < no_nodes(); ++node_idx) {
      const auto x = assignment[node_idx];
      assert(x >= 0 && x <= 1);
      result += costs_[node_idx] * x;
    }

#ifndef NDEBUG
    if (!compute_feasibility(assignment))
      result = infinity;
#endif

    return result;
  }

  void finalize_graph()
  {
    if (finalized_graph_)
      return;

    // Look for nodes that are not part of any clique. If a node is not part of
    // any clique, there is no lambda that can affect the reduced cost of the
    // node. However, the code assumes that we can shift the reduced cost of
    // any node below zero. To simplify the code we add a fake clique.
    {
      std::vector<bool> tmp(no_nodes(), false);
      for (index clique_idx = 0; clique_idx < no_cliques(); ++clique_idx) {
        const auto& cl = clique_indices_[clique_idx];
        for (index idx = cl.begin; idx < cl.end; ++idx) {
          const auto node_idx = clique_index_data_[idx];
          tmp[node_idx] = true;
        }
      }

      std::vector<index> clique;
      for (index node_idx = 0; node_idx < no_orig(); ++node_idx) {
        if (!tmp[node_idx]) {
          clique.resize(1);
          clique[0] = node_idx;
          add_clique(clique);
        }
      }
    }

    //
    // Construct node to clique mapping.
    //

    std::vector<std::vector<index>> tmp(no_nodes());
    for (index clique_idx = 0; clique_idx < no_cliques(); ++clique_idx) {
      const auto& cl = clique_indices_[clique_idx];
      for (index idx = cl.begin; idx < cl.end; ++idx) {
        const auto node_idx = clique_index_data_[idx];
        tmp[node_idx].push_back(clique_idx);
      }
    }

    node_cliques_.resize(no_nodes());
    for (index nidx = 0; nidx < no_nodes(); ++nidx) {
      auto& current = tmp[nidx];
      auto& nc = node_cliques_[nidx];
      nc.begin = node_cliques_data_.size();
      node_cliques_data_.insert(node_cliques_data_.end(), current.begin(), current.end());
      nc.end = node_cliques_data_.size();
      nc.size = nc.end - nc.begin;
    }

    //
    // Construct node neighborhood.
    //

    for (auto& vec : tmp)
      vec.clear();

    for (const auto& cl : clique_indices_) {
      for (index idx0 = cl.begin; idx0 < cl.end; ++idx0)
        for (index idx1 = cl.begin; idx1 < cl.end; ++idx1)
          if (idx0 != idx1)
            tmp[clique_index_data_[idx0]].push_back(clique_index_data_[idx1]);
    }

    node_neighs_.resize(no_nodes());
    for (index nidx = 0; nidx < no_nodes(); ++nidx) {
      auto& current = tmp[nidx];
      std::sort(current.begin(), current.end());
      auto end = std::unique(current.begin(), current.end());

      auto& neighbors = node_neighs_[nidx];
      neighbors.begin = node_neigh_data_.size();
      node_neigh_data_.insert(node_neigh_data_.end(), current.begin(), end);
      neighbors.end = node_neigh_data_.size();
      neighbors.size = neighbors.end - neighbors.begin;
    }

    //
    // Construct trivial labeling of zero cost.
    //

    // Set assignment_latest_ to zero labeling.
    assignment_latest_.resize(no_nodes());
    for (index nidx = 0; nidx < no_nodes(); ++nidx)
      assignment_latest_[nidx] = nidx < no_orig() ? 0 : 1;
    value_latest_ = compute_primal(assignment_latest_);
    assert(std::abs(value_latest_) < 1e-8);

    // Set assignment_best_ to zero labeling.
    assignment_best_ = assignment_latest_;
    value_best_ = value_latest_;

    // Set assignment_relaxed_ to zero labeling.
    assignment_relaxed_.assign(assignment_latest_.cbegin(), assignment_latest_.cend());
    value_relaxed_ = value_latest_;

    // Set assignment_relaxed_best_ to zero labeling.
    assignment_relaxed_best_ = assignment_relaxed_;
    value_relaxed_best_ = value_relaxed_;

    //
    // Initialize remaining things.
    //

    alphas_.resize(no_cliques());

    scratch_greedy_indices_.resize(no_cliques());
    std::iota(scratch_greedy_indices_.begin(), scratch_greedy_indices_.end(), 0);

    scratch_qpbo_indices_.resize(no_orig());

    finalized_graph_ = true;
  }

  void finalize_costs()
  {
    if (finalized_costs_)
      return;

    if constexpr (initial_reparametrization) {
      // Update all lambdas (without smoothing, invariants do not hold) to ensure
      // that invariants (negative node costs) hold.
      for (index clique_idx = 0; clique_idx < no_cliques(); ++clique_idx)
        update_lambda<false>(clique_idx);

      std::cout << "initial reparametrization: lb=" << dual_relaxed() << " ub=" << primal() << std::endl;

      auto it = std::min_element(costs_.cbegin(), costs_.cend());
      scaling_ = std::abs(*it);
    } else {
      auto it = std::max_element(costs_.cbegin(), costs_.cend());
      scaling_ = std::abs(*it);
    }

    constant_ /= scaling_;
    for (auto& c : costs_)
      c /= scaling_;

    // We update the cached values for the corresponding assignments (costs
    // have most likely changed the assignment between calls to
    // finalize_costs).
    value_relaxed_ = compute_primal_relaxed(assignment_relaxed_);
    update_relaxed_best(assignment_relaxed_, value_relaxed_);
    value_best_ = compute_primal(assignment_best_);
    update_relaxed_best(assignment_best_, value_best_);
    iterations_ = 0;

    // This will set up the corresponding costs in the exponential domain. We
    // do this so that all invariants hold, e.g., for all functions that call
    // `assert_unset_alphas`.
    init_exponential_domain();

    finalized_costs_ = true;
  }

  void init_exponential_domain() {
    // Reset alphas to 1.
    std::fill(alphas_.begin(), alphas_.end(), 1.0);

    // Recompute exponentiated costs.
    std::transform(costs_.cbegin(), costs_.cend(), assignment_relaxed_.begin(),
      [this](const auto c) { return std::exp(c / temperature_); });

#ifndef NDEBUG
    for (const auto v : assignment_relaxed_)
      assert(std::isfinite(v));
#endif
  }

  template<bool smoothing=true>
  void copy_clique_in(const range& cl)
  {
    scratch_.resize(cl.size, 0.0);
    for (index idx = 0; idx < cl.size; ++idx)
      if constexpr (smoothing)
        scratch_[idx] = assignment_relaxed_[clique_index_data_[cl.begin + idx]];
      else
        scratch_[idx] = costs_[clique_index_data_[cl.begin + idx]];
  }

  template<bool smoothing=true>
  void copy_clique_out(const range& cl)
  {
    for (index idx = 0; idx < cl.size; ++idx)
      if constexpr (smoothing)
        assignment_relaxed_[clique_index_data_[cl.begin + idx]] = scratch_[idx];
      else
        costs_[clique_index_data_[cl.begin + idx]] = scratch_[idx];
  }

  template<bool smoothing=true>
  void update_lambda(const index clique_idx)
  {
    const auto& cl = clique_indices_[clique_idx];
    copy_clique_in<smoothing>(cl);

    if constexpr (smoothing) {
      auto& alpha = alphas_[clique_idx];
      const auto s = std::reduce(scratch_.cbegin(), scratch_.cend());
      alpha /= s;
      for (auto& x : scratch_)
        x /= s;
    } else {
      const auto msg = std::reduce(scratch_.cbegin(), scratch_.cend(), -infinity, [](cost_exp a, cost_exp b) { return std::max(a, b); });
      assert(std::isfinite(msg));

      constant_ += msg;
      for (auto& c : scratch_)
        c -= msg;
    }

    copy_clique_out<smoothing>(cl);
  }

  void reparametrize(const index clique_idx, const cost v)
  {
    assert(clique_idx >= 0 && clique_idx < no_cliques());
    assert(std::isfinite(v));
    assert(std::isfinite(constant_));

    foreach_node_in_clique(clique_idx, [&](const auto node_idx) {
      auto& cost = costs_[node_idx]; assert(std::isfinite(cost));
      cost -= v; assert(std::isfinite(cost));
    });

    constant_ += v;
    assert(std::isfinite(constant_));
  }

  void reparametrize(const index clique_idx)
  {
    assert(clique_idx >= 0 && clique_idx < no_cliques());
    const auto alpha = alphas_[clique_idx];
    assert(std::isfinite(alpha));
    reparametrize(clique_idx, -temperature_ * std::log(alpha));
  }

  void reparametrize()
  {
    foreach_clique([this](const auto clique_idx) {
      reparametrize(clique_idx);
    });
  }

  template<typename T>
  void update_relaxed_best(const std::vector<T>& assignment, const cost& value)
  {
    assert(assignment_relaxed_best_.size() == assignment.size());
    if (value > value_relaxed_best_) {
      assignment_relaxed_best_.assign(assignment.cbegin(), assignment.cend());
      value_relaxed_best_ = value;
    }
  }

  void compute_relaxed_truncated_projection()
  {
    auto node_cost = [this](const index node_idx) -> cost_exp* {
      assert(node_idx < no_orig());
      return &assignment_relaxed_[node_idx];
    };

    auto slack = [this](const index clique_idx) -> cost_exp* {
      assert(no_orig() + clique_idx < assignment_relaxed_.size());
      return &assignment_relaxed_[no_orig() + clique_idx];
    };

    auto max_allowed = [this, slack](const index node_idx) -> cost_exp {
      assert(node_idx < no_orig());
      const auto& nc = node_cliques_[node_idx];
      cost_exp result = 1.0;
      for (index idx = nc.begin; idx < nc.end; ++idx) {
        const auto clique_idx = node_cliques_data_[idx];
        result = std::min(result, *slack(clique_idx));
      }
      assert(!std::isinf(result));
      return result;
    };

    auto reduce_max_allowed = [this, &slack](const index node_idx, const cost value) {
      assert(node_idx < no_orig());
      const auto& nc = node_cliques_[node_idx];
      for (index idx = nc.begin; idx < nc.end; ++idx) {
        const auto clique_idx = node_cliques_data_[idx];
        assert(value <= *slack(clique_idx));
        *slack(clique_idx) -= value;
      }
    };

    for (index node_idx = 0; node_idx < no_orig(); ++node_idx)
      *node_cost(node_idx) = std::exp(costs_[node_idx] / temperature_);

    for (index clique_idx = 0; clique_idx < no_cliques(); ++clique_idx)
      *slack(clique_idx) = 1.0;

    scratch_qpbo_indices_.resize(no_orig());
    std::iota(scratch_qpbo_indices_.begin(), scratch_qpbo_indices_.end(), 0);
    std::sort(scratch_qpbo_indices_.begin(), scratch_qpbo_indices_.end(), [&node_cost](index a, index b) {
      return *node_cost(a) > *node_cost(b);
    });
    for (index node_idx : scratch_qpbo_indices_) {
      cost_exp* x = node_cost(node_idx);
      *x = std::min(*x, max_allowed(node_idx));
      reduce_max_allowed(node_idx, *x);
    }

    value_relaxed_ = compute_primal_relaxed(assignment_relaxed_);
    update_relaxed_best(assignment_relaxed_, value_relaxed_);
  }

  bool update_integer_assignment(int greedy_generations)
  {
    bool has_improved = false;
    for (int i = 0; i < greedy_generations; ++i)
      has_improved |= update_integer_assignment();
    return has_improved;
  }

  bool update_integer_assignment()
  {
    greedy();
#ifdef ENABLE_QPBO
    const auto old_value_best_ = value_best_;
    fusion_move();
    return value_best_ > old_value_best_;
#else
    if (value_latest_ > value_best_) {
      value_best_ = value_latest_;
      assignment_best_ = assignment_latest_;
      return true;
    }
    return false;
#endif
  }

  void greedy()
  {
    std::cout << "g " << std::flush;
    std::shuffle(scratch_greedy_indices_.begin(), scratch_greedy_indices_.end(), gen_);

    std::fill(assignment_latest_.begin(), assignment_latest_.end(), -1);
    for (const auto clique_idx : scratch_greedy_indices_)
      greedy_clique(clique_idx);

    value_latest_ = compute_primal(assignment_latest_);
    update_relaxed_best(assignment_latest_, value_latest_);
  }

  void greedy_clique(const index clique_idx)
  {
    assert(clique_idx >= 0 && clique_idx < no_cliques());

    int count = 0;
    foreach_node_in_clique(clique_idx, [&](const auto node_idx) {
      count += assignment_latest_[node_idx] == 1 ? 1 : 0;
    });
    assert(count == 0 || count == 1);

    if (count > 0)
      return;

    cost max = -infinity;
    index argmax = -1;
    foreach_node_in_clique(clique_idx, [&](const auto node_idx) {
      if (assignment_latest_[node_idx] != 0 && costs_[node_idx] > max) {
        max = costs_[node_idx];
        argmax = node_idx;
      }
    });

    assignment_latest_[argmax] = 1;
    foreach_node_neigh(argmax, [&](const auto node_idx) {
      assert(node_idx != argmax);
      assert(assignment_latest_[node_idx] != 1);
      assignment_latest_[node_idx] = 0;
    });
  }

#ifdef ENABLE_QPBO
  bool fuse_two_assignments(std::vector<int>& a0, const std::vector<int>& a1)
  {
    constexpr cost QPBO_INF = 1e20;
    assert(a0.size() == no_nodes());
    assert(a1.size() == no_nodes());

    auto reset_qpbo_indices = [&]() {
      std::fill(scratch_qpbo_indices_.begin(), scratch_qpbo_indices_.end(), -1);
    };

    auto is_node_present = [&](auto nidx) {
      return nidx < no_orig() && scratch_qpbo_indices_[nidx] != -1;
    };

    auto for_each_present_node_tuple = [&](auto func) {
      for (index nidx1 = 0; nidx1 < no_orig(); ++nidx1) {
        if (is_node_present(nidx1)) {
          for (index idx = node_neighs_[nidx1].begin; idx < node_neighs_[nidx1].end; ++idx) {
            const auto nidx2 = node_neigh_data_[idx];
            if (nidx1 < nidx2 && is_node_present(nidx2)) {
              func(nidx1, nidx2);
            }
          }
        }
      }
    };

    auto enable_all_dumies = [&]() {
      std::fill(a0.begin() + no_orig(), a0.end(), 1);
    };

    auto disable_dummy_of_clique = [&](const auto clique_idx) {
      assert(clique_idx >= 0 && clique_idx < clique_indices_.size());
      assert(clique_idx >= 0 && clique_idx < clique_indices_.size());
      const auto& cl = clique_indices_[clique_idx];
      assert(cl.size >= 2);
      assert(cl.end - 1 >= 0 && cl.end - 1 < clique_index_data_.size());
      const auto nidx = clique_index_data_[cl.end - 1];
      assert(nidx >= no_orig());
      assert(nidx < no_nodes());
      assert(a0[nidx] == 1);
      a0[nidx] = 0;
    };

    auto disable_dummy_for_node = [&](const auto nidx) {
      assert(nidx >= 0 && nidx < no_orig());
      assert(a0[nidx] == 1);
      const auto& nc = node_cliques_[nidx];
      for (auto idx = nc.begin; idx < nc.end; ++idx) {
        assert(idx >= 0 && idx < node_cliques_data_.size());
        const auto clique_idx = node_cliques_data_[idx];
        disable_dummy_of_clique(clique_idx);
      }
    };

    qpbo_.Reset();
    reset_qpbo_indices();

    for (index nidx = 0; nidx < no_orig(); ++nidx) {
      assert(a0[nidx] == 0 || a0[nidx] == 1);
      assert(a1[nidx] == 0 || a1[nidx] == 1);

      if (a0[nidx] != a1[nidx]) {
        const bool l0 = a0[nidx] == 1, l1 = a1[nidx] == 1;
        cost c0 = l0 ? -orig_[nidx] : 0.0;
        cost c1 = l1 ? -orig_[nidx] : 0.0;
        const auto qpbo_index = qpbo_.AddNode();
        qpbo_.AddUnaryTerm(qpbo_index, c0, c1);
        scratch_qpbo_indices_[nidx] = qpbo_index;
      }
    }

#ifndef NDEBUG
    const auto qpbo_size = qpbo_.GetNodeNum();
    std::cout << "[DBG] qpbo_size = " << qpbo_size << " / " << no_orig() << " ("
              << (100.0f * qpbo_size / no_orig()) << "%)" << std::endl;
#endif

    for_each_present_node_tuple([&](const auto nidx1, const auto nidx2) {
      const cost c01 = a0[nidx1] == 1 && a1[nidx2] == 1 ? QPBO_INF : 0.0;
      const cost c10 = a1[nidx1] == 1 && a0[nidx2] == 1 ? QPBO_INF : 0.0;

#ifndef NDEBUG
      const cost c00 = a0[nidx1] == 1 && a0[nidx2] == 1 ? QPBO_INF : 0.0;
      const cost c11 = a1[nidx1] == 1 && a1[nidx2] == 1 ? QPBO_INF : 0.0;
      assert(std::abs(c00) < 1e-8);
      assert(std::abs(c11) < 1e-8);
#else
      const cost c00 = 0.0, c11 = 0.0;
#endif

      if (c00 || c01 || c10 || c11) {
        const auto qpbo_idx1 = scratch_qpbo_indices_[nidx1];
        const auto qpbo_idx2 = scratch_qpbo_indices_[nidx2];
        qpbo_.AddPairwiseTerm(qpbo_idx1, qpbo_idx2, c00, c01, c10, c11);
      }
    });

    qpbo_.Solve();

    bool changed = false;
    enable_all_dumies();
    for (index nidx = 0; nidx < no_orig(); ++nidx) {
      const auto qpbo_idx = scratch_qpbo_indices_[nidx];
      const auto l = qpbo_idx != -1 ? qpbo_.GetLabel(qpbo_idx) : 0;
      changed = changed || (l != 0);
      a0[nidx] = l == 0 ? a0[nidx] : a1[nidx];

      if (a0[nidx] == 1)
        disable_dummy_for_node(nidx);
    }

    assert(compute_feasibility(a0));

    return changed;
  }

  void fusion_move()
  {
    std::cout << "f " << std::flush;
#ifndef NDEBUG
    const auto value_best_old = value_best_;
#endif
    if (fuse_two_assignments(assignment_best_, assignment_latest_))
      value_best_ = compute_primal(assignment_best_);
    assert(dbg::are_identical(value_best_, compute_primal(assignment_best_)));
    assert(value_best_ >= value_best_old - 1e-8);
    update_relaxed_best(assignment_best_, value_best_);
  }
#endif

  void assert_unset_alphas() const
  {
#ifndef NDEBUG
    for (const auto v : alphas_)
      assert(std::abs(v - 1) < epsilon);
#endif
  }

  void assert_negative_node_costs() const
  {
#ifndef NDEBUG
    for (const auto c : costs_)
      assert(c <= 0);
#endif
  }

  bool finalized_graph_, finalized_costs_;
  std::vector<cost> costs_;
  std::vector<cost> orig_;
  cost constant_;
  double scaling_;

  int iterations_;
  double temperature_;

  std::vector<range> clique_indices_;
  std::vector<index> clique_index_data_;

  std::vector<range> node_cliques_;
  std::vector<index> node_cliques_data_;

  std::vector<range> node_neighs_;
  std::vector<index> node_neigh_data_;

  mutable std::vector<cost_exp> scratch_;
  mutable std::vector<index> scratch_greedy_indices_;
  mutable std::vector<index> scratch_qpbo_indices_;

  cost value_latest_, value_best_;
  std::vector<int> assignment_latest_, assignment_best_;

  cost value_relaxed_, value_relaxed_best_;
  std::vector<cost_exp> assignment_relaxed_, assignment_relaxed_best_;
  std::vector<cost_exp> alphas_;

  std::default_random_engine gen_;
#ifdef ENABLE_QPBO
  qpbo::QPBO<cost> qpbo_;
#endif

  cost_exp threshold_feasibility_;
  cost_exp threshold_stability_;
  cost temperature_drop_factor_;
};

} // namespace mpopt::mwis::bregman_exp
} // namespace mpopt::mwis
} // namespace mpopt

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
