#ifndef LIBMPOPT_MWIS_SOLVER_CONT_TEMP_HPP
#define LIBMPOPT_MWIS_SOLVER_CONT_TEMP_HPP

namespace mpopt {
namespace mwis {
namespace original {

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
    temperature_ = 10000;
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

  cost dual_relaxed() const { return dual_relaxed_scaled() * scaling_;}

  cost primal() const { return value_best_ * scaling_; }

  cost primal_relaxed() const { return value_relaxed_ * scaling_; }

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

    for (int i = 0; i < max_batches && !h.signaled(); ++i) {
      t_total.start();
      for (int j = 0; j < batch_size; ++j)
        single_pass();
      update_integer_assignment(greedy_generations);
      iterations_ += batch_size;
      t_total.stop();

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
                << "t=" << t_total.seconds<true>() << " "
                << "T=" << temperature_ << " "
                << "total=" << t_total.milliseconds<true>() / iterations_ << "ms/it" << std::endl;

      if (gap < 1e-2) {
        std::cout << "Gap limit reached." << std::endl;
        break;
      }
    }

    std::cout << "Optimization stopped: "
              << "d=" << dual_relaxed() << " "
              << "p=" << primal() << std::endl;
  }

  index no_nodes() const { return costs_.size(); }
  index no_orig() const { return orig_.size(); }
  index no_cliques() const { return clique_indices_.size(); }

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

  cost dual_relaxed_scaled() const
  {
    // Compute $D(\lambda) = \sum_i \lambda_i + \sum_i \max_{x \in [0, 1]^N} <c^\lambda, x>$.
    // Note that the sum of all lambdas = constant_.
    if constexpr (initial_reparametrization) {
      // We now that the second sum is always zero, because this is ensured after the first
      // reparametrization run.
      return constant_;
    } else {
      // Compute $\sum_i \max_{x \in [0, 1]^N} <c^\lambda, x>$.
      const auto f = [](const auto a, const auto b) {
        return a + std::max(b, cost{0});
      };

      const auto sum_max_i = std::accumulate(costs_.cbegin(), costs_.cend(), cost{0}, f);
      return constant_ + sum_max_i;
    }
  }

  cost dual_smoothed_scaled() const
  {
    // Compute $D^T(\lambda) = \sum_i \lambda_i + \max_{x \in [0, 1]^N} [ <c^\lambda, x> + T H(x) ]$.
    // The max of $cx - Tx log x$ is obtained at $x = exp(c/T - 1)$.
    // If $c^\lambda <= 0$ it is within the range [0, 1] and the obtained value is
    // $T / e * exp(c / T)$.
    assert_negative_node_costs();

    // sum of all lambdas = constant_
    auto f = [this](const auto a, const auto c) { return a + std::exp(c / temperature_); };
    return constant_ + temperature_ * std::accumulate(costs_.cbegin(), costs_.cend(), 0.0, f);
  }

  cost entropy_scaled() const
  {
    // Estimate entropy $- \sum_i x_i log x_i$ by assuming $x_i = exp(c^\lambda_i / T)$.
    // Note that the fully simplified formula is more stable (log(0) impossible).
    assert_negative_node_costs();

    auto f1 = [this](const auto a, const auto c_i) {
      return a + std::exp(c_i / temperature_);
    };

    auto f2 = [this](const auto a, const auto c_i) {
      return a + c_i * std::exp(c_i / temperature_);
    };

    return std::accumulate(costs_.cbegin(), costs_.cend(), 0.0, f1) -
           std::accumulate(costs_.cbegin(), costs_.cend(), 0.0, f2) / temperature_;
  }

  void update_temperature()
  {
    // Note: This is the implementation for section "7.2 Method 2: Duality gap" of the paper.
    const auto d = dual_smoothed_scaled();
    const auto p = std::max(value_relaxed_, value_best_);

    // std::cout << "update_temperature => d=" << d << " p=" << p << " entropy=" << entropy_scaled() << " drop=" << temperature_drop_factor_ << std::endl;

    auto new_temp = (d - p) / (entropy_scaled() / temperature_drop_factor_);
    assert(std::isnormal(new_temp) && new_temp >= 0.0);

    temperature_ = std::max(std::min(temperature_, new_temp), 1e-10);
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

    assignment_latest_.resize(no_nodes());
    for (index nidx = 0; nidx < no_nodes(); ++nidx)
      assignment_latest_[nidx] = nidx < no_orig() ? 0 : 1;
    value_latest_ = compute_primal(assignment_latest_);
    assert(std::abs(value_latest_) < 1e-8);

    assignment_best_ = assignment_latest_;
    value_best_ = value_latest_;

    assignment_relaxed_.assign(assignment_latest_.cbegin(), assignment_latest_.cend());
    value_relaxed_ = value_latest_;

    //
    // Initialize remaining things.
    //

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
    value_best_ = compute_primal(assignment_best_);
    iterations_ = 0;

    // Try to improve naive assignment by greedily sampling an assignment.
    // This will be used for inital temperature selection.
    greedy();
    if (value_latest_ > value_best_) {
      value_best_ = value_latest_;
      assignment_best_ = assignment_latest_;
    }
    update_temperature();

    finalized_costs_ = true;
  }

  void single_pass()
  {
    for (index clique_idx = 0; clique_idx < no_cliques(); ++clique_idx)
      update_lambda(clique_idx);

    compute_relaxed_truncated_projection();
    update_temperature();
  }

  void copy_clique_in(const range& cl)
  {
    scratch_.resize(cl.size, 0.0);
    for (index idx = 0; idx < cl.size; ++idx)
      scratch_[idx] = costs_[clique_index_data_[cl.begin + idx]];
  }

  void copy_clique_out(const range& cl)
  {
    for (index idx = 0; idx < cl.size; ++idx)
      costs_[clique_index_data_[cl.begin + idx]] = scratch_[idx];
  }

  template<bool smoothing=true>
  void update_lambda(const index clique_idx)
  {
    const auto& cl = clique_indices_[clique_idx];
    copy_clique_in(cl);

    auto f = [](const cost a, const cost b) { return std::max(a, b); };
    const cost maximum = std::accumulate(scratch_.begin(), scratch_.end(), -infinity, f);

    cost msg = 0.0;
    if constexpr (smoothing) {
      for (auto c : scratch_)
        msg += std::exp((c - maximum) / temperature_);
      msg = maximum + temperature_ * std::log(msg);
    } else {
      msg = maximum;
    }
    assert(std::isfinite(msg));

    constant_ += msg;
    for (auto& c : scratch_)
      c -= msg;

    copy_clique_out(cl);
  }

  void compute_relaxed_truncated_projection()
  {
    auto slack = [this](const index clique_idx) {
      assert(no_orig() + clique_idx < assignment_relaxed_.size());
      return &assignment_relaxed_[no_orig() + clique_idx];
    };

    for (index clique_idx = 0; clique_idx < no_cliques(); ++clique_idx)
      *slack(clique_idx) = 1.0;

    for (index node_idx = 0; node_idx < no_orig(); ++node_idx) {
      const auto& nc = node_cliques_[node_idx];
      const auto x = std::exp(costs_[node_idx] / temperature_);

      cost max_allowed_x = 1.0;
      for (index idx = nc.begin; idx < nc.end; ++idx) {
        const auto clique_idx = node_cliques_data_[idx];
        max_allowed_x = std::min(max_allowed_x, *slack(clique_idx));
      }
      assert(!std::isinf(max_allowed_x));

      const auto y = std::min(x, max_allowed_x);
      assignment_relaxed_[node_idx] = y;

      for (index idx = nc.begin; idx < nc.end; ++idx) {
        const auto clique_idx = node_cliques_data_[idx];
        *slack(clique_idx) -= y;
      }
    }

    value_relaxed_ = compute_primal_relaxed(assignment_relaxed_);
  }

  bool update_integer_assignment(int greedy_generations)
  {
    bool has_improved = false;
    for (int i = 0; i < greedy_generations; ++i)
      has_improved |= update_integer_assignment();
    if (greedy_generations > 0)
      std::cout << std::endl;
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
  }
#endif

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

  mutable std::vector<cost> scratch_;
  mutable std::vector<index> scratch_greedy_indices_;
  mutable std::vector<index> scratch_qpbo_indices_;

  cost value_latest_;
  std::vector<int> assignment_latest_;
  cost value_best_;
  std::vector<int> assignment_best_;

  cost value_relaxed_;
  std::vector<cost> assignment_relaxed_;

  std::default_random_engine gen_;
#ifdef ENABLE_QPBO
  qpbo::QPBO<cost> qpbo_;
#endif

  double temperature_drop_factor_;
};

} // namespace mpopt::mwis::original
} // namespace mpopt::mwis
} // namespace mpopt

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
