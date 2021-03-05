#ifndef LIBMPOPT_MWIS_SOLVER_HPP
#define LIBMPOPT_MWIS_SOLVER_HPP

namespace mpopt {
namespace mwis {

struct range {
  index begin;
  index end;
  index size;
};

class solver {
public:

  solver()
  : finalized_(false)
  , gen_(std::random_device()())
  , qpbo_(0, 0)
  {
  }

  index add_node(cost cost)
  {
    assert(!finalized_);
    assert(no_cliques() == 0);
    costs_.push_back(cost);
    orig_.push_back(cost);
    return costs_.size() - 1;
  }

  index add_clique(std::vector<index> indices)
  {
    assert(!finalized_);

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
    if (finalized_)
      return;

    //
    // Initialize parameters.
    //

    temperature_ = 1;
    constant_ = 0;

    //
    // Construct node to clique mapping.
    //
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
    value_latest_ = primal(assignment_latest_);
    assert(std::abs(value_latest_) < 1e-8);

    assignment_best_ = assignment_latest_;
    value_best_ = value_latest_;

    assignment_relaxed_.assign(assignment_latest_.cbegin(), assignment_latest_.cend());
    value_relaxed_ = value_latest_;

    //
    // Update all lambdas (without smoothing, invariants do not hold) to ensure
    // that invariants (negative node costs) hold.
    //

    for (index clique_idx = 0; clique_idx < no_cliques(); ++clique_idx)
      update_lambda<false>(clique_idx);

    //
    // Try to improve naive assignment by greedily sampling an assignment.
    // This will be used for inital temperature selection.
    //

    greedy();
    if (value_latest_ > value_best_) {
      value_best_ = value_latest_;
      assignment_best_ = assignment_latest_;
    }
    update_temperature();

    finalized_ = true;
  }

  cost dual_relaxed() const
  {
    // Compute $D(\lambda) = \sum_i \lambda_i + \max_{x \in [0, 1]^N} <c^\lambda, x>$.
    // Note that the last term is zero if c^\lambda <= 0 (i.e. after updates
    // have been performed).
    assert_negative_node_costs();

    // sum of all lambdas = constant_
    return constant_;
  }

  cost dual_smoothed() const
  {
    // Compute $D^T(\lambda) = \sum_i \lambda_i + \max_{x \in [0, 1]^N} [ <c^\lambda, x> + T H(x) ]$.
    // The max of $cx - Tx log x$ is obtained at $x = exp(c/T - 1)$.
    // If $c^\lambda <= 0$ it is within the range [0, 1] and the obtained value is
    // $T / e * exp(c / T)$.
    assert_negative_node_costs();

    // sum of all lambdas = constant_
    auto f = [this](const auto a, const auto c) { return a + std::exp(c / temperature_); };
    return constant_ + temperature_ / std::exp(1) * std::accumulate(costs_.cbegin(), costs_.cend(), 0.0, f);
  }

  cost primal(const std::vector<int>& assignment) const
  {
    // Same as relaxed objective, we just check that $x_i \in {0, 1}$.
#ifndef NDEBUG
    for (auto a : assignment)
      assert(a == 0 || a == 1);
#endif

    return primal_relaxed(assignment);
  }

  template<typename T>
  cost primal_relaxed(const std::vector<T>& assignment) const
  {
    // Compute $<c, x>$ s.t. uniqueness constraints.
    assert(assignment.size() == no_nodes());
    cost result = constant_;

    for (index node_idx = 0; node_idx < no_nodes(); ++node_idx) {
      const auto x = assignment[node_idx];
      assert(x >= 0 && x <= 1);
      result += costs_[node_idx] * x;
    }

    for (const auto& cl : clique_indices_) {
      double sum = 0;
      for (index idx = cl.begin; idx < cl.end; ++idx)
        sum += assignment[clique_index_data_[idx]];
      if (std::abs(sum - 1) >= 1e-6)
        result = -infinity;
    }

    return result;
  }

  template<typename T>
  cost primal_smoothed(const std::vector<T>& assignment) const
  {
    // Compute $<c, x> + T H(x)$ s.t. uniqueness constraints.
    auto relaxed = primal_relaxed(assignment);

    // Compute real entropy of `assignment` (not an estimated one like the
    // `entropy()`).
    auto f = [](const auto a, const auto x) {
      return a + (x > 1e-8 ? x * std::log(x) : 0.0);
    };
    const auto H = -std::accumulate(assignment.cbegin(), assignment.cend(), 0.0, f);

    return relaxed + temperature_ * H;
  }

  cost entropy() const
  {
    // Estimate entropy $- \sum_i x_i log x_i$ by assuming $x_i = exp(c^\lambda_i / T)$.
    // Note that the fully simplified formula is more stable (log(0) impossible).
    assert_negative_node_costs();

    auto f = [this](const auto a, const auto c_i) {
      return a + c_i * std::exp(c_i / temperature_);
    };

    return -std::accumulate(costs_.cbegin(), costs_.cend(), 0.0, f) / temperature_;
  }

  void update_temperature()
  {
    const double gamma = 2.0;
    const auto d = dual_relaxed();
    const auto p = value_relaxed_;

    const auto new_temp = (d - p) / (gamma * entropy());
    temperature_ = std::min(temperature_, new_temp);
  }

  void run(const int batch_size=default_batch_size, const int max_batches=default_max_batches)
  {
    std::cout << "initial dual = " << dual_relaxed() << std::endl;
    signal_handler h;
    for (int i = 0; i < max_batches && !h.signaled(); ++i) {
      auto start = std::chrono::steady_clock::now();
      for (int j = 0; j < batch_size; ++j)
        single_pass();
      update_integer_assignment();
      auto end = std::chrono::steady_clock::now();
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

      std::cout << "it=" << (i+1) * batch_size << " "
                << "d=" << dual_relaxed() << " "
                << "p=" << value_relaxed_ << " "
                << "p01=" << value_latest_ << " "
                << "p01*=" << value_best_ << " "
#ifndef NDEBUG
                << "d_T=" << dual_smoothed() << " "
                << "p_T=" << primal_smoothed(assignment_relaxed_) << " "
#endif
                << "T=" << temperature_ << " "
                << "t/it=" << ms / batch_size << std::endl;
    }
  }

  index no_nodes() const { return costs_.size(); }
  index no_orig() const { return orig_.size(); }
  index no_cliques() const { return clique_indices_.size(); }

protected:
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

    assert(!std::isnan(msg));
    constant_ += msg;
    assert(!std::isnan(constant_));
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

      cost max_allowed_x = infinity;
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

    value_relaxed_ = primal_relaxed(assignment_relaxed_);
  }

  void update_integer_assignment()
  {
    greedy();
#ifndef LIBMPOPT_MWIS_DISABLE_FUSION_MOVES
    fusion_move();
#else
    if (value_latest_ > value_best_) {
      value_best_ = value_latest_;
      assignment_best_ = assignment_latest_;
    }
#endif
  }

  void greedy()
  {
    std::shuffle(clique_indices_.begin(), clique_indices_.end(), gen_);

    std::fill(assignment_latest_.begin(), assignment_latest_.end(), -1);
    for (const auto& cl : clique_indices_)
      round(cl);

    value_latest_ = primal(assignment_latest_);
  }

  void round(const range& cl)
  {
    int count = 0;
    for (index idx = cl.begin; idx < cl.end; ++idx)
      count += assignment_latest_[clique_index_data_[idx]] == 1 ? 1 : 0;
    assert(count == 0 || count == 1);

    if (count > 0)
      return;

    copy_clique_in(cl);
    for (index idx = 0; idx < cl.size; ++idx) {
      const auto nidx = clique_index_data_[cl.begin + idx];
      if (assignment_latest_[nidx] == 0)
        scratch_[idx] = -infinity;
    }

    const auto argmax = std::max_element(scratch_.begin(), scratch_.end()) - scratch_.begin();
    const auto nidx = clique_index_data_[cl.begin + argmax];
    assignment_latest_[nidx] = 1;
    for (index idx = node_neighs_[nidx].begin; idx < node_neighs_[nidx].end; ++idx)
      assignment_latest_[node_neigh_data_[idx]] = 0;
  }

  void fuse_two_assignments(std::vector<int>& a0, const std::vector<int>& a1)
  {
    constexpr cost QPBO_INF = 1e20;
    assert(a0.size() == no_nodes());
    assert(a1.size() == no_nodes());

    qpbo_.Reset();
    qpbo_.AddNode(no_orig());

    for (index nidx = 0; nidx < no_orig(); ++nidx) {
      assert(a0[nidx] == 0 || a0[nidx] == 1);
      assert(a1[nidx] == 0 || a1[nidx] == 1);

      const bool l0 = a0[nidx] == 1, l1 = a1[nidx] == 1;
      cost c0 = l0 ? -orig_[nidx] : 0.0;
      cost c1 = l1 ? -orig_[nidx] : 0.0;

      if (l0 == l1)
        c1 = QPBO_INF;

      qpbo_.AddUnaryTerm(nidx, c0, c1);
    }

    for (index nidx = 0; nidx < no_orig(); ++nidx) {
      for (index idx = node_neighs_[nidx].begin; idx < node_neighs_[nidx].end; ++idx) {
        const auto nidx2 = node_neigh_data_[idx];
        if (nidx < nidx2 && nidx2 < no_orig()) {
          cost c00 = a0[nidx] == 1 && a0[nidx2] == 1 ? QPBO_INF : 0.0;
          cost c01 = a0[nidx] == 1 && a1[nidx2] == 1 ? QPBO_INF : 0.0;
          cost c10 = a1[nidx] == 1 && a0[nidx2] == 1 ? QPBO_INF : 0.0;
          cost c11 = a1[nidx] == 1 && a1[nidx2] == 1 ? QPBO_INF : 0.0;
          if (c00 || c01 || c10 || c11)
            qpbo_.AddPairwiseTerm(nidx, nidx2, c00, c01, c10, c11);
        }
      }
    }

    qpbo_.Solve();

    for (index nidx = 0; nidx < no_orig(); ++nidx)
      a0[nidx] = qpbo_.GetLabel(nidx) == 0 ? a0[nidx] : a1[nidx];

    // We have built the QPBO problem only with `no_orig()` nodes, so the
    // remaining "dummy" nodes of `costs_` are still unset. We set them to the
    // correct value now.

    for (const auto& cl : clique_indices_) {
      assert(cl.size >= 3);

      int count = 0;
      for (index idx = cl.begin; idx < cl.end - 1; ++idx) {
        const auto nidx = clique_index_data_[idx];
        count += a0[nidx] == 1 ? 1 : 0;
      }

      assert(count == 0 || count == 1);
      const auto nidx = clique_index_data_[cl.end - 1];
      a0[nidx] = count == 1 ? 0 : 1;
    }
  }

  void fusion_move()
  {
#ifndef NDEBUG
    const auto value_best_old = value_best_;
#endif
    fuse_two_assignments(assignment_best_, assignment_latest_);
    value_best_ = primal(assignment_best_);
    assert(value_best_ >= value_best_old - 1e-8);
  }

  void assert_negative_node_costs() const
  {
#ifndef NDEBUG
    for (const auto c : costs_)
      assert(c <= 0);
#endif
  }

  bool finalized_;
  std::vector<cost> costs_;
  std::vector<cost> orig_;
  cost constant_;

  std::vector<range> clique_indices_;
  std::vector<index> clique_index_data_;

  std::vector<range> node_cliques_;
  std::vector<index> node_cliques_data_;

  std::vector<range> node_neighs_;
  std::vector<index> node_neigh_data_;

  double temperature_;
  mutable std::vector<cost> scratch_;

  cost value_latest_;
  std::vector<int> assignment_latest_;
  cost value_best_;
  std::vector<int> assignment_best_;

  cost value_relaxed_;
  std::vector<double> assignment_relaxed_;

  std::default_random_engine gen_;
  qpbo::QPBO<cost> qpbo_;
};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
