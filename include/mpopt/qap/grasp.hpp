#ifndef LIBMPOPT_QAP_GRASP_HPP
#define LIBMPOPT_QAP_GRASP_HPP

namespace mpopt::qap {

template<typename ALLOCATOR>
struct candidate
{
  using allocator_type = ALLOCATOR;
  using graph_type = graph<allocator_type>;
  using unary_node_type = typename graph_type::unary_node_type;

  const unary_node_type* node;
  index primal;
  cost costs;

  candidate(const unary_node_type *node, index primal, cost costs)
      : node(node), primal(primal), costs(costs)
  {}
};

template<typename ALLOCATOR>
struct candidate_comparer
{
  //Use descending order to obtain min-heap
  bool operator () (const candidate<ALLOCATOR>& a, const candidate<ALLOCATOR>& b) const { return a.costs > b.costs; }
};

template<typename ALLOCATOR>
class grasp {
public:
  using allocator_type = ALLOCATOR;
  using graph_type = graph<allocator_type>;
  using unary_node_type = typename graph_type::unary_node_type;
  using uniqueness_node_type = typename graph_type::uniqueness_node_type;
  using pairwise_node_type = typename graph_type::pairwise_node_type;
  using clock_type = std::chrono::high_resolution_clock;
  using candidate_type = typename mpopt::qap::candidate<ALLOCATOR>;

  double alpha_;
  unsigned iterations_;

  grasp(const graph_type& graph, double alpha, unsigned iterations)
  : alpha_(alpha)
    , iterations_(iterations)
    , graph_(&graph)
    , gen_(std::random_device()())
  {
    //assert(graph_->unaries().size() > 0);
    const size_t max_label_size = std::accumulate(
      graph_->unaries().begin(), graph_->unaries().end(), 0,
      [](const size_t a, const auto* node) {
        return std::max(a, node->factor.size());
      });

    scratch_costs_.reserve(max_label_size);
    cost_matrix_.reserve(graph_->unaries().size() * max_label_size);
    candidate_heap_.reserve(graph_->unaries().size() * max_label_size);
    frontier_set_.resize(graph_->unaries().size());
    max_label_size_ = max_label_size;
  }


  void run()
  {
    //FIXME: This should be done in the constructor, but apparently graph_->unaries_ etc. is not initialized there
    const size_t max_label_size = std::accumulate(
        graph_->unaries().begin(), graph_->unaries().end(), 0,
        [](const size_t a, const auto* node) {
          return std::max(a, node->factor.size());
        });
    scratch_costs_.reserve(max_label_size);
    cost_matrix_.resize(graph_->unaries().size() * max_label_size);
    candidate_heap_.reserve(graph_->unaries().size() * max_label_size);
    frontier_set_.resize(graph_->unaries().size());
    max_label_size_ = max_label_size;

    cost best_cost = infinity;
    primal_storage best_primals(*graph_);

    for (unsigned i = 0; i < iterations_; ++i) {
      reset();
      const auto* root = initialize_assignment();
      complete_assignment(root);
      local_search();
      if (current_cost_ < best_cost) {
        best_cost = current_cost_;
        best_primals.save();
      }
    }

    best_primals.restore();

    for (const auto* node : graph_->uniqueness())
      if (!node->factor.is_primal_set())
        node->factor.primal() = node->factor.size();
  }

protected:

  const graph_type* graph_;

  std::default_random_engine gen_;
  index unlabeled_;
  std::vector<cost> scratch_costs_;
  std::vector<cost> cost_matrix_;
  std::vector<candidate_type> candidate_heap_;
  candidate_comparer<ALLOCATOR> candidate_comparer_;
  size_t max_label_size_;
  std::vector<bool> frontier_set_;
  cost current_cost_;

  void reset()
  {
    current_cost_ = 0;
    frontier_set_.assign(frontier_set_.size(), false);
    unlabeled_ = graph_->unaries().size();
    graph_->for_each_node([](const auto* node) {
      node->factor.reset_primal();
    });
  }

  bool is_feasible(const unary_node_type *node, index p) const
  {
    return node->uniqueness[p].node == nullptr || !node->uniqueness[p].node->factor.is_primal_set();
  }

  const unary_node_type* initialize_assignment()
  {
    const unary_node_type *root = find_new_root();
    label_root(root);
    return root;
  }

  void init_cost_matrix()
  {
    two_dimension_array_accessor a(graph_->unaries().size(), max_label_size_);
    for (const unary_node_type* node : graph_->unaries()) {
      for (index p = 0; p < node->factor.size(); ++p) {
        cost_matrix_[a.to_linear(node->idx, p)] = node->factor.get(p);
      }
    }
  }

  void update_candidate_heap(const unary_node_type* last_labeled_node)
  {
    two_dimension_array_accessor a(graph_->unaries().size(), max_label_size_);
    update_cost_matrix(last_labeled_node, a);

    candidate_heap_.clear();
    bool candidate_added = false;

    for (const unary_node_type* node : graph_->unaries()) {
      if (!frontier_set_[node->idx]) {
        continue;
      }

      for (index p = 0; p < node->factor.size(); ++p) {
        if (is_feasible(node, p)) {
          candidate_added = true;
          candidate_heap_.emplace_back(node, p, cost_matrix_[a.to_linear(node->idx, p)]);
        }
      }
    }

    if (!candidate_added) {
      const auto* root = find_new_root();
      for (index p = 0; p < root->factor.size(); ++p) {
        if (is_feasible(root, p)) {
          candidate_heap_.emplace_back(root, p, cost_matrix_[a.to_linear(root->idx, p)]);
        }
      }
    }

    std::make_heap(candidate_heap_.begin(), candidate_heap_.end(), candidate_comparer_);
  }

  void update_cost_matrix(const unary_node_type *last_labeled_node, const two_dimension_array_accessor &a)
  {
    for (const pairwise_node_type* edge : last_labeled_node->forward)
    {
      if (edge->unary1->factor.is_primal_set()) {
        continue;
      }

      for (index p = 0; p < edge->unary1->factor.size(); ++p) {
        if (!is_feasible(edge->unary1, p))
        {
          continue;
        }
        cost_matrix_[a.to_linear(edge->unary1->idx, p)] +=
            edge->factor.get(last_labeled_node->factor.primal(), p);
      }
    }

    for (const pairwise_node_type* edge : last_labeled_node->backward)
    {
      if (edge->unary0->factor.is_primal_set()) {
        continue;
      }

      for (index p = 0; p < edge->unary0->factor.size(); ++p) {
        if (!is_feasible(edge->unary0, p))
        {
          continue;
        }
        cost_matrix_[a.to_linear(edge->unary0->idx, p)] +=
            edge->factor.get(p, last_labeled_node->factor.primal());
      }
    }
  }

  void complete_assignment(const unary_node_type* root)
  {
    init_cost_matrix();
    const unary_node_type* previous_node = root;

    while (unlabeled_ > 0)
    {
      update_candidate_heap(previous_node);

      std::uniform_int_distribution<index> dist(0, std::ceil(alpha_ * candidate_heap_.size()));
      const auto candidate_index = dist(gen_);
      for (size_t i = 0; i < candidate_index; i++)
      {
        std::pop_heap(candidate_heap_.begin(), candidate_heap_.end(), candidate_comparer_);
      }
      const auto candidate = candidate_heap_.front();
      label_node(candidate.node, candidate.primal);
      current_cost_ += candidate.costs;
      previous_node = candidate.node;
    }
  }

  void local_search()
  {
    bool has_improved = true;

    while (has_improved) {
      has_improved = false;

      for (const unary_node_type* node : graph_->unaries()) {
        const auto current_primal = node->factor.primal();
        const auto node_unlabel_cost = cost_of_unlabel(node);
        assert(cost_of_label(node, current_primal) == -node_unlabel_cost);
        
        for (index p = 0; p < node->factor.size(); ++p) {
          if (p == current_primal) {
            continue;
          }

          const unary_node_type* swap_node = get_unary_with_primal(node, p);
          index swap_primal = get_swap_primal(node, swap_node);

          cost costs = node_unlabel_cost;
          costs += cost_of_label(node, p, swap_node, swap_primal);

          //FIXME The local search is running in infinite loops... again. Probably errors in cost calculation.
          cost tmp = cost_of_label(node, current_primal, swap_node, p);
          tmp = cost_of_label(node, current_primal, swap_node, p);
          assert(tmp == -node_unlabel_cost);
          if (swap_node != nullptr) {
            costs += cost_of_unlabel(swap_node, node);
            //Pass primal_unset to avoid counting pairwise cost twice if node and swap_node are adjacent.
            costs += cost_of_label(swap_node, swap_primal, node, node->factor.primal_unset);
          }

          if (costs < 0) {
            has_improved = true;
            label_node(node, p);
            if (swap_node != nullptr) {
              label_node(swap_node, swap_primal);
            }
            current_cost_ += costs;
          }
        }
      }
    }
  }
  
  index get_swap_primal(const unary_node_type *node, const unary_node_type *swap_node) const
  {
    index swap_primal = 0;
    if (swap_node != nullptr) {
      //default to dummy (?) in case the (non-null) uniqueness_node is not linked with swap_node
      swap_primal = swap_node->factor.size() - 1;
      const uniqueness_node_type* uniqueness_node = node->uniqueness[node->factor.primal()].node;
      //FIXME If there's a way to obtain the primal in swap_node associated with uniqueness_node without iterating
      // over all primals, it should be used here.
      // Something like uniqueness_node->unaries[swap_node->idx].slot (which currently would be wrong, afaik)
      for (index q = 0; q < swap_node->factor.size(); q++) {
        if (swap_node->uniqueness[q].node == uniqueness_node) {
          //If uniqueness_node == nullptr, then swap_primal will be a dummy label.
          swap_primal = q;
          break;
        }
      }
    }
    return swap_primal;
  }

  const unary_node_type *get_unary_with_primal(const unary_node_type *node, index p) const
  {
    const uniqueness_node_type* uniqueness_node = node->uniqueness[p].node;
    if (uniqueness_node == nullptr) {
      return nullptr;
    }
    if (uniqueness_node->factor.is_primal_set()) {
      return uniqueness_node->unaries[uniqueness_node->factor.primal()].node;
    }
    return nullptr;
  }

  double cost_of_relabel(const unary_node_type* node, const index primal, const unary_node_type* swap_node, const index swap_primal)
  {
    cost costs = 0;

    costs += cost_of_unlabel(node);
    if (swap_node != nullptr) {
      costs += cost_of_unlabel(swap_node, node);
    }

    costs += cost_of_label(node, primal, swap_node, swap_primal);
    if (swap_node != nullptr) {
      //Pass primal_unset to avoid counting pairwise cost twice if node and swap_node are adjacent.
      costs += cost_of_label(swap_node, swap_primal, node, node->factor.primal_unset);
    }

    return costs;
  }

  cost cost_of_label(const unary_node_type *node, const index primal, const unary_node_type *swap_node = nullptr, const index swap_primal = 0) const
  {
    cost costs = 0;
    costs += node->factor.get(primal);
    for (const pairwise_node_type* edge : node->forward) {
      if (edge->unary1 == swap_node) {
        //This is kind of a confusing hack to avoid accounting for pairwise cost twice.
        if (swap_primal != swap_node->factor.primal_unset) {
          costs += edge->factor.get(primal, swap_primal);
        }
        continue;
      }
      //FIXME: There are cases where primal == primal1 and costs go to infinity.
      const auto [primal0, primal1] = edge->factor.primal();
      costs += edge->factor.get(primal, primal1);
    }

    for (const pairwise_node_type* edge : node->backward) {
      if (edge->unary0 == swap_node) {
        //This is kind of a confusing hack to avoid accounting for pairwise cost twice.
        if (swap_primal != swap_node->factor.primal_unset) {
          costs += edge->factor.get(swap_primal, primal);
        }
        continue;
      }
      const auto [primal0, primal1] = edge->factor.primal();
      costs += edge->factor.get(primal0, primal);
    }

    return costs;
  }

  cost cost_of_unlabel(const unary_node_type *node, const unary_node_type *exclude_node = nullptr) const
  {
    cost costs = 0;
    costs -= node->factor.get(node->factor.primal());
    for (const pairwise_node_type* edge : node->forward) {
      if (edge->unary1 == exclude_node) {
        continue;
      }
      const auto [primal0, primal1] = edge->factor.primal();
      costs -= edge->factor.get(primal0, primal1);
    }

    for (const pairwise_node_type* edge : node->backward) {
      if (edge->unary0 == exclude_node) {
        continue;
      }
      const auto [primal0, primal1] = edge->factor.primal();
      costs -= edge->factor.get(primal0, primal1);
    }
    return costs;
  }

  const unary_node_type* find_new_root()
  {
    assert(unlabeled_ > 0);

    const unary_node_type* root = nullptr;
    std::uniform_int_distribution<index> unlabeled_dist(0, unlabeled_ - 1);
    index idx = unlabeled_dist(gen_);
    for (const auto* node : graph_->unaries()) {
      if (!node->factor.is_primal_set()) {
        if (idx == 0) {
          root = node;
          break;
        }
        --idx;
      }
    }

    return root;
  }

  const unary_node_type* label_root(const unary_node_type *root) {
    // Copy costs.
    scratch_costs_.resize(root->factor.size());
    for (index p = 0; p < root->factor.size(); ++p)
      scratch_costs_[p] = root->factor.get(p);

#ifndef NDEBUG
    for (const auto* pairwise : root->backward)
      assert(!pairwise->unary0->factor.is_primal_set());

    for (const auto* pairwise : root->forward)
      assert(!pairwise->unary1->factor.is_primal_set());
#endif

    // Remove infeasible connections.
    root->traverse_uniqueness([this](const auto& edge, const auto slot) {
      if (edge.node->factor.is_primal_set()) {
        assert(edge.node->factor.primal() != edge.slot);
        scratch_costs_[slot] = infinity;
      }
    });

    // Label the root.
    const auto it = std::min_element(scratch_costs_.cbegin(), scratch_costs_.cend());
    assert(*it < infinity);
    label_node(root, it - scratch_costs_.cbegin());
    current_cost_ += *it;
    return root;
  }

  void label_node(const unary_node_type* node, index primal)
  {
    //assert(!node->factor.is_primal_set());
    node->factor.primal() = primal;
    label_node_neighbors(node);
    frontier_set_[node->idx] = false;
    --unlabeled_;
  }

  void label_node_neighbors(const unary_node_type* node)
  {
    assert(node->factor.is_primal_set());
    const auto primal = node->factor.primal();

    for (const pairwise_node_type* pairwise : node->backward) {
      if (!pairwise->unary0->factor.is_primal_set()) {
        frontier_set_[pairwise->unary0->idx] = true;
      }
      const auto left = std::get<0>(pairwise->factor.primal());
      pairwise->factor.primal() = std::tuple(left, primal);
    }

    for (const auto* pairwise : node->forward) {
      if (!pairwise->unary1->factor.is_primal_set()) {
        frontier_set_[pairwise->unary1->idx] = true;
      }
      const auto right = std::get<1>(pairwise->factor.primal());
      pairwise->factor.primal() = std::tuple(primal, right);
    }

    assert(primal < node->uniqueness.size());
    const auto& edge = node->uniqueness[primal];
    if (edge.node != nullptr) {
      //assert(!edge.node->factor.is_primal_set());
      edge.node->factor.primal() = edge.slot;
    }
  }

};

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
