#ifndef LIBMPOPT_QAP_GREEDY_HPP
#define LIBMPOPT_QAP_GREEDY_HPP

namespace mpopt {
namespace qap {

template<typename ALLOCATOR>
class greedy {
public:
  using allocator_type = ALLOCATOR;
  using graph_type = graph<allocator_type>;
  using unary_node_type = typename graph_type::unary_node_type;
  using uniqueness_node_type = typename graph_type::uniqueness_node_type;
  using pairwise_node_type = typename graph_type::pairwise_node_type;
  using clock_type = std::chrono::high_resolution_clock;

  greedy(const graph_type& graph)
  : graph_(&graph)
  , gen_(std::random_device()())
  {
    frontier_.reserve(graph_->unaries().size());

    const size_t max_label_size = std::accumulate(
      graph_->unaries().begin(), graph_->unaries().end(), 0,
      [](const size_t a, const auto* node) {
        return std::max(a, node->factor.size());
      });

    cost_storage_.reserve(max_label_size);
  }

  void run()
  {
    frontier_.clear();
    unlabeled_ = graph_->unaries().size();
    graph_->for_each_node([](const auto* node) {
      node->factor.reset_primal();
    });

    while (unlabeled_ > 0) {
      if (frontier_.size() == 0)
        find_new_root();

      while (frontier_.size() > 0) {
        randomize_frontier();
        const unary_node_type* current = frontier_.back();
        frontier_.pop_back();
        process_frontier_node(current);
      }
    }

    assert(unlabeled_ == 0);
    assert(frontier_.size() == 0);

    for (const auto* node : graph_->uniqueness())
      if (!node->factor.is_primal_set())
        node->factor.primal() = node->factor.size();
  }

protected:
  const unary_node_type* find_new_root()
  {
    assert(frontier_.size() == 0);
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

    // Copy costs.
    cost_storage_.resize(root->factor.size());
    for (index p = 0; p < root->factor.size(); ++p)
      cost_storage_[p] = root->factor.get(p);

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
        cost_storage_[slot] = infinity;
      }
    });

    // Label the root.
    const auto it = std::min_element(cost_storage_.cbegin(), cost_storage_.cend());
    assert(*it < infinity);
    label_node(root, it - cost_storage_.cbegin());

    frontier_.push_back(root);
    return root;
  }

  void randomize_frontier()
  {
    std::shuffle(frontier_.begin(), frontier_.end(), gen_);
  }

  void label_node(const unary_node_type* node)
  {
    assert(!node->factor.is_primal_set());
    node->factor.round_independently();
    label_node_neighbors(node);
    --unlabeled_;
  }

  void label_node(const unary_node_type* node, index primal)
  {
    assert(!node->factor.is_primal_set());
    node->factor.primal() = primal;
    label_node_neighbors(node);
    --unlabeled_;
  }

  void label_node_neighbors(const unary_node_type* node)
  {
    assert(node->factor.is_primal_set());
    const auto primal = node->factor.primal();

    for (const auto* pairwise : node->backward) {
      const auto left = std::get<0>(pairwise->factor.primal());
      pairwise->factor.primal() = std::tuple(left, primal);
    }

    for (const auto* pairwise : node->forward) {
      const auto right = std::get<1>(pairwise->factor.primal());
      pairwise->factor.primal() = std::tuple(primal, right);
    }

    assert(primal < node->uniqueness.size());
    const auto& edge = node->uniqueness[primal];
    if (edge.node != nullptr) {
      assert(!edge.node->factor.is_primal_set());
      edge.node->factor.primal() = edge.slot;
    }
  }

  void process_frontier_node(const unary_node_type* node)
  {
    assert(node->factor.is_primal_set());
    const auto* pairwise_node = find_pairwise(node);
    if (pairwise_node != nullptr) {
      frontier_.push_back(node); // reinsert current node

      const unary_node_type* neighbor;
      if (pairwise_node->unary0 == node) {
        assert(pairwise_node->unary1 != node);
        neighbor = pairwise_node->unary1;
      } else {
        assert(pairwise_node->unary1 == node);
        neighbor = pairwise_node->unary0;
      }
      assert(!neighbor->factor.is_primal_set());
      frontier_.push_back(neighbor);

      // First copy unary costs.
      cost_storage_.resize(neighbor->factor.size());
      for (index p = 0; p < neighbor->factor.size(); ++p)
        cost_storage_[p] = neighbor->factor.get(p);

      // Aggregate all edge costs.
      for (const auto* pairwise : neighbor->backward) {
        if (pairwise->unary0->factor.is_primal_set()) {
          const auto p0 = pairwise->unary0->factor.primal();
          for (index p1 = 0; p1 < neighbor->factor.size(); ++p1) {
            cost_storage_[p1] += pairwise->factor.get(p0, p1);
          }
        }
      }

      for (const auto* pairwise : neighbor->forward) {
        if (pairwise->unary1->factor.is_primal_set()) {
          const auto p1 = pairwise->unary1->factor.primal();
          for (index p0 = 0; p0 < neighbor->factor.size(); ++p0) {
            cost_storage_[p0] += pairwise->factor.get(p0, p1);
          }
        }
      }

      // Remove infeasible connections.
      neighbor->traverse_uniqueness([this](const auto& edge, const auto slot) {
        if (edge.node->factor.is_primal_set()) {
          assert(edge.node->factor.primal() != edge.slot);
          cost_storage_[slot] = infinity;
        }
      });

      const auto it = std::min_element(cost_storage_.cbegin(), cost_storage_.cend());
      assert(*it < infinity);
      const index p = it - cost_storage_.cbegin();
      label_node(neighbor, p);
    }
  }

  const pairwise_node_type* find_pairwise(const unary_node_type* node)
  {
    index no_bw = 0, no_fw = 0;
    for (const auto* pairwise : node->backward)
      no_bw += pairwise->unary0->factor.is_primal_set() ? 0 : 1;
    for (const auto* pairwise : node->forward)
      no_fw += pairwise->unary1->factor.is_primal_set() ? 0 : 1;

    if (no_bw == 0 && no_fw == 0)
      return nullptr;

    std::uniform_int_distribution<index> dist(0, no_bw + no_fw - 1);
    index idx = dist(gen_);
    if (idx < no_bw) {
      for (const auto* pairwise : node->backward) {
        if (!pairwise->unary0->factor.is_primal_set()) {
          if (idx == 0)
            return pairwise;
          --idx;
        }
      }
    } else {
      idx -= no_bw;
      for (const auto* pairwise : node->forward) {
        if (!pairwise->unary1->factor.is_primal_set()) {
          if (idx == 0)
            return pairwise;
          --idx;
        }
      }
    }

    assert(false && "Should never be reached!");
    std::abort();
  }

  const graph_type* graph_;
  std::default_random_engine gen_;
  index unlabeled_;
  std::vector<const unary_node_type*> frontier_;
  std::vector<cost> cost_storage_;
};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
