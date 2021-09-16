#ifndef LIBMPOPT_QAP_LOCAL_SEARCH_HPP
#define LIBMPOPT_QAP_LOCAL_SEARCH_HPP

namespace mpopt::qap {

template<typename ALLOCATOR>
class local_search
{
public:
  using allocator_type = ALLOCATOR;
  using graph_type = graph<allocator_type>;
  using unary_node_type = typename graph_type::unary_node_type;
  using uniqueness_node_type = typename graph_type::uniqueness_node_type;
  using pairwise_node_type = typename graph_type::pairwise_node_type;

  explicit local_search(const graph_type *graph)
      : graph_(graph)
  {}

  double run()
  {
    if (!initialized_) {
      initialize();
    }

    return two_exchange();
  }

protected:
  const graph_type* graph_;
  std::vector<index> eq_primal_lookup_{};
  size_t max_label_size_{};
  bool initialized_{};

  double two_exchange()
  {
    cost cost_change = 0;
    bool has_improved = true;

    while (has_improved) {
      has_improved = false;
      for (const unary_node_type *node: graph_->unaries()) {
        const auto current_primal = node->factor.primal();
        const auto node_unlabel_cost = cost_of_unlabel(node);

        for (index p = 0; p < node->factor.size(); ++p) {
          if (p == current_primal) {
            continue;
          }

          const unary_node_type *swap_node = get_unary_with_primal(node, p);
          assert(node != swap_node);
          index swap_primal = get_swap_primal(node, swap_node);

          cost costs = node_unlabel_cost;
          costs += cost_of_label(node, p, swap_node, swap_primal);

          if (swap_node != nullptr) {
            costs += cost_of_unlabel(swap_node, node);
            //Pass primal_unset to avoid counting pairwise cost twice if node and swap_node are adjacent.
            costs += cost_of_label(swap_node, swap_primal, node, node->factor.primal_unset);
          }

          if (costs < -epsilon) {
            has_improved = true;
            swap_labels(node, p, swap_node, swap_primal);
            cost_change += costs;
            break;
          }
        }

        if (has_improved) {
          break;
        }
      }
    }

    return cost_change;
  }

  void initialize()
  {
    max_label_size_ = std::accumulate(
        graph_->unaries().begin(), graph_->unaries().end(), 0,
        [](const size_t a, const auto* node) {
          return std::max(a, node->factor.size());
        });
    eq_primal_lookup_.resize(graph_->unaries().size() * max_label_size_ * graph_->unaries().size());
    prepare_eq_primal_lookup();
    initialized_ = true;
  }

  void prepare_eq_primal_lookup()
  {
    two_dimension_array_accessor a(graph_->unaries().size(), max_label_size_);
    two_dimension_array_accessor b(graph_->unaries().size() * max_label_size_, graph_->unaries().size());
    for (const unary_node_type* node : graph_->unaries()) {
      for (const unary_node_type* other : graph_->unaries()) {
        if (node == other) {
          continue;
        }

        for (int p = 0; p < node->factor.size(); ++p) {
          eq_primal_lookup_[b.to_linear(a.to_linear(node->idx, p), other->idx)] = get_equivalent_primal(node, p, other);
        }
      }
    }
  }

  index get_equivalent_primal(const unary_node_type *node, const index primal, const unary_node_type *swap_node) const
  {
    const uniqueness_node_type* uniqueness_node = node->uniqueness[primal].node;
    for (index q = 0; q < swap_node->factor.size(); q++) {
      if (swap_node->uniqueness[q].node == uniqueness_node) {
        //If uniqueness_node == nullptr, then swap_primal will be a dummy label.
        return q;
      }
    }

    //default to dummy (?) in case the (non-null) uniqueness_node is not linked with swap_node
    return swap_node->factor.size() - 1;
  }

  index get_swap_primal(const unary_node_type *node, const unary_node_type *swap_node) const
  {
    if (swap_node == nullptr) {
      return 0;
    }
    two_dimension_array_accessor a(graph_->unaries().size(), max_label_size_);
    two_dimension_array_accessor b(graph_->unaries().size() * max_label_size_, graph_->unaries().size());
    return eq_primal_lookup_[b.to_linear(a.to_linear(node->idx, node->factor.primal()), swap_node->idx)];
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
      const auto primal1 = edge->unary1->factor.primal();
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
      const auto primal0 = edge->unary0->factor.primal();
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
      const auto primal0 = edge->unary0->factor.primal();
      const auto primal1 = edge->unary1->factor.primal();
      costs -= edge->factor.get(primal0, primal1);
    }

    for (const pairwise_node_type* edge : node->backward) {
      if (edge->unary0 == exclude_node) {
        continue;
      }
      const auto primal0 = edge->unary0->factor.primal();
      const auto primal1 = edge->unary1->factor.primal();
      costs -= edge->factor.get(primal0, primal1);
    }
    return costs;
  }

  void swap_labels(const unary_node_type* node, index primal, const unary_node_type* swap_node, index swap_primal) {
    assert(node->factor.is_primal_set());
    if (swap_node == nullptr) {
      reset_uniqueness(node);
      node->factor.primal() = primal;
      label_uniqueness(node);
    } else {
      reset_uniqueness(node);
      reset_uniqueness(swap_node);
      node->factor.primal() = primal;
      swap_node->factor.primal() = swap_primal;
      label_uniqueness(node);
      label_uniqueness(swap_node);
    }
  }

  void reset_uniqueness(const unary_node_type* node) const
  {
    assert(node->factor.is_primal_set());
    const uniqueness_node_type* uniqueness_node = node->uniqueness[node->factor.primal()].node;
    if (uniqueness_node != nullptr) {
      uniqueness_node->factor.reset_primal();
    }
  }

  void label_uniqueness(const unary_node_type *node) const
  {
    const link_info<uniqueness_node_type>& link = node->uniqueness[node->factor.primal()];
    if (link.node != nullptr) {
      link.node->factor.primal() = link.slot;
    }
  }

};

}

#endif //LIBMPOPT_QAP_LOCAL_SEARCH_HPP
