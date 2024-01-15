#ifndef LIBMPOPT_QAP_PRIMAL_STORAGE_HPP
#define LIBMPOPT_QAP_PRIMAL_STORAGE_HPP

namespace mpopt {
namespace qap {

template<typename ALLOCATOR>
class primal_storage {
public:
  using allocator_type = ALLOCATOR;
  using graph_type = graph<allocator_type>;

  primal_storage(const graph_type &graph)
  : graph_(&graph)
  , data_(graph_->unaries().size())
  , assignment_(graph_->unaries().size())
  {
  }

  void resize()
  {
    data_.resize(graph_->unaries().size());
    assignment_.resize(graph_->unaries().size());
  }

  void save()
  {
    auto it = data_.begin(); auto as = assignment_.begin();
    for (const auto* node : graph_->unaries()) {
      assert(it != data_.end()); assert(as != assignment_.end());
      *it++ = node->factor.primal();

      *as = -1;
      if (node->factor.is_primal_set()) {
        auto p = node->factor.primal();
        assert(p < node->uniqueness.size());

        assert((node->uniqueness[p].node == nullptr) == (p == node->uniqueness.size() - 1));
        if (p < node->uniqueness.size() - 1)
          *as = node->uniqueness[p].node->label_idx;
      }
      ++as;
    }
    assert(it == data_.end()); assert(as == assignment_.end());
  }

  void restore()
  {
    // Restore the unary primals.

    auto it = data_.begin();
    for (const auto* node : graph_->unaries()) {
      assert(it != data_.end());
      node->factor.primal() = *it++;
    }
    assert(it == data_.end());

    // Set the uniqueness and pairwise primals consistently.

    for (const auto* node : graph_->uniqueness()) {
      node->factor.primal() = node->unaries.size();
      node->traverse_unaries([&](const auto& edge, const index slot) {
        if (edge.node->factor.primal() == edge.slot) {
          assert(node->factor.primal() == node->unaries.size());
          node->factor.primal() = slot;
        }
      });
    }

    for (const auto* node : graph_->pairwise()) {
      node->factor.reset_primal();
      const auto* left = node->unary0;
      const auto* right = node->unary1;
      node->factor.primal() = std::tuple(left->factor.primal(), right->factor.primal());
    }
  }

  index get(index idx) const
  {
    assert(idx >= 0 && idx < data_.size());
    return data_[idx];
  }

  void set(index idx, index label)
  {
    assert(idx >= 0 && idx < data_.size());
    data_[idx] = label;
  }

  const auto& data() const { return data_; }
  const auto& assignment() const { return assignment_; }

protected:
  const graph_type* graph_;
  std::vector<index> data_;
  std::vector<index> assignment_;
};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
