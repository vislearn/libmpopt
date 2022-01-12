#ifndef LIBMPOPT_QAP_QPBO_HPP
#define LIBMPOPT_QAP_QPBO_HPP

namespace mpopt {
namespace qap {

#ifdef ENABLE_QPBO

template<typename ALLOCATOR>
class qpbo_model_builder {
public:
  using allocator_type = ALLOCATOR;
  using graph_type = graph<allocator_type>;
  using unary_node_type = typename graph_type::unary_node_type;
  using uniqueness_node_type = typename graph_type::uniqueness_node_type;
  using pairwise_node_type = typename graph_type::pairwise_node_type;
  using qpbo_type = qpbo::QPBO<cost>;

  static constexpr cost INFTY = 1e20;

  struct qpbo_unary_type {
    index label0, label1;
    int node;

    qpbo_unary_type(index label0, index label1)
    : label0(label0)
    , label1(label1)
    {
    }

    bool has_two() const {
      return label0 != label1;
    }
  };

  qpbo_model_builder(const graph_type& graph)
  : qpbo_(graph.unaries().size(), graph.uniqueness().size() + graph.pairwise().size())
  , graph_(&graph)
  , enable_weak_persistency_(true)
  , enable_probe_(false)
  , enable_improve_(true)
  , unaries_(graph.unaries().size())
  , uniqueness_(graph.uniqueness().size())
  , pairwise_(graph.pairwise().size())
  {
  }

  void reset()
  {
    qpbo_.Reset();
    unaries_.assign(graph_->unaries().size(), {});
    uniqueness_.assign(graph_->uniqueness().size(), {});
    pairwise_.assign(graph_->pairwise().size(), {});
    mapping_.clear();
  }

  void set_constant(const cost constant)
  {
    // TODO: Implement this.
  }

  void add_factor(const unary_node_type* node, index label0, index label1)
  {
    assert(node != nullptr);
    assert(label0 >= 0 && label0 < node->factor.size());
    assert(label1 >= 0 && label1 < node->factor.size());
    assert(!unaries_[node->idx]);

    auto& unary = unaries_[node->idx].emplace(label0, label1);
    unary.node = qpbo_.AddNode();
    const auto e0 = node->factor.get(label0);
    const auto e1 = unary.has_two() ? node->factor.get(label1) : INFTY;
    qpbo_.AddUnaryTerm(unary.node, e0, e1);
  }

  void add_factor(const uniqueness_node_type* node)
  {
    assert(!uniqueness_[node->idx]);
    uniqueness_[node->idx] = true;

    qpbo_unary_type* first = nullptr;
    qpbo_unary_type* second = nullptr;
    index first_label = -1, second_label = -1;

    // FIXME: This clean up this mess.
    node->traverse_unaries([&](const auto edge, const index slot) {
      auto& tmp = unaries_[edge.node->idx];
      if (tmp) {
        if (tmp->label0 == edge.slot || tmp->label1 == edge.slot) {
          if (first == nullptr) {
            first = &*tmp;
            first_label = edge.slot;
          } else {
            assert(second == nullptr);
            second = &*tmp;
            second_label = edge.slot;
          }
        }
      }
    });

    if (first != nullptr && second != nullptr) {
      auto cost = [](bool x) { return x ? INFTY : 0; };
      const auto e00 = cost(first->label0 == first_label && second->label0 == second_label);
      const auto e01 = cost(first->label0 == first_label && second->label1 == second_label);
      const auto e10 = cost(first->label1 == first_label && second->label0 == second_label);
      const auto e11 = cost(first->label1 == first_label && second->label1 == second_label);
      qpbo_.AddPairwiseTerm(first->node, second->node, e00, e01, e10, e11);
    }
  }

  void add_factor(const pairwise_node_type* node)
  {
    assert(!pairwise_[node->idx]);
    pairwise_[node->idx] = true;

    const auto& left = unaries_[node->unary0->idx];
    const auto& right = unaries_[node->unary1->idx];
    assert(left && right);

    const auto e00 = node->factor.get(left->label0, right->label0);
    const auto e01 = node->factor.get(left->label0, right->label1);
    const auto e10 = node->factor.get(left->label1, right->label0);
    const auto e11 = node->factor.get(left->label1, right->label1);

    qpbo_.AddPairwiseTerm(left->node, right->node, e00, e01, e10, e11);
  }

  auto enable_weak_persistency() { return enable_weak_persistency_; }
  void enable_weak_persistency(bool v=true) { enable_weak_persistency_ = v; }

  auto enable_probe() { return enable_probe_; }
  void enable_probe(bool v=true) { enable_probe_ = v; }

  auto enable_improve() { return enable_improve_; }
  void enable_improve(bool v=true) { enable_improve_ = v; }

  void finalize()
  {
    qpbo_.MergeParallelEdges();
  }

  void optimize()
  {
#ifndef NDEBUG
    std::cout << "Running QPBO ...";
#endif
    qpbo_.Solve();

    if (enable_weak_persistency_) {
#ifndef NDEBUG
      std::cout << " Weak Persistency ...";
#endif
      qpbo_.ComputeWeakPersistencies();
    }

    if (enable_probe_) {
#ifndef NDEBUG
      std::cout << " Probe ...";
#endif
      qpbo_type::ProbeOptions probe_options;
      mapping_.resize(qpbo_.GetNodeNum());
      qpbo_.Probe(mapping_.data(), probe_options);
    }

    if (enable_improve_) {
#ifndef NDEBUG
      std::cout << " Improve ... ";
#endif
      auto prev = qpbo_.ComputeTwiceEnergy();
      for (int i = 0; i < 5; ++i) {
        qpbo_.Improve();
#ifndef NDEBUG
        std::cout << ".";
#endif
        const auto curr = qpbo_.ComputeTwiceEnergy();
        if (curr < prev) {
          prev = curr;
          i = 0;
        }
      }
    }

#ifndef NDEBUG
    std::cout << std::endl;
#endif
  }

  void update_primals()
  {
    for (const auto* node : graph_->unaries()) {
      const auto& info = unaries_[node->idx];
      if (info) {
        node->factor.reset_primal();
        int l;
        if (mapping_.size() > 0) {
          const int m = mapping_[info->node];
          const int n = m / 2;
          l = (qpbo_.GetLabel(n) + 1) % 2;
          node->factor.primal() = l == 1 ? info->label1 : info->label0;
        } else {
          l = qpbo_.GetLabel(info->node);
        }
        node->factor.primal() = l == 1 ? info->label1 : info->label0;
      }
    }

    for (const auto* node : graph_->uniqueness()) {
      if (uniqueness_[node->idx]) {
        node->factor.primal() = node->unaries.size();
        node->traverse_unaries([&](const auto& edge, const index slot) {
          if (edge.node->factor.primal() == edge.slot)
            node->factor.primal() = slot;
        });
      }
    }

    for (const auto* node : graph_->pairwise()) {
      if (pairwise_[node->idx]) {
        node->factor.reset_primal();
        const auto* left = node->unary0;
        const auto* right = node->unary1;
        node->factor.primal() = std::tuple(left->factor.primal(), right->factor.primal());
      }
    }
  }


protected:
  qpbo_type qpbo_;
  const graph_type* graph_;
  std::vector<std::optional<qpbo_unary_type>> unaries_;
  std::vector<bool> uniqueness_;
  std::vector<bool> pairwise_;
  std::vector<int> mapping_;
  bool enable_weak_persistency_;
  bool enable_probe_;
  bool enable_improve_;
};

#endif

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
