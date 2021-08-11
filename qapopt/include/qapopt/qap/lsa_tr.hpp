#ifndef LIBQAPOPT_QAP_LSA_TR_HPP
#define LIBQAPOPT_QAP_LSA_TR_HPP

namespace qapopt {
namespace qap {

#ifdef ENABLE_LSATR

template<typename ALLOCATOR>
class lsatr_model_builder {
public:
  using allocator_type = ALLOCATOR;
  using graph_type = graph<allocator_type>;
  using unary_node_type = typename graph_type::unary_node_type;
  using uniqueness_node_type = typename graph_type::uniqueness_node_type;
  using pairwise_node_type = typename graph_type::pairwise_node_type;

  static constexpr cost INFTY = 1e20;

  struct lsatr_unary_type {
    index label0, label1;
    int unary;

    lsatr_unary_type(index label0, index label1)
    : label0(label0)
    , label1(label1)
    {
    }

    bool has_two() const {
      return label0 != label1;
    }
  };

  lsatr_model_builder(const graph_type& graph)
  {
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

    lsatr_unary_type unary(label0, label1);
    const auto e0 = node->factor.get(label0);
    const auto e1 = unary.has_two() ? node->factor.get(label1) : INFTY;
    unary.unary = lsatr_.add_unary(e0, e1);

    const bool did_insert = unaries_.emplace(std::make_pair(node, unary)).second;
    assert(did_insert);
  }

  void add_factor(const uniqueness_node_type* node)
  {
    bool did_insert = uniqueness_.insert(node).second;
    assert(did_insert);

    lsatr_unary_type* first = nullptr;
    index first_label;
    lsatr_unary_type* second = nullptr;
    index second_label;

    node->traverse_unaries([&](const auto edge, const index slot) {
      auto it = unaries_.find(edge.node);
      if (it != unaries_.end()) {
        if (it->second.label0 == edge.slot || it->second.label1 == edge.slot) {
          if (first == nullptr) {
            first = &it->second;
            first_label = edge.slot;
          } else {
            assert(second == nullptr);
            second = &it->second;
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
      lsatr_.add_pairwise(first->unary, second->unary, e00, e01, e10, e11);
    }
  }

  void add_factor(const pairwise_node_type* node)
  {
    auto did_insert = pairwise_.insert(node).second;
    assert(did_insert);

    const auto& left = unaries_.at(node->unary0);
    const auto& right = unaries_.at(node->unary1);

    const auto e00 = node->factor.get(left.label0, right.label0);
    const auto e01 = node->factor.get(left.label0, right.label1);
    const auto e10 = node->factor.get(left.label1, right.label0);
    const auto e11 = node->factor.get(left.label1, right.label1);

    lsatr_.add_pairwise(left.unary, right.unary, e00, e01, e10, e11);
  }

  void finalize()
  {
  }

  void optimize()
  {
    sol_ = lsatr_.optimize();
  }

  void update_primals()
  {
    for (const auto& [node, info] : unaries_) {
      node->factor.reset_primal();
      const int l = sol_[info.unary];
      node->factor.primal() = l == 1 ? info.label1 : info.label0;
    }

    for (const auto* node : uniqueness_) {
      node->factor.primal() = node->unaries.size();
      node->traverse_unaries([&](const auto& edge, const index slot) {
        if (edge.node->factor.primal() == edge.slot)
          node->factor.primal() = slot;
      });
    }

    for (const auto* node : pairwise_) {
      node->factor.reset_primal();
      const auto* left = node->unary0;
      const auto* right = node->unary1;
      node->factor.primal() = std::tuple(left->factor.primal(), right->factor.primal());
    }
  }

protected:
  lsatr::lsa_tr lsatr_;
  const graph_type* graph_;
  std::unordered_map<const unary_node_type*, lsatr_unary_type> unaries_;
  std::unordered_set<const uniqueness_node_type*> uniqueness_;
  std::unordered_set<const pairwise_node_type*> pairwise_;
  std::vector<char> sol_;
};

#endif

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
